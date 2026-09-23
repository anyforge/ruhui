"""ruhui 微调训练脚本（参照 Laya 的 RLCD + 软蒸馏 + 温度校准）。

流程：
  1. 加载 multilingual 底座（mmBERT-base + DecisionModel）
  2. 读训练 items（由 prepare_train_data.py 生成）
  3. RLCD（恰当评分规则策略梯度）+ 软交叉熵蒸馏 联合训练
  4. 训练后温度校准（按 qtype 拟合）
  5. 保存为 ruhui checkpoint（rl_agent_config.json + model.safetensors + encoder/ + tokenizer/）

用法（单卡）:
  python3 scripts/train.py \
    --model_dir <base_model_dir> \
    --train_items <train_items.pt> \
    --output_dir <output_dir> \
    --epochs 4 --micro_batch 8 --grad_accum 4

可选 DDP: torchrun --nproc_per_node=2 scripts/train.py ...
"""
import os
import sys
import json
import time
import argparse
import random

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ruhui.common import build_model, proper_reward, QTYPES


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att,
        "marker_pos": mpos, "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    """LBFGS 拟合单温度。sel: [(logits, target), ...]"""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--train_items", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro_batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4, help="GRPO 探索样本数")
    parser.add_argument("--lr_encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr_head", type=float, default=1.0e-4)
    parser.add_argument("--sigma_start", type=float, default=0.4)
    parser.add_argument("--sigma_end", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--head_max_len", type=int, default=256)
    args = parser.parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # 加载底座
    with open(os.path.join(args.model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len

    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(args.model_dir, "encoder"))
    weights = load_file(os.path.join(args.model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.to(device)

    # 训练 items
    all_items = torch.load(args.train_items, weights_only=False)
    print(f"训练 items: {len(all_items)}")

    # 参数分组
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)

    MICRO, ACCUM, GROUP = args.micro_batch, args.grad_accum, args.group_size
    total_updates = (len(all_items) // (MICRO * ACCUM)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        random.shuffle(all_items)
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress

        for b_idx in range(0, len(all_items), MICRO):
            chunk = all_items[b_idx:b_idx + MICRO]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, act = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)

            # 1. GRPO 探索：G 个噪声分布（零均值投影）
            eps = torch.randn((GROUP,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k.clamp(min=1)) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. 恰当评分规则 reward（w_sph=0.75 强调软目标匹配）
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. 策略梯度 + 软交叉熵
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / ACCUM

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % ACCUM == 0 or (b_idx + MICRO) >= len(all_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * ACCUM
            n_batches += 1

        print(f"Epoch {epoch+1}/{args.epochs} 完成, 平均 loss {epoch_loss/max(1,n_batches):.4f}, "
              f"耗时 {time.time()-t0:.0f}s", flush=True)

        # 滚动 checkpoint
        ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
        os.makedirs(ckpt_dir, exist_ok=True)
        sd = {kk: vv.half().contiguous().cpu() for kk, vv in model.state_dict().items()}
        save_file(sd, os.path.join(ckpt_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
        tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
        with open(os.path.join(ckpt_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # ---- 温度校准 ----
    print("\n拟合温度...")
    model.eval()
    calib = all_items[::15][:400]
    calib_preds = []
    with torch.no_grad():
        for c_idx in range(0, len(calib), 16):
            cb = collate_train_batch(calib[c_idx:c_idx+16], tok.pad_token_id)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                                 cb["marker_pos"].to(device), cb["marker_mask"].to(device), cb["qtype"].to(device))
            l_np = l_sub.float().cpu().numpy()
            for r_i, it in enumerate(calib[c_idx:c_idx+16]):
                kk = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[r_i, :kk], it["target"]))

    fitted = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, t) for q_t, z, t in calib_preds if q_t == qt]
        if sel:
            fitted[qt] = fit_one_temp(sel)
    print(f"温度 (choice, score, noul): {[round(t, 3) for t in fitted]}")

    # ---- 保存最终模型 ----
    os.makedirs(args.output_dir, exist_ok=True)
    sd = {kk: vv.half().contiguous().cpu() for kk, vv in model.state_dict().items()}
    save_file(sd, os.path.join(args.output_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
    tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "ruhui"
    cfg["temperature"] = fitted
    # 注意：cfg["encoder"] 保持原底座 mmBERT-base（encoder 是骨干架构，不是模型 repo）
    with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n模型保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
