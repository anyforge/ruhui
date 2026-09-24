"""把训练好的 LoRA checkpoint 合并成完整模型（adapter + head 折进底座）。

关键：adapter 加载必须和 ruhui.llm.checkpoint 的 _load_torch 一致——
用 DecisionModel(meta.base, lora=None) 得到 .model 部分的 lm，再 PeftModel.from_pretrained(lm, ...)
挂 adapter，最后 merge_and_unload。直接 AutoModelForCausalLM 会因层名前缀对不上而丢权重。

用法:
  python scripts/merge_model.py \
    --checkpoint runs/ruhui-0.8b \
    --base models/Qwen3.5-0.8B-Base \
    --out runs/ruhui-0.8b-merged

合并后:
  merged/ 含 model.safetensors(完整权重) + config + tokenizer + head.pt
  用 ruhui.llm.LLMAgent(checkpoint_dir=merged, base_dir=merged) 加载（自动走 merged 分支）
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="训练产出目录（含 adapter_model.safetensors + head.pt）")
    ap.add_argument("--base", required=True, help="底座目录（本地路径）")
    ap.add_argument("--out", required=True, help="合并后输出目录")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"],
                    help="合并后权重的 dtype（默认 bf16，体积减半、概率误差<0.01；fp32 为训练主权重精度）")
    args = ap.parse_args()

    import torch
    from peft import PeftModel

    from ruhui.llm.model import DecisionModel, load_tokenizer
    from ruhui.llm.checkpoint import read_meta

    # 1. 读 checkpoint 元信息（base/head_dim/weights_dtype 等）
    meta = read_meta(args.checkpoint)

    # 2. 用和 _load_torch 一致的方式建 DecisionModel（lm = AutoModel(...).model，无 lora）
    base_name = os.path.abspath(args.base) if os.path.isdir(args.base) else meta.base
    local_base = os.path.isdir(args.base)
    tok = load_tokenizer(base_name, revision=None if local_base else meta.base_revision)
    # 合并后目标 dtype（用户可控，默认 bf16 减半体积）
    target_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print(f"加载底座 {base_name} (fp32，merge 在 fp32 下精确进行) ...")
    model = DecisionModel(
        base_name, tok, "cpu", lora=None,
        revision=None if local_base else meta.base_revision,
        head_dim=meta.head_dim,
        option_isolation=meta.option_isolation,
        dtype=torch.float32,   # 固定 fp32 加载，merge 的 W+=delta 需在 fp32 做
    )

    # 3. 挂 adapter + 合并（与 checkpoint._load_torch 完全一致）
    print(f"挂载 LoRA adapter {args.checkpoint} ...")
    model.lm = PeftModel.from_pretrained(model.lm, args.checkpoint, torch_device="cpu").to("cpu")
    model.lm = model.lm.merge_and_unload()
    if target_dtype != torch.float32:
        model.lm = model.lm.to(target_dtype)   # merge 完成后才转目标 dtype
        print(f"转换 dtype -> {args.dtype}")

    # 4. 保存完整模型（lm 部分 + head）
    os.makedirs(args.out, exist_ok=True)
    print(f"保存合并模型到 {args.out} ...")
    # 保存 lm（合并后的完整 backbone）
    model.lm.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # 5. 保存 head.pt（pointer head 决策头，含 meta）
    head_src = os.path.join(args.checkpoint, "head.pt")
    if os.path.exists(head_src):
        shutil.copy(head_src, os.path.join(args.out, "head.pt"))
        print(f"复制 head.pt -> {args.out}")
    else:
        print("警告: 未找到 head.pt")

    print(f"\n✅ 合并完成: {args.out}")
    print(f"   加载: ruhui.llm.LLMAgent(checkpoint_dir='{args.out}')")


if __name__ == "__main__":
    main()
