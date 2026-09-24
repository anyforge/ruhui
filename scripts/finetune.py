"""ruhuipro 微调启动脚本：用 KEV 式 LLM 后端 + 你的软标签数据，微调 0.8B。

封装 ruhui.llm.train 的复杂参数，一条命令开跑。

用法:
  python scripts/finetune.py \
    --data datas/train_final.jsonl \
    --base models/Qwen3.5-0.8B-Base \
    --out runs/ruhui-0.8b \
    --epochs 2

关键参数说明（全字段）:
  --data        训练数据（KEV 格式 JSONL，由 convert_to_kev.py 生成）
  --base        底座目录（本地路径）或 hub id
  --out         输出目录（保存 adapter + head.pt）
  --epochs      训练轮数（KEV 0.8B 配方用 2）
  --lr          学习率（0.8B 配方 1e-4）
  --batch       records per forward（0.8B 配方 8）
  --accum       梯度累积步数
  --device      cuda / mps / cpu
  --init_from   可选：从一个已有 checkpoint 继续微调（delta 模式，如 models/kev-0.8b）
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="ruhuipro LLM 微调")
    ap.add_argument("--data", required=True, help="KEV 格式训练数据 JSONL")
    ap.add_argument("--base", default="models/Qwen3.5-0.8B-Base", help="底座目录（本地）或 hub id")
    ap.add_argument("--out", default="runs/ruhui-0.8b")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--init_from", default="", help="可选 delta 微调起点（本地 checkpoint 目录或 hub id）")
    ap.add_argument("--resume", action="store_true", help="从 out/train_state.pt 断点续跑")
    ap.add_argument("--max_state", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 拼参数调用 ruhui.llm.train.main
    from ruhui.llm.train import main as train_main

    argv = [
        "--data", args.data,
        "--base", args.base,
        "--out", args.out,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch", str(args.batch),
        "--accum", str(args.accum),
        "--lora", str(args.lora),
        "--dtype", args.dtype,
        "--max_state", str(args.max_state),
        "--seed", str(args.seed),
    ]
    if args.device:
        argv += ["--device", args.device]
    if args.init_from:
        argv += ["--init_from", args.init_from]
    if args.resume:
        argv += ["--resume"]

    import sys as _sys
    _sys.argv = ["finetune.py"] + argv
    train_main()


if __name__ == "__main__":
    main()
