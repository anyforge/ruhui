"""把 datas/soft_*.jsonl 的软标签转成 ruhui 训练 items（.pt）。

软标签文件每行: {id, source_dataset, source_index, source_hash, text, qtype, criteria, instruction, soft_label}
soft_label: {probabilities: {key: p}, label: argmax}

转成训练 item: {ids, markers, qtype, target, label}
- ids/markers: build_sequence 的输出
- target: 归一化软概率（按 criteria/marker 顺序）
- label: argmax 的 marker 索引

用法:
  python3 scripts/prepare_train_data.py --model_dir <base_model_dir> --out <train_items.pt>
"""
import os
import sys
import json
import argparse
import glob

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ruhui.common import build_sequence, render_options, QTYPES


def prepare(model_dir, soft_files, out_path, max_len=1024, head_max_len=256):
    from transformers import AutoTokenizer
    from ruhui.agent import _fix_tokenizer_config

    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    items = []
    skipped = {"empty": 0, "no_probs": 0, "marker_mismatch": 0, "parse_err": 0}
    total = 0

    for sf in soft_files:
        with open(sf) as f:
            for line in f:
                total += 1
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    skipped["parse_err"] += 1
                    continue

                text = o.get("text", "").strip()
                qtype = o.get("qtype")
                soft = o.get("soft_label", {})
                probs = soft.get("probabilities", {})
                criteria = o.get("criteria")
                instruction = o.get("instruction", "")

                if not text:
                    skipped["empty"] += 1
                    continue
                if not probs:
                    skipped["no_probs"] += 1
                    continue

                # 构造 question 内部格式
                if qtype == "noul":
                    q = {"t": "noul", "ins": instruction, "crit": None}
                    # target 顺序: [false, true]
                    target = [float(probs.get("false", 0.0)), float(probs.get("true", 0.0))]
                elif qtype == "choice":
                    if not criteria:
                        skipped["no_probs"] += 1
                        continue
                    keys = list(criteria.keys())
                    q = {"t": "choice", "ins": instruction, "crit": criteria}
                    target = [float(probs.get(k, 0.0)) for k in keys]
                elif qtype == "score":
                    # score criteria 是 list
                    crit_list = criteria if isinstance(criteria, list) else []
                    if not crit_list:
                        skipped["no_probs"] += 1
                        continue
                    q = {"t": "score", "ins": instruction, "crit": crit_list}
                    target = [float(probs.get(str(i), 0.0)) for i in range(len(crit_list))]
                else:
                    skipped["no_probs"] += 1
                    continue

                # 归一化 target
                s = sum(target)
                if s <= 0:
                    target = [1.0 / len(target)] * len(target)
                else:
                    target = [v / s for v in target]

                # 过滤噪声 key（choice 里 criteria 之外的 key 已在上层处理，这里 target 已按 criteria 对齐）

                # build_sequence
                seq, markers = build_sequence(tok, text, q, max_len, head_max_len)
                k = len(render_options(q))
                if len(markers) != k:
                    skipped["marker_mismatch"] += 1
                    continue

                label = target.index(max(target))
                items.append({
                    "ids": seq,
                    "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "target": target,
                    "label": label,
                })

    torch.save(items, out_path)
    print(f"总行数 {total}, 有效 items {len(items)}")
    print(f"跳过: {dict(skipped)}")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="multilingual 底座目录（含 tokenizer）")
    parser.add_argument("--soft_dir", default=None, help="软标签目录（含 soft_*.jsonl）")
    parser.add_argument("--out", required=True, help="输出 .pt 路径")
    args = parser.parse_args()

    if not args.soft_dir:
        print("请用 --soft_dir 指定软标签目录")
        sys.exit(1)
    soft_files = sorted(glob.glob(os.path.join(args.soft_dir, "soft_*.jsonl")))
    if not soft_files:
        print(f"未找到软标签文件 in {args.soft_dir}")
        sys.exit(1)
    print(f"找到 {len(soft_files)} 个软标签文件")
    prepare(args.model_dir, soft_files, args.out)
