"""把 ruhui 软标签（soft_*.jsonl）转成 KEV 微调格式（train.jsonl）。

输入（soft_*.jsonl 每行）:
  {id, source_dataset, text, qtype, criteria, instruction,
   soft_label: {probabilities: {key: p}, label: argmax}}

输出（KEV train.jsonl 每行）:
  {state, questions: {qid: {type, instructions, criteria, label, target}}}

映射:
  text          -> state
  instruction   -> instructions
  qtype         -> type
  criteria      -> criteria（choice=dict, score=list, noul=None）
  soft_label.label -> label（choice=选项名, noul=true/false, score=等级索引int）
  soft_label.probabilities -> target（软标签，按 key 对齐）

用法:
  python convert_to_kev.py --soft_dir <soft_dir> --out datas/train.jsonl
"""
import argparse
import glob
import json
import os


def convert_one(o):
    """单条 soft 标签 -> KEV 记录，失败返回 None。"""
    text = o.get("text", "").strip()
    qtype = o.get("qtype")
    instruction = o.get("instruction", "")
    criteria = o.get("criteria")
    soft = o.get("soft_label", {})
    probs = soft.get("probabilities", {})
    label = soft.get("label")

    if not text or not probs or qtype not in ("choice", "noul", "score"):
        return None

    # 先算 criteria keys（choice）/ 等级数（score），供 label 校验和 target 对齐用
    if qtype == "choice":
        keys = list(criteria.keys()) if isinstance(criteria, dict) else []
        if not keys:
            return None
    elif qtype == "noul":
        keys = ["false", "true"]
    else:
        keys = [str(i) for i in range(len(criteria))] if isinstance(criteria, list) else []
        if not keys:
            return None

    # label 转 KEV 期望的类型（并校验 label 是否在 keys 内，越界用 argmax 重算）
    if qtype == "choice":
        label_val = str(label) if label is not None else None
        if label_val not in keys:
            # 脏数据：label 不在 criteria 里（LLM 幻觉），用 target 的 argmax 重算
            label_val = max(probs, key=lambda k: probs.get(k, 0.0)) if probs else keys[0]
    elif qtype == "noul":
        label_val = bool(label) if isinstance(label, bool) else str(label).lower() in ("true", "1")
    else:  # score
        try:
            label_val = int(label)
        except (TypeError, ValueError):
            label_val = int(max(probs, key=lambda k: probs.get(k, 0.0)))
        if not (0 <= label_val < len(keys)):
            label_val = int(max(probs, key=lambda k: probs.get(k, 0.0))) if probs else 0

    # target（软标签）：按 criteria key 对齐，归一化
    if qtype == "choice":
        t = [float(probs.get(k, 0.0)) for k in keys]
    elif qtype == "noul":
        t = [float(probs.get("false", 0.0)), float(probs.get("true", 0.0))]
    else:  # score
        n = len(criteria) if isinstance(criteria, list) else 0
        if not n:
            return None
        t = [float(probs.get(str(i), 0.0)) for i in range(n)]

    s = sum(t)
    if s <= 0:
        target = None  # 无效概率，退化为纯硬标签
    else:
        if qtype == "choice":
            tkeys = keys
        elif qtype == "noul":
            tkeys = ["false", "true"]
        else:
            tkeys = [str(i) for i in range(len(criteria))]
        target = {k: v / s for k, v in zip(tkeys, t)}

    qdef = {"type": qtype, "instructions": instruction, "label": label_val}
    if qtype == "choice":
        qdef["criteria"] = criteria
        if target is not None:
            qdef["target"] = target
    elif qtype == "score":
        qdef["criteria"] = criteria
        if target is not None:
            qdef["target"] = target
    else:  # noul
        if criteria:
            qdef["criteria"] = criteria
        if target is not None:
            qdef["target"] = target

    return {"state": text, "questions": {"q": qdef}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_dir", required=True, help="soft_*.jsonl 所在目录")
    ap.add_argument("--out", default="datas/train.jsonl", help="输出 KEV train.jsonl")
    ap.add_argument("--max_per_file", type=int, default=0, help="每文件最多转几条（0=全部）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.soft_dir, "soft_*.jsonl")))
    if not files:
        raise SystemExit(f"未找到 soft_*.jsonl in {args.soft_dir}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for f in files:
            n = 0
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    rec = convert_one(o)
                    if rec is None:
                        skipped += 1
                        continue
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
                    n += 1
                    if args.max_per_file and n >= args.max_per_file:
                        break
    print(f"转换完成: {total} 条写入 {args.out}，跳过 {skipped} 条")


if __name__ == "__main__":
    main()
