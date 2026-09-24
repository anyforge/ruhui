"""Anchor targets: the frozen base model's zero-shot distribution over each training question's options, read from
next-token letter logits (the same readout as scripts/base_mmlu_probe.py). Used by `kev.train --anchor --anchor_w` as a
KL target so the fine-tune keeps what the base already knows (PLAN.md: fine-tuning erodes base capability).

    uv run python -m kev.anchors --base Qwen/Qwen3-4B-Base --suite evals/v6/decision-v6 --out runs/anchors/v6-4b.json

Targets are keyed by record id and question id, values keyed by option key, so they align with any option permutation
the trainer applies; questions whose option set changed (none-of-the-above inserted) are skipped at training time.
No Jev or other external model is involved: the teacher is the student's own base.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ruhui.llm.api import question_keys
from ruhui.llm.device import default_device
from ruhui.llm.suite import load_split, write_json

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def question_prompt(state, q):
    text = state if isinstance(state, str) else " ".join(f"{k}: {v}" for k, v in state.items()) if isinstance(state, dict) else json.dumps(state)
    keys = question_keys(q["type"], q.get("criteria"))
    if q["type"] == "noul": texts = ["No", "Yes"]
    elif q["type"] == "score": texts = list(q["criteria"])
    else: texts = [v if v is not None else k for k, v in q["criteria"].items()]
    prompt = f"{text}\n{q['instructions']}\n" + "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts)) + "\nAnswer:"
    return prompt, keys


def build(base, suite, out, split="train", device=None, revision=None, max_options=26, batch=16):
    device = device or default_device()
    tok = AutoTokenizer.from_pretrained(base, revision=revision); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16 if device != "cpu" else torch.float32).to(device).eval()
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    records = load_split(suite, split)
    jobs = [(r["_meta"]["id"], qid, *question_prompt(r["state"], q)) for r in records for qid, q in r["questions"].items() if len(q.get("criteria", [0, 1])) <= max_options or q["type"] == "noul"]
    targets, skipped = {}, 0
    with torch.no_grad():
        for i in range(0, len(jobs), batch):
            chunk = jobs[i : i + batch]
            enc = tok([p for _, _, p, _ in chunk], return_tensors="pt", padding=True, truncation=True, max_length=3072).to(device)
            logits = model(**enc).logits[:, -1].float()
            for (rid, qid, _, keys), row in zip(chunk, logits):
                if len(keys) > len(letter_ids): skipped += 1; continue
                p = torch.softmax(row[letter_ids[: len(keys)]], -1).tolist()
                targets.setdefault(rid, {})[qid] = dict(zip(keys, p))
            if (i // batch) % 50 == 0: print(f"anchors {i}/{len(jobs)}", flush=True)
    meta = {"base": base, "revision": revision, "suite": str(suite), "split": split, "readout": "zero-shot next-token letter logits",
            "records": len(targets), "questions": sum(len(v) for v in targets.values()), "skipped_over_26_options": skipped}
    write_json(out, {"_meta": meta, "targets": targets})
    print(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--suite", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--device"); ap.add_argument("--revision")
    a = ap.parse_args()
    if Path(a.out).exists(): raise FileExistsError(a.out)
    build(a.base, a.suite, a.out, a.split, a.device, a.revision)


if __name__ == "__main__":
    main()
