[![PyPI](https://img.shields.io/pypi/v/ruhui.svg)](https://pypi.org/project/ruhui/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-anyforge%2Fruhui-blue)](https://huggingface.co/anyforge/ruhui)
[![ModelScope](https://img.shields.io/badge/ModelScope-anyforge%2Fruhui-624aff.svg)](https://modelscope.cn/models/anyforge/ruhui)

# Ruhui · 如晦

**A non-autoregressive System 1 decision engine for Chinese & multilingual text, with calibrated probabilities.**

Named after Du Ruhui (杜如晦, courtesy name Keming 克明) of the legendary *Fang Mou Du Duan* (房谋杜断) pair — Fang Xuanling was the strategist, Du Ruhui the decisive judge. *Ruhui* inherits the "decisive" half: it makes a fast System 1 decision, generates no text, has nothing to parse, and therefore cannot hallucinate.

---

## What it is

Ruhui answers **typed questions** — `choice`, `score`, `noul` (yes/no) — over any state (text, email, ticket, or JSON) in a **single forward pass**, returning a **calibrated probability** for every option. No text generation, no parsing, no hallucination.

Two backends share the same interface:

| Backend | Architecture | Size | Latency | Strength |
|---|---|---|---|---|
| **bert** | bidirectional encoder (mmBERT-base) + decision head | 322M | ~33 ms | fast, CPU-friendly, Chinese/English |
| **llm** | Causal LM (Qwen3.5) + LoRA + PointerHead | 0.8B+ | hundreds of ms | stronger generalization |

---

## Installation

```bash
pip install ruhui
```

Python 3.10+. Core deps: `torch`, `transformers`, `safetensors`, `huggingface_hub`, `numpy`. The LLM backend additionally needs `peft`.

---

## Quick Start

### bert backend (the original, unchanged)

```python
import ruhui

agent = ruhui.load("/path/to/anyforge/ruhui")   # hub, or a local directory

result = agent.predict(
    {"message": "我被重复扣款了，请退款"},
    {
        "intent": {"type": "choice", "instructions": "客户想做什么？",
                   "criteria": {"refund": "退款", "technical": "技术问题", "billing": "账单咨询"}},
        "churn_risk": {"type": "noul", "instructions": "客户是否威胁要离开？"},
    },
)
print(result["answers"])
```

### llm backend (larger model, stronger generalization)

```python
from ruhui.llm import LLMAgent

# a merged (self-contained) model — no base_dir needed
agent = LLMAgent(checkpoint_dir="/path/to/anyforge/ruhui/0.8B")

result = agent.predict(
    {"message": "我被重复扣款了，请退款"},
    {"intent": {"type": "choice", "instructions": "客户想做什么？",
                "criteria": {"refund": "退款", "billing": "账单"}}},
)
print(result["answers"])
```

---

## How the two backends work

### bert backend — encoder + decision head

A bidirectional encoder reads the whole input, then a 2-layer decision head scores each option at its own `[MASK]` slot, all in parallel. Probabilities come from a softmax over those option slots, trained with **RLCD** (reinforcement learning from strictly-proper-scoring-rule rewards) so the reported confidence is statistically meaningful.

```
[CLS] question + [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
   → bidirectional encoder
   → gather the [MASK] slot vectors
   → parallel scorer → softmax → calibrated probabilities
```

### llm backend — Causal LM + PointerHead (KEV-style)

A frozen causal LM runs **prefill-only** (never generates tokens). Each question becomes a branch sharing one state prefix, isolated by a block-causal mask. A pointer head then reads the `<decide>` position and "points" at the option boundary tokens — the attention scores become the option probabilities.

```
[state] [q: instr <opt>opt A</opt> <opt>opt B</opt> <decide>]
   → Causal LM (prefill only)
   → PointerHead: q(decide) · k(option) → logits → softmax
```

The LoRA adapter is folded into the base weights at inference (or merged permanently with `merge_model.py`).

---

## Decision primitives

| Primitive | Output |
|---|---|
| `choice` | top label + full probability distribution + confidence |
| `score` | expected level on an ordinal rubric |
| `noul` | calibrated P(true) |

Confidence is normalized entropy (`1 − H(p)/log K`), so it is safe to gate on:

```python
if conf >= 0.85:
    route_automatically(dept)   # high confidence
else:
    escalate_to_human(dept)     # low confidence
```

---

## Fine-tuning

### llm backend (KEV-style)

```bash
# 1. convert soft labels to KEV-format training data
python scripts/convert_to_kev.py --soft_dir <dir> --out datas/train.jsonl

# 2. fine-tune from an existing checkpoint (delta mode)
python scripts/finetune.py \
  --data datas/train.jsonl \
  --base Qwen/Qwen3.5-0.8B-Base \
  --init_from anyforge/ruhui/0.8B \
  --out runs/ruhui-0.8b \
  --epochs 2 --device cuda

# 3. resume if interrupted
python scripts/finetune.py ... --resume

# 4. merge LoRA into the base weights (bf16 halves the size)
python scripts/merge_model.py \
  --checkpoint runs/ruhui-0.8b \
  --base Qwen/Qwen3.5-0.8B-Base \
  --out runs/ruhui-0.8b-merged \
  --dtype bf16
```

### bert backend (Laya-style)

```bash
python scripts/train.py \
  --model_dir <base_model_dir> \
  --train_items <train_items.pt> \
  --output_dir <output_dir> \
  --epochs 4
```

---

## Repository layout

```
ruhuipro/
  ruhui/
    bert/           # encoder backend (agent / router / common / presets / ...)
    llm/            # LLM backend (model / api / checkpoint / train / data / agent)
  scripts/
    convert_to_kev.py      # soft labels → KEV format
    finetune.py            # LLM fine-tune (--init_from / --resume)
    merge_model.py         # LoRA merge + dtype control
    train.py               # bert fine-tune
  tests/
```

---

## Model repositories

- Hugging Face: `anyforge/ruhui` (root = bert model; `0.8B/` = LLM 0.8B merged model)
- ModelScope: `anyforge/ruhui` (same layout)

---

## Acknowledgments

- **Laya** ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), Apache 2.0) — the non-autoregressive System 1 decision paradigm and RLCD training that the bert backend is forked from.
- **KEV** ([jaredpalmer/kev](https://github.com/jaredpalmer/kev), Apache 2.0) — the Causal LM + LoRA + PointerHead architecture that the llm backend is built on.

## License

Apache 2.0. Developed by AnyForge.
