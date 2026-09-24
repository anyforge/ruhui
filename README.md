[![PyPI](https://img.shields.io/pypi/v/ruhui.svg)](https://pypi.org/project/ruhui/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-anyforge%2Fruhui-blue)](https://huggingface.co/anyforge/ruhui)
[![ModelScope](https://img.shields.io/badge/ModelScope-anyforge%2Fruhui-624aff.svg)](https://modelscope.cn/models/anyforge/ruhui)
[![GitHub](https://img.shields.io/badge/GitHub-anyforge%2Fruhui-181717.svg?logo=github)](https://github.com/anyforge/ruhui)

# Ruhui · 如晦

**A non-autoregressive System 1 decision engine for Chinese & multilingual text, with calibrated probabilities.**
**非自回归 System 1 决策引擎（中文/多语言），带校准概率。**

命名取自「房谋杜断」的杜如晦（字克明），「晦」音近「hui」。房玄龄善谋、杜如晦善断——Ruhui 取「断」之意：System 1 快速决策，不生成文本、无可解析输出、因此无幻觉。

架构参照 [Laya](https://github.com/NandhaKishorM/laya)（Apache 2.0），提供**两个后端**：

| 后端 | 底座 | 特点 | 适用场景 |
|---|---|---|---|
| **bert**（原 ruhui）| mmBERT-base（322M）| 33ms 级、CPU 可跑、中英双语 | 高吞吐、低延迟、资源受限 |
| **llm**（新增）| Qwen3.5-0.8B + LoRA + PointerHead | 大模型通用性更强 | 复杂决策、泛化优先 |

---

## Architecture · 架构

三种决策原语，单次前向传播并行输出：

| Primitive · 原语 | Output · 输出 |
|---|---|
| **choice** | top label + full probability distribution + confidence |
| **score** | expected level on an ordinal rubric |
| **noul** | calibrated P(true) |

---

## Installation · 安装

```bash
pip install ruhui
```

Python 3.10+。依赖：`torch`、`transformers`、`safetensors`、`huggingface_hub`、`numpy`。
LLM 后端额外需要 `peft`。

---

## Quick Start · 快速开始

### bert 后端（原用法，不变）

```python
import ruhui

agent = ruhui.load("anyforge/ruhui")   # 从 HF/ModelScope 拉取，或本地目录
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

### llm 后端（大模型，通用性更强）

```python
from ruhui.llm import LLMAgent

# 合并后的完整模型（自包含，无需 base_dir）
agent = LLMAgent(checkpoint_dir="anyforge/ruhui/0.8B")

result = agent.predict(
    {"message": "我被重复扣款了，请退款"},
    {"intent": {"type": "choice", "instructions": "客户想做什么？",
                "criteria": {"refund": "退款", "billing": "账单"}}},
)
print(result["answers"])
```

---

## Fine-Tuning · 微调

### llm 后端微调（KEV 式：Causal LM + LoRA + PointerHead）

```bash
# 1. 软标签 → KEV 格式训练数据
python scripts/convert_to_kev.py \
  --soft_dir <soft_label_dir> --out datas/train.jsonl

# 2. 在原作者权重基础上微调（delta 模式）
python scripts/finetune.py \
  --data datas/train_final.jsonl \
  --base models/Qwen3.5-0.8B-Base \
  --init_from models/kev-0.8b \
  --out runs/ruhui-0.8b \
  --epochs 2 --device cuda

# 3. 断点续跑
python scripts/finetune.py ... --resume

# 4. 合并 LoRA 成完整模型
python scripts/merge_model.py \
  --checkpoint runs/ruhui-0.8b \
  --base models/Qwen3.5-0.8B-Base \
  --out runs/ruhui-0.8b-merged
```

### bert 后端微调（Laya 式：encoder + 决策头）

```bash
python scripts/train.py \
  --model_dir <base_model_dir> \
  --train_items <train_items.pt> \
  --output_dir <output_dir> \
  --epochs 4
```

---

## Repository Layout · 目录结构

```
ruhuipro/
  ruhui/
    bert/           # bert 后端（原 ruhui：agent/router/common/presets/...）
    llm/            # llm 后端（KEV 式：model/api/checkpoint/train/data/agent）
  models/           # 本地底座 + adapter
  datas/            # 转换后的训练数据
  scripts/
    convert_to_kev.py      # 软标签 → KEV 格式
    finetune.py            # llm 微调启动（支持 --init_from / --resume）
    merge_model.py         # LoRA 合并导出
    train.py               # bert 后端微调
```

---

## Model Repositories · 模型仓库

- Hugging Face: `anyforge/ruhui`
  - 根目录 = bert 后端模型（322M）
  - `0.8B/` 子目录 = LLM 后端 0.8B（Qwen3.5-0.8B 合并模型）
  - `4B/` 子目录 = LLM 后端 4B（计划中）
- ModelScope: `anyforge/ruhui`（同上）

> bert 用 `ruhui.load("anyforge/ruhui")` 加载；
> LLM 用 `ruhui.llm.LLMAgent(checkpoint_dir="anyforge/ruhui/0.8B")` 加载。

## License

Apache 2.0 (inherited from Laya). Developed by AnyForge.
Apache 2.0（参照 Laya）。Developed by AnyForge。
