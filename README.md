[![PyPI](https://img.shields.io/pypi/v/ruhui.svg)](https://pypi.org/project/ruhui/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-anyforge%2Fruhui-blue)](https://huggingface.co/anyforge/ruhui)
[![ModelScope](https://img.shields.io/badge/ModelScope-anyforge%2Fruhui-624aff.svg)](https://modelscope.cn/models/anyforge/ruhui)
[![GitHub](https://img.shields.io/badge/GitHub-anyforge%2Fruhui-181717.svg?logo=github)](https://github.com/anyforge/ruhui)

# Ruhui · 如晦

**A non-autoregressive System 1 decision engine for Chinese & multilingual text, with calibrated probabilities.**

**非自回归 System 1 决策引擎（中文/多语言），带校准概率。**

Named after Du Ruhui (杜如晦, courtesy name Keming 克明) of the "Fang Mou Du Duan" (房谋杜断) pair — Fang Xuanling was the strategist, Du Ruhui the decisive judge. *Ruhui* inherits the "decisive" half: a fast System 1 decision maker that generates no text, has nothing to parse, and therefore cannot hallucinate.

命名取自「房谋杜断」的杜如晦（字克明），「晦」音近「hui」。房玄龄善谋、杜如晦善断——Ruhui 取「断」之意：System 1 快速决策，不生成文本、无可解析输出、因此无幻觉。

Architecture forked from [Laya](https://github.com/NandhaKishorM/laya) (Apache 2.0), with two key changes:
架构参照 [Laya](https://github.com/NandhaKishorM/laya)（Apache 2.0），在此基础上：

- **Chinese/multilingual backbone**: `mmBERT-base` (100+ languages) instead of English-only ModernBERT.
  **中文/多语言底座**：`mmBERT-base`（100+ 语言），而非英语-only 的 ModernBERT。
- **Bilingual soft-label fine-tuning**: 30+ domain datasets (intent / sentiment / safety / agent decision / tool-calling / …).
  **中英双语软标签微调**：覆盖 30+ 领域数据集（意图/情感/安全/智能体决策/工具调用等）。

---

## Architecture · 架构

```
bidirectional encoder (mmBERT-base)
  → type injection (choice / score / noul)
  → 2-layer decision head
  → parallel scoring at [MASK] slots
  → softmax distribution (calibrated probabilities via strictly-proper-scoring-rule RLCD)
```

Three decision primitives · 三种决策原语：

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

Python 3.10+. Dependencies: `torch`, `transformers`, `safetensors`, `huggingface_hub`, `numpy`.
依赖：`torch`、`transformers`、`safetensors`、`huggingface_hub`、`numpy`。

---

## Quick Start · 快速开始

```python
import ruhui

# Load the fine-tuned ruhui checkpoint (from the hub, or a local directory)
# 加载微调后的 ruhui checkpoint（从仓库拉取，或本地目录路径）
agent = ruhui.load("anyforge/ruhui")

# The Router also accepts a local path (skips hub download):
# Router 同样支持本地路径（不走仓库下载）：
router = ruhui.Router(model_path="./pretrained/ruhui")
result = router.predict(state, questions)

# Answer multiple structured questions in a single forward pass
# 一次前向传播回答多个结构化问题
result = agent.predict(
    {"message": "我被重复扣款了，请退款"},
    {
        "intent": {
            "type": "choice",
            "instructions": "客户想做什么？",
            "criteria": {"refund": "退款", "technical": "技术问题", "billing": "账单咨询"},
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "客户是否威胁要离开？",
        },
    },
)
print(result["answers"])
```

---

## Fine-Tuning · 微调

Fine-tune ruhui on your own domain data (RLCD + soft distillation + temperature calibration).
在自己的领域数据上微调（RLCD + 软蒸馏 + 温度校准）。

1. **Prepare soft-label data** · 准备软标签数据
   `datas/soft_*.jsonl`, each line: `text` + `qtype` + `soft_label.probabilities`.
   每行含 `text` + `qtype` + `soft_label.probabilities`。

2. **Build training items** · 转训练 items：
   ```bash
   python3 scripts/prepare_train_data.py \
     --model_dir <base_model_dir> \
     --soft_dir <soft_label_dir> \
     --out <train_items.pt>
   ```

3. **Train** · 训练：
   ```bash
   python3 scripts/train.py \
     --model_dir <base_model_dir> \
     --train_items <train_items.pt> \
     --output_dir <output_dir> \
     --epochs 4
   ```

4. **Inference** · 推理：
   ```bash
   python3 scripts/predict.py --model_dir <output_dir> --text "..." \
     --question-type noul --instruction "..."
   ```

---

## Repository Layout · 目录结构

```
ruhui/
  ruhui/            # package · 包（agent / router / common / presets / …）
  scripts/          # fine-tuning + inference scripts · 微调 + 推理脚本
    prepare_train_data.py
    train.py
    predict.py
  tests/
  pyproject.toml
```

---

## Model Repositories · 模型仓库

- Hugging Face: `anyforge/ruhui`
- ModelScope: `anyforge/ruhui`

> Single checkpoint: fine-tuned from `mmBERT-base`, bilingual (Chinese/English). Load it directly with `ruhui.load("anyforge/ruhui")`.
> 单一 checkpoint：基于 `mmBERT-base` 微调，中英双语。直接 `ruhui.load("anyforge/ruhui")` 加载。

## License

Apache 2.0 (inherited from Laya). Developed by AnyForge.
Apache 2.0（参照 Laya）。Developed by AnyForge。
