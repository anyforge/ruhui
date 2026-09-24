"""Ruhui（如晦）：非自回归 System 1 决策引擎（中文/多语言），带校准概率。

两个后端：
  - bert 后端（原 ruhui）：encoder + 决策头，33ms 级，`ruhui.load(...)` / `ruhui.Agent(...)`
  - llm 后端（KEV 式）：Causal LM + LoRA + PointerHead，`ruhui.LLMAgent(...)`

原用法完全不变：`ruhui.load("anyforge/ruhui")` 仍是 bert 后端。
"""
from .bert import *  # noqa: F401,F403  —— 原 ruhui 全部接口
from .bert import __all__ as _BERT_ALL

__version__ = "0.2.0"

# LLM 后端（延迟导入，避免 torch 未装时报错）
def _llm_all():
    return ["LLMAgent"]


def LLMAgent(*args, **kwargs):
    """LLM 后端入口：与 Agent 同款 predict 用法。

    agent = ruhui.LLMAgent(checkpoint_dir="models/kev-0.8b", base_dir="models/Qwen3.5-0.8B-Base")
    agent.predict(state, questions)
    """
    from .llm.agent import LLMAgent as _A

    return _A(*args, **kwargs)


__all__ = list(_BERT_ALL) + ["LLMAgent", "__version__"]
