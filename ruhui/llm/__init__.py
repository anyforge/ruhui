"""Ruhui LLM 决策后端（KEV 式架构：Causal LM + LoRA + PointerHead）。

提供两类接口：
  1. 底层：DecisionModel / PointerHead / to_record / to_answers（KEV 原语）
  2. 高层：LLMAgent（与 ruhui 的 bert.Agent 同款 predict(state, questions) 用法）
"""
from .model import DecisionModel, PointerHead, load_tokenizer
from .api import SystemOneRequest, to_record, to_answers, render, question_keys

__all__ = [
    "DecisionModel",
    "PointerHead",
    "SystemOneRequest",
    "to_record",
    "to_answers",
    "render",
    "question_keys",
    "load_tokenizer",
    "LLMAgent",
]


def LLMAgent(*args, **kwargs):
    """高层封装：与 bert.Agent 同款 predict(state, questions) 用法。

    延迟导入，避免 torch 在纯 bert 使用时被强制加载。
    """
    from .agent import LLMAgent as _Agent

    return _Agent(*args, **kwargs)
