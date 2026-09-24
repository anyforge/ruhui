"""LLM 后端高层封装：与 bert.Agent 同款的 predict(state, questions) 接口。

加载一个 KEV 式 checkpoint（底座 + LoRA adapter + PointerHead），对结构化问题
做非自回归决策。用法对齐 ruhui 的 bert.Agent，切换后端不改调用方代码。

示例:
    from ruhui.llm import LLMAgent
    agent = LLMAgent(
        base_dir="models/Qwen3.5-0.8B-Base",   # 本地底座
        checkpoint_dir="models/kev-0.8b",       # adapter + head.pt
        device="mps",
    )
    result = agent.predict(
        {"message": "我被重复扣款了"},
        {"intent": {"type": "choice", "instructions": "客户想做什么？",
                    "criteria": {"refund": "退款", "billing": "账单"}}},
    )
"""
import json
import os
from typing import Any, Dict, Optional, Union


class LLMAgent:
    def __init__(
        self,
        checkpoint_dir: str,
        base_dir: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,          # "fp32" | "bf16" | None
        merge: bool = True,                    # LoRA 是否合并进底座权重
        temperature: Optional[float] = None,   # None = 用 checkpoint 自带温度
    ):
        """
        checkpoint_dir: adapter 目录（含 adapter_config.json / adapter_model.safetensors / head.pt）
        base_dir:       底座目录（含 model.safetensors / tokenizer）。None 时用 head.pt 里的 meta.base
                        （会去 HF 下载），本地微调/推理请显式传本地路径。
        """
        import torch

        from .checkpoint import Checkpoint, LoadOptions

        self.checkpoint_dir = checkpoint_dir
        self.base_dir = base_dir

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # 判断是「adapter 分离」还是「已合并」
        has_adapter = os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors"))
        if has_adapter:
            # 标准路径：底座 + LoRA adapter + head
            dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype)
            opts = LoadOptions(dtype=dt, merge=merge, temperature=temperature)
            ck = Checkpoint(checkpoint_dir)
            if base_dir:
                if not os.path.isdir(base_dir):
                    raise FileNotFoundError(f"base_dir 目录不存在: {base_dir!r}")
                ck.meta.base = os.path.abspath(base_dir)
                ck.meta.base_revision = None
            self._tok, self._model = ck.load(device, opts)
            self._meta = ck.meta
        else:
            # 合并路径：完整模型（无 adapter），直接加载 + head.pt
            self._load_merged(checkpoint_dir, dtype, temperature)

    def _load_merged(self, model_dir, dtype=None, temperature=None):
        """加载合并后的完整模型（无 adapter）：DecisionModel(lora=None) + head.pt。"""
        import torch

        from .model import DecisionModel, load_tokenizer
        from .checkpoint import read_meta

        meta = read_meta(model_dir)
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype)
        if dt is None:
            dt = torch.float32

        tok = load_tokenizer(model_dir)
        model = DecisionModel(
            model_dir, tok, self.device, lora=None,
            head_dim=meta.head_dim,
            option_isolation=meta.option_isolation,
            dtype=dt,
        )
        model.head.load_state_dict(meta.head)
        model.head.temperature = meta.temperature if temperature is None else temperature
        model.eval()

        self._tok = tok
        self._model = model
        self._meta = meta

    def _questions_to_records(self, state, questions):
        """把 ruhui 风格的 questions 转成 SystemOneRequest，走 to_record。"""
        from .api import SystemOneRequest, to_record

        # questions 已是 {qid: {type, instructions, criteria}} 形状，直接构造
        req = SystemOneRequest(state=state, questions=questions)
        rec, meta = to_record(req)
        return req, rec, meta

    @staticmethod
    def _enc_of(model, tok, rec):
        return model.encode(tok, rec)

    def predict(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """对齐 bert.Agent.predict：返回 {"answers": {...}, ...}。"""
        import torch

        from .api import to_answers

        req, rec, meta = self._questions_to_records(state, questions)
        enc = self._model.encode(self._tok, rec)
        with torch.no_grad():
            probs = self._model.probs(enc)   # list of tensors, 每问一个
        probs_list = [p.tolist() for p in probs]
        answers = to_answers(probs_list, meta)
        return {"model": "ruhui-llm", "answers": answers, "usage": {"input_tokens": len(enc["ids"]), "output_tokens": 0}}


def load(checkpoint_dir, base_dir=None, device=None, **kwargs):
    """便捷加载：load("models/kev-0.8b", base_dir="models/Qwen3.5-0.8B-Base")。"""
    return LLMAgent(checkpoint_dir, base_dir=base_dir, device=device, **kwargs)
