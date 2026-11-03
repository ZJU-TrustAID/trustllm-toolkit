"""YuFeng-XGuard's native template, labels, scores and explanations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from threading import Lock
import time

from .model_source import resolve_model_source

DEFAULT_MODEL = "Alibaba-AAIG/YuFeng-XGuard-Reason-8B"

# Display translations only; the model's own id2risk mapping is authoritative.
LABEL_NAMES_ZH = {
    "sec": "安全", "pc": "色情违禁品", "dc": "毒品犯罪", "dw": "危险武器",
    "pi": "财产侵害", "ec": "经济犯罪", "ac": "辱骂", "def": "诽谤",
    "ti": "威胁与恐吓", "cy": "网络欺凌", "ph": "身体健康", "mh": "心理健康",
    "se": "社会伦理", "sci": "科学伦理", "pp": "个人隐私", "cs": "商业秘密",
    "acc": "访问控制", "mc": "恶意代码", "ha": "黑客攻击", "ps": "物理安全",
    "ter": "暴力恐怖活动", "sd": "社会秩序破坏", "ext": "极端主义思想",
    "fin": "不当金融建议", "med": "不当医疗建议", "law": "不当法律建议",
    "cm": "对未成年人的不良诱导", "ma": "未成年人虐待与剥削", "md": "未成年人违法行为",
}


@dataclass
class GuardrailResult:
    stage: str
    verdict: str
    label: str | None
    risk_name: str | None
    risk_name_zh: str | None
    explanation: str | None
    explanation_complete: bool
    raw_output: str
    scores: list[dict]
    score_type: str
    label_probability_mass: float
    decision_basis: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    truncated: bool

    def to_dict(self) -> dict:
        return asdict(self)


def parse_output(raw: str, labels: dict[str, str]) -> dict:
    """Parse a native label first; never reinterpret an unknown label as safe."""
    match = re.match(r"\s*([A-Za-z]+)(?=\s|<|$)", raw)
    label = match.group(1) if match else None
    known = label in labels
    verdict = "safe" if known and label == "sec" else "unsafe" if known else "unknown"
    explanation = re.search(r"<explanation>(.*?)(?:</explanation>|$)", raw, re.S)
    return {
        "label": label,
        "verdict": verdict,
        "risk_name": labels.get(label),
        "risk_name_zh": LABEL_NAMES_ZH.get(label),
        "explanation": explanation.group(1).strip() or None if explanation else None,
        "explanation_complete": bool(explanation and "</explanation>" in raw),
    }


class XGuard:
    ALLOWED_DTYPES = {"auto", "float16", "bfloat16", "float32"}

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "auto", max_input_tokens: int = 8192,
                 dtype: str = "auto"):
        if dtype not in self.ALLOWED_DTYPES:
            raise ValueError(f"dtype 必须是 {sorted(self.ALLOWED_DTYPES)} 之一")
        self.model_path = resolve_model_source(model_path)
        self.device = device
        self.dtype = dtype
        self.max_input_tokens = max_input_tokens
        if Path(self.model_path).is_absolute():
            config_path = Path(self.model_path) / "tokenizer_config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(self.model_path, "tokenizer_config.json"))
        with config_path.open(encoding="utf-8") as handle:
            self.labels = json.load(handle)["id2risk"]
        if "sec" not in self.labels:
            raise ValueError("该模型缺少 XGuard 原生 sec 标签映射")
        self._lock = Lock()
        self._model = None
        self._tokenizer = None
        self._label_ids = None
        self.state = "unloaded"

    def metadata(self) -> dict:
        return {
            "model": Path(self.model_path).name,
            "model_path": self.model_path,
            "model_name": "YuFeng-XGuard-Reason-8B",
            "state": self.state,
            "device": self.device,
            "dtype": self.dtype,
            "max_input_tokens": self.max_input_tokens,
            "capabilities": {"input": True, "output": True, "explanation": True, "multi_turn": False},
            "labels": [{"label": key, "name": value, "name_zh": LABEL_NAMES_ZH.get(key, value)}
                       for key, value in self.labels.items()],
        }

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.state = "loading"
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            label_ids = []
            for label in self.labels:
                ids = tokenizer.encode(label, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"XGuard 标签 {label!r} 不是单个 token")
                label_ids.append(ids[0])
            # Explicit dtype wins; "auto" keeps the per-device default
            # (float32 on CPU for numerical safety, checkpoint dtype on GPU).
            if self.dtype != "auto" and self.device != "cpu":
                dtype = getattr(torch, self.dtype)
            else:
                dtype = torch.float32 if self.device == "cpu" else "auto"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=dtype,
                device_map=self.device, attn_implementation="sdpa",
            ).eval()
            self._tokenizer, self._model, self._label_ids = tokenizer, model, label_ids
            self.state = "ready"
        except Exception:
            self.state = "error"
            raise

    def check_input(self, text: str, *, explain: bool = True, max_new_tokens: int = 384, policy: str | None = None):
        return self._check("input", text, None, explain, max_new_tokens, policy)

    def check_output(self, text: str, response: str, *, explain: bool = True,
                     max_new_tokens: int = 384, policy: str | None = None):
        return self._check("output", text, response, explain, max_new_tokens, policy)

    def _check(self, stage, text, response, explain, max_new_tokens, policy):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入用户文本")
        if stage == "output" and (not isinstance(response, str) or not response.strip()):
            raise ValueError("输出检查需要待检测的助手回答")
        if not 32 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens 必须在 32–1024 之间")
        if len(text) + len(response or "") + len(policy or "") > 24000:
            raise ValueError("输入总长度不能超过 24000 个字符")
        if not self._lock.acquire(blocking=False):
            raise BlockingIOError("模型正在处理另一条请求，请稍后重试")
        started = time.perf_counter()
        try:
            self._load()
            import torch
            from transformers import LogitsProcessor, LogitsProcessorList

            messages = [{"role": "user", "content": text}]
            if stage == "output":
                messages.append({"role": "assistant", "content": response})
            rendered = self._tokenizer.apply_chat_template(
                messages, policy=policy, reason_first=False, tokenize=False,
            )
            inputs = self._tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_length = inputs["input_ids"].shape[1]
            if input_length > self.max_input_tokens:
                raise ValueError(f"输入含模板共 {input_length} tokens，超过 {self.max_input_tokens}；请缩短文本")
            inputs = inputs.to(self._model.device)
            label_ids = self._label_ids

            class CaptureFirstScores(LogitsProcessor):
                # Retain only the first step's label scores, not the entire
                # vocabulary tensor for every explanation token.
                values = None

                def __call__(self, input_ids, scores):
                    if self.values is None:
                        logits = scores[0].float()
                        selected = logits[label_ids]
                        self.values = (
                            torch.exp(selected - torch.logsumexp(logits, dim=0)).tolist(),
                            torch.softmax(selected, dim=0).tolist(),
                        )
                    return scores

            capture = CaptureFirstScores()
            budget = max_new_tokens if explain else 1
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs, do_sample=False, max_new_tokens=budget, use_cache=True,
                    logits_processor=LogitsProcessorList([capture]),
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            output_ids = generated[0, input_length:]
            raw = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            parsed = parse_output(raw, self.labels)
            global_probs, label_probs = capture.values
            scores = [
                {"label": label, "name": name, "name_zh": LABEL_NAMES_ZH.get(label, name),
                 "token_probability": probability, "label_probability": relative}
                for (label, name), probability, relative in zip(self.labels.items(), global_probs, label_probs)
            ]
            scores.sort(key=lambda row: row["label_probability"], reverse=True)
            eos = self._model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            truncated = bool(explain and len(output_ids) >= budget and int(output_ids[-1]) not in eos_ids)
            basis = {
                "safe": "原始分类标签为 sec，按内置标签规则判为安全。",
                "unsafe": f"原始分类标签 {parsed['label']} 属于内置风险类别，按内置标签规则判为不安全。",
                "unknown": "未识别到有效的内置分类标签，无法据此判断安全性。",
            }[parsed["verdict"]]
            return GuardrailResult(
                stage=stage, **parsed, raw_output=raw, scores=scores,
                score_type="first_token_full_vocab_and_builtin_label_softmax",
                label_probability_mass=sum(global_probs), decision_basis=basis,
                model=self.model_path, latency_ms=round((time.perf_counter() - started) * 1000, 1),
                input_tokens=input_length, output_tokens=len(output_ids), truncated=truncated,
            )
        finally:
            self._lock.release()
