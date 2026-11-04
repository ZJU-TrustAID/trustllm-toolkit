from __future__ import annotations

from types import TracebackType
from typing import Any

import numpy as np
from typing_extensions import Self

from hallucination.mitigation.core.errors import UnsupportedModelError


def find_decoder_layers(model: Any) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise UnsupportedModelError(
        "cannot locate decoder layers; supported families are Llama/Qwen/Mistral and GPT-2-like models"
    )


def find_attention_output_projection(layer: Any) -> Any:
    for attr in ("self_attn", "attn", "attention"):
        module = getattr(layer, attr, None)
        if module is None:
            continue
        for name in ("o_proj", "out_proj", "c_proj"):
            projection = getattr(module, name, None)
            if projection is not None:
                return projection
    raise UnsupportedModelError("cannot locate attention output projection for ITI")


class LayerOutputHook:
    """Add a vector to a decoder layer's residual-stream output."""

    def __init__(
        self, layer: int, vector: np.ndarray, *, position: str = "generated"
    ) -> None:
        if position not in {"generated", "all"}:
            raise ValueError("position must be 'generated' or 'all'")
        self.layer = layer
        self.vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self.position = position
        self.active = False
        self.calls = 0

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        if not self.active:
            return output
        import torch

        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(tensor):
            return output
        vector = torch.as_tensor(self.vector, dtype=tensor.dtype, device=tensor.device)
        if tensor.dim() == 3:
            if self.position == "all":
                tensor = tensor + vector
            else:
                tensor = tensor.clone()
                tensor[:, -1, :] = tensor[:, -1, :] + vector
        elif tensor.dim() == 2:
            tensor = tensor.clone()
            tensor[-1, :] = tensor[-1, :] + vector
        else:
            return output
        self.calls += 1
        if isinstance(output, (tuple, list)):
            return (tensor, *output[1:])
        return tensor


class ITIHeadHook:
    """Add per-head biases to the input of an attention output projection."""

    def __init__(
        self, heads: list[Any], *, alpha: float, position: str = "generated"
    ) -> None:
        if position not in {"generated", "all"}:
            raise ValueError("position must be 'generated' or 'all'")
        self.heads = heads
        self.alpha = alpha
        self.position = position
        self.active = False
        self.calls = 0

    def __call__(self, module: Any, args: Any) -> Any:
        if not self.active or not args:
            return None
        import torch

        tensor = args[0]
        if not torch.is_tensor(tensor) or not self.heads:
            return None
        head_dim = self.heads[0].direction.size
        width = tensor.shape[-1]
        if width % head_dim != 0:
            raise UnsupportedModelError(
                f"attention width {width} is not divisible by head_dim {head_dim}"
            )
        bias = torch.zeros(width, dtype=tensor.dtype, device=tensor.device)
        for head in self.heads:
            start = head.head * head_dim
            end = start + head_dim
            if end > width:
                raise UnsupportedModelError(
                    f"ITI head {head.head} exceeds attention width {width}; check model fingerprint"
                )
            direction = torch.as_tensor(
                head.direction, dtype=tensor.dtype, device=tensor.device
            )
            bias[start:end] += direction * (self.alpha * head.scale)
        tensor = tensor.clone()
        if tensor.dim() == 3:
            if self.position == "all":
                tensor = tensor + bias
            else:
                tensor[:, -1, :] = tensor[:, -1, :] + bias
        elif tensor.dim() == 2:
            tensor[-1, :] = tensor[-1, :] + bias
        else:
            return None
        self.calls += 1
        return (tensor, *args[1:])


class HookManager:
    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._hooks: list[Any] = []
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = bool(value)
        for hook in self._hooks:
            if hasattr(hook, "active"):
                hook.active = self._active

    def add(self, module: Any, hook: Any, *, pre_hook: bool = False) -> None:
        hook.active = self._active
        self._hooks.append(hook)
        if pre_hook:
            self._handles.append(module.register_forward_pre_hook(hook))
        else:
            self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hooks.clear()
        self.active = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.remove()
