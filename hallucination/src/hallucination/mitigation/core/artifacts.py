from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hallucination.mitigation.core.errors import (
    ArtifactMismatchError,
    ArtifactNotFoundError,
    ConfigError,
)


def _checksum(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float32).tobytes()
    ).hexdigest()


@dataclass
class SteeringArtifact:
    method: str
    layer: int
    vector: np.ndarray
    model: dict[str, Any] = field(default_factory=dict)
    multiplier: float = 1.0
    behavior: str = "hallucination"
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str = "hm/steering/v1"

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        if self.vector.size == 0:
            raise ConfigError("steering vector must be non-empty")
        if not np.all(np.isfinite(self.vector)):
            raise ConfigError("steering vector contains NaN/Inf")
        if self.layer < 0:
            raise ConfigError("layer must be >= 0")

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    @property
    def checksum(self) -> str:
        return _checksum(self.vector)

    def validate_for(self, model: dict[str, Any]) -> None:
        expected = self.model.get("hidden_size")
        actual = model.get("hidden_size")
        if expected is not None and actual is not None and int(expected) != int(actual):
            raise ArtifactMismatchError(
                f"steering hidden_size mismatch: artifact={expected}, model={actual}"
            )
        for key in ("model", "revision"):
            if self.model.get(key) and model.get(key) and self.model[key] != model[key]:
                raise ArtifactMismatchError(
                    f"steering {key} mismatch: artifact={self.model[key]!r}, model={model[key]!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "method": self.method,
            "layer": self.layer,
            "vector": self.vector.tolist(),
            "model": self.model,
            "multiplier": self.multiplier,
            "behavior": self.behavior,
            "metadata": self.metadata,
            "norm": self.norm,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SteeringArtifact:
        try:
            vector = payload["vector"]
            return cls(
                method=str(payload.get("method", "caa")),
                layer=int(payload["layer"]),
                vector=np.asarray(vector, dtype=np.float32),
                model=dict(payload.get("model", {})),
                multiplier=float(payload.get("multiplier", 1.0)),
                behavior=str(payload.get("behavior", "hallucination")),
                metadata=dict(payload.get("metadata", {})),
                format=str(payload.get("format", "hm/steering/v1")),
            )
        except KeyError as exc:
            raise ConfigError(f"steering artifact missing field: {exc}") from exc

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".json":
            target.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        np.savez(target, vector=self.vector)
        meta_path = target.with_suffix(".json")
        meta = self.to_dict()
        meta.pop("vector", None)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> SteeringArtifact:
        target = Path(path)
        if not target.exists():
            raise ArtifactNotFoundError(f"steering artifact not found: {target}")
        if target.suffix.lower() == ".json":
            payload = json.loads(target.read_text(encoding="utf-8"))
            return cls.from_dict(payload)
        meta_path = target.with_suffix(".json")
        if not meta_path.exists():
            raise ConfigError(f"missing steering metadata sidecar: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(target, allow_pickle=False) as data:
            meta["vector"] = np.asarray(data["vector"], dtype=np.float32)
        return cls.from_dict(meta)


@dataclass
class ITIHead:
    layer: int
    head: int
    direction: np.ndarray
    scale: float
    accuracy: float = 0.0

    def __post_init__(self) -> None:
        self.direction = np.asarray(self.direction, dtype=np.float32).reshape(-1)
        if self.direction.size == 0 or not np.all(np.isfinite(self.direction)):
            raise ConfigError("ITI direction must be finite and non-empty")
        if self.scale < 0:
            raise ConfigError("ITI scale must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "head": self.head,
            "direction": self.direction.tolist(),
            "scale": self.scale,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ITIHead:
        return cls(
            layer=int(payload["layer"]),
            head=int(payload["head"]),
            direction=np.asarray(payload["direction"], dtype=np.float32),
            scale=float(payload["scale"]),
            accuracy=float(payload.get("accuracy", 0.0)),
        )


@dataclass
class ITIArtifact:
    heads: list[ITIHead]
    model: dict[str, Any] = field(default_factory=dict)
    alpha: float = 15.0
    top_k: int = 48
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str = "hm/iti/v1"

    def __post_init__(self) -> None:
        if not self.heads:
            raise ConfigError("ITI artifact must contain at least one head")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "model": self.model,
            "alpha": self.alpha,
            "top_k": self.top_k,
            "heads": [head.to_dict() for head in self.heads],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ITIArtifact:
        return cls(
            heads=[ITIHead.from_dict(item) for item in payload["heads"]],
            model=dict(payload.get("model", {})),
            alpha=float(payload.get("alpha", 15.0)),
            top_k=int(payload.get("top_k", len(payload["heads"]))),
            metadata=dict(payload.get("metadata", {})),
            format=str(payload.get("format", "hm/iti/v1")),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ITIArtifact:
        target = Path(path)
        if not target.exists():
            raise ArtifactNotFoundError(f"ITI artifact not found: {target}")
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
