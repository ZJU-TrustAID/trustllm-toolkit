from __future__ import annotations

import numpy as np
import pytest

from hallucination.mitigation.core.artifacts import (
    ITIArtifact,
    ITIHead,
    SteeringArtifact,
)
from hallucination.mitigation.core.errors import ArtifactMismatchError
from hallucination.mitigation.methods.caa import fit_caa_from_activations


def test_fit_caa_uses_mean_difference_and_normalizes() -> None:
    positive = np.array([[1.0, 0.0], [3.0, 0.0]])
    negative = np.array([[0.0, 0.0], [0.0, 0.0]])
    artifact = fit_caa_from_activations(
        positive, negative, layer=3, model={"hidden_size": 2}
    )
    assert artifact.layer == 3
    assert np.isclose(artifact.norm, 1.0)
    assert artifact.vector[0] > 0


def test_steering_artifact_json_roundtrip(tmp_path) -> None:
    artifact = SteeringArtifact(
        method="caa", layer=2, vector=np.array([0.6, 0.8], dtype=np.float32)
    )
    path = tmp_path / "caa.json"
    artifact.save(path)
    loaded = SteeringArtifact.load(path)
    assert loaded.layer == 2
    assert np.allclose(loaded.vector, artifact.vector)


def test_steering_artifact_npz_roundtrip(tmp_path) -> None:
    artifact = SteeringArtifact(
        method="caa", layer=2, vector=np.array([0.6, 0.8], dtype=np.float32)
    )
    path = tmp_path / "caa.npz"
    artifact.save(path)
    loaded = SteeringArtifact.load(path)
    assert np.allclose(loaded.vector, artifact.vector)


def test_steering_artifact_rejects_hidden_size_mismatch() -> None:
    artifact = SteeringArtifact(
        method="caa", layer=2, vector=np.array([1.0, 0.0]), model={"hidden_size": 2}
    )
    with pytest.raises(ArtifactMismatchError):
        artifact.validate_for({"hidden_size": 4})


def test_iti_artifact_roundtrip(tmp_path) -> None:
    artifact = ITIArtifact(
        heads=[
            ITIHead(
                layer=1, head=2, direction=np.array([1.0, 0.0]), scale=0.5, accuracy=0.8
            )
        ],
        model={"hidden_size": 2, "num_heads": 2, "head_dim": 1},
        alpha=15.0,
        top_k=1,
    )
    path = tmp_path / "iti.json"
    artifact.save(path)
    loaded = ITIArtifact.load(path)
    assert loaded.heads[0].head == 2
    assert loaded.alpha == 15.0
