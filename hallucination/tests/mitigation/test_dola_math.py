from __future__ import annotations

import numpy as np

from hallucination.mitigation.methods.dola import (
    apc_mask,
    contrast_logits,
    jsd,
    select_premature_layer,
)


def test_jsd_zero_for_identical_distributions() -> None:
    logits = np.array([0.0, 1.0, 2.0])
    assert jsd(logits, logits) == 0.0


def test_jsd_positive_for_different_distributions() -> None:
    assert jsd(np.array([10.0, 0.0]), np.array([0.0, 10.0])) > 0.0


def test_select_premature_layer_prefers_largest_divergence() -> None:
    mature = np.array([0.0, 10.0, -5.0])
    candidates = {
        2: np.array([0.0, 10.0, -5.0]),
        4: np.array([10.0, 0.0, -5.0]),
    }
    layer, scores = select_premature_layer(mature, candidates)
    assert layer == 4
    assert scores[4] > scores[2]


def test_apc_mask_keeps_only_relative_top() -> None:
    masked = apc_mask(np.array([0.0, -10.0, 5.0]), 0.1)
    assert np.isfinite(masked[2])
    assert not np.isfinite(masked[1])
    assert not np.isfinite(masked[0])


def test_contrast_logits_is_stable_and_finite() -> None:
    mature = np.array([1.0, 2.0, 3.0])
    premature = np.array([3.0, 2.0, 1.0])
    out = contrast_logits(mature, premature, relative_top=0.1, post_softmax=True)
    assert out.shape == mature.shape
    assert np.all(np.isfinite(out))
