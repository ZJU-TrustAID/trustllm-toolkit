from __future__ import annotations

import numpy as np

from hallucination.mitigation.core.sampling import log_softmax, softmax
from hallucination.mitigation.methods.cad import combine_context_logits, distribution_kl


def test_alpha_zero_equals_context_log_softmax() -> None:
    context = np.array([0.2, 1.0, -0.5])
    no_context = np.array([1.0, -1.0, 0.3])
    combined = combine_context_logits(context, no_context, 0.0)
    assert np.allclose(combined, log_softmax(context))


def test_identical_views_return_same_distribution() -> None:
    logits = np.array([0.2, 1.0, -0.5])
    combined = combine_context_logits(logits, logits, 0.5)
    assert np.allclose(softmax(combined), softmax(logits))


def test_distribution_kl_zero_for_identical() -> None:
    logits = np.array([0.2, 1.0, -0.5])
    assert abs(distribution_kl(logits, logits)) < 1e-9
