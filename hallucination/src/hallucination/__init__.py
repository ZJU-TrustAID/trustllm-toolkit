"""Book-companion toolkit for LLM hallucination detection and inference-time mitigation.

Subpackages
-----------
- ``hallucination.detection``: score and report hallucination risk without
  changing the model output (``Detector`` facade and friends).
- ``hallucination.mitigation``: inference-time methods that change the model
  output, such as DoLa, CAD, CAA, ITI and CoVe (``Mitigator`` facade).
"""

from __future__ import annotations

from hallucination._version import __version__
from hallucination.detection import (
    DetectionExample,
    DetectionReport,
    DetectionResult,
    Detector,
    ScoreResult,
)
from hallucination.mitigation import (
    ComparisonReport,
    MitigationReport,
    Mitigator,
)

__all__ = [
    "ComparisonReport",
    "DetectionExample",
    "DetectionReport",
    "DetectionResult",
    "Detector",
    "MitigationReport",
    "Mitigator",
    "ScoreResult",
    "__version__",
]
