"""Hallucination mitigation toolkit."""

from __future__ import annotations

from hallucination._version import __version__
from hallucination.mitigation.api.mitigate import Mitigator
from hallucination.mitigation.api.report import ComparisonReport, MitigationReport

__all__ = ["ComparisonReport", "MitigationReport", "Mitigator", "__version__"]
