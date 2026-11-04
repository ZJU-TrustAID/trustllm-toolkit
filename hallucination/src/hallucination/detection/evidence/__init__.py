"""Claim-level external-evidence verification components."""

from hallucination.detection.evidence.claims import ClaimDecomposer
from hallucination.detection.evidence.corpus import load_textbook_evidence
from hallucination.detection.evidence.factscore import FActScoreScorer
from hallucination.detection.evidence.retrieval import LocalCorpusRetriever
from hallucination.detection.evidence.verification import (
    ClaimVerdict,
    NLIEvidenceVerifier,
)

__all__ = [
    "ClaimDecomposer",
    "ClaimVerdict",
    "FActScoreScorer",
    "LocalCorpusRetriever",
    "NLIEvidenceVerifier",
    "load_textbook_evidence",
]
