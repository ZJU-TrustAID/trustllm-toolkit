import math

from hallucination.detection.core.results import DetectionExample
from hallucination.detection.whitebox.token_probability import LNTPScorer, MTPScorer


def test_mtp_uses_minimum_probability():
    result = MTPScorer().score(DetectionExample(response="x", token_logprobs=[math.log(0.8), math.log(0.2)]))
    assert result.score == 0.2
    assert result.risk == 0.8


def test_lntp_uses_geometric_mean():
    result = LNTPScorer().score(DetectionExample(response="x", token_logprobs=[math.log(0.25), math.log(1.0)]))
    assert round(result.score, 6) == 0.5

