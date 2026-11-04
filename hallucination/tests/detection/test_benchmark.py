from hallucination.detection.backends.api import CallableLLMBackend
from hallucination.detection.backends.nli import NLIProbabilities
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.datasets import load_dataset
from hallucination.detection.evaluation import BenchmarkRunner
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline


def _nli(premise: str, hypothesis: str) -> NLIProbabilities:
    if "海藻" in hypothesis:
        return NLIProbabilities(entailment=0.0, contradiction=1.0)
    if "海葵" in hypothesis and ("动物" in premise or "刺胞" in premise):
        return NLIProbabilities(entailment=1.0, contradiction=0.0)
    return NLIProbabilities(entailment=0.0, contradiction=0.0, neutral=1.0)


def test_benchmark_runner_evaluates_sentence_labels_and_exports_reports(tmp_path):
    pipeline = BlackBoxSamplingPipeline(
        CallableLLMBackend(lambda prompt, **kwargs: "unused"),
        sentence_scorers=[SelfCheckNLIScorer(_nli)],
    )
    report = BenchmarkRunner(pipeline).evaluate(load_dataset("textbook_toy"))
    assert report.metrics["selfcheck_nli"]["count"] == 2.0
    assert report.metrics["selfcheck_nli"]["coverage"] == 1.0
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    report.write_json(json_path)
    report.write_csv(csv_path)
    assert json_path.is_file()
    assert csv_path.is_file()
