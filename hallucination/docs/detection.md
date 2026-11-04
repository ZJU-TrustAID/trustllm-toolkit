# Hallucination Detection Toolkit

`hallucination.detection` is the executable companion to the hallucination-detection chapter in *Trustworthy Large Language Models*. It provides small, inspectable implementations of white-box confidence signals, black-box response/sentence consistency signals, and external-evidence verification.

The package detects and reports risk. It does not rewrite, regenerate, refuse, or otherwise mitigate answers.

## Status (v0.3)

This release provides:

- response-level exact-match, cosine consistency, and semantic entropy;
- sentence-level SelfCheck n-gram, BERTScore, and NLI;
- deterministic white-box MTP, LNTP, token entropy, and experimental hidden-state methods;
- FActScore-style local evidence verification;
- P(True), CoVe detector, LLM-as-a-Judge, and LLM judge panels;
- per-method calibration bundles (threshold, Platt/isotonic/temperature, ECE/Brier/NLL) that `Detector` and benchmarks apply only to matching scores;
- a one-click `Detector` facade and a `hallucination-detection` CLI;
- dataset adapters, an offline benchmark runner, 13 real-backend tutorials, and a local web demo.

Not included: MQAG, LLM-prompt SelfCheck, and trained text-classifier baselines.

Reference commits and intentional deviations from the official implementations are documented in [detection-fidelity.md](detection-fidelity.md).

## Install

From the repository root (the toolkit is one distribution; detection is the
`hallucination.detection` subpackage):

```bash
python -m pip install -e ".[dev]"
```

Optional backends:

```bash
python -m pip install -e ".[whitebox]"          # local Hugging Face causal LM
python -m pip install -e ".[nli]"               # NLI / semantic entropy / SelfCheck-NLI
python -m pip install -e ".[embeddings]"        # sentence-transformer cosine
python -m pip install -e ".[selfcheck-bertscore]"
python -m pip install -e ".[datasets]"          # remote dataset adapters
python -m pip install -e ".[openai]"            # OpenAI Responses API
python -m pip install -e ".[dataframe]"         # DetectionReport.to_dataframe()
```

## One-Click Detection

```python
from hallucination.detection import Detector

detector = Detector(preset="offline_toy")
report = detector.detect("请介绍蓝鲸的食性。")
print(report.to_markdown())
report.to_json("report.json")
```

`offline_toy` is deterministic and never calls a network or downloads a model. Other presets:

```python
detector = Detector(preset="openai_blackbox", model="gpt-4o-mini", num_samples=5)
detector = Detector(preset="hf_whitebox", model="/path/to/causal-lm")
```

CLI:

```bash
python -m hallucination.detection detect --preset offline_toy --prompt "请介绍蓝鲸的食性。"
python -m hallucination.detection detect --response "蓝鲸主要以磷虾为食。" --samples '["蓝鲸以磷虾为食。", "蓝鲸吃磷虾。"]'
python -m hallucination.detection detect --prompt "请介绍蓝鲸的食性。" --calibrator cal.json
python -m hallucination.detection benchmark --dataset textbook_toy
python -m hallucination.detection calibrate --scores scores.jsonl --method isotonic --score-name response_semantic_entropy --granularity response --out cal.json
python -m hallucination.detection list-methods
```

Pass `--prompt` to generate the response and its samples, or `--response` together with `--samples` to score answers you already have. Response-only runs need samples because every packaged scorer is a self-consistency method; without them the CLI exits with an error instead of inventing a comparison. `--samples` also accepts `@samples.json` when shell quoting makes inline JSON awkward, for example on Windows PowerShell.

## Result Semantics

All scorers return `score` and `risk` in `[0, 1]`:

- `score`: higher is more favorable for the scorer's own signal.
- `risk`: higher is less favorable for the scorer's own signal.
- `signal_type`: records what the score measures.

| Signal type | Meaning | Current examples |
|---|---|---|
| `generation_confidence` | Local uncertainty during generation | MTP, LNTP, token entropy, EigenScore, EffectiveRank |
| `self_consistency` | Agreement among independent samples from the same model | exact-match consistency, response cosine, response semantic entropy, SelfCheck NLI/BERTScore/n-gram |
| `evidence_support` | Support from a reference answer or external evidence | reference match, FActScore-style verification |
| `self_reported_confidence` | The model's reported confidence | P(True), CoVe detector |
| `judge_assessment` | An independent judge model's assessment | LLM judge, LLM panel |

Self-consistency is not factual verification. A model can produce the same incorrect statement repeatedly. `DetectionPipeline` rejects aggregation across different signal types unless callers explicitly opt in after validating such a combination.

`ScoreResult` includes assumptions, warnings, provenance, and optional calibrated values. `DetectionResult.signal_scores` groups scorer outputs by signal type. Sentence-level methods use `UnitResult` for traceable diagnostics.

## Methods and Granularity

| Method | Signal type | Granularity | State |
|---|---|---|---|
| MTP | generation confidence | token/response | available |
| LNTP | generation confidence | response | available |
| Token entropy | generation confidence | token/response | available |
| EigenScore | generation confidence | response | experimental |
| Effective rank | generation confidence | response | experimental |
| Reference match | evidence support | response | available |
| Exact-match consistency | self consistency | response | available |
| Response cosine consistency | self consistency | response | available with embedding backend |
| Response semantic entropy | self consistency | response | available with NLI backend |
| SelfCheck n-gram | self consistency | sentence | available |
| SelfCheck BERTScore | self consistency | sentence | available with BERTScore backend |
| SelfCheck NLI | self consistency | sentence | available with NLI backend |
| P(True) | self-reported confidence | response | available (logprob or verbalized) |
| CoVe detector | self-reported confidence | response | available with a generation backend |
| LLM judge | judge assessment | response | available (binary/continuous/Likert) |
| LLM panel | judge assessment | response | available |
| FActScore-style verification | evidence support | claim | available with retriever and verifier |

Response-level cosine and semantic entropy follow the textbook formulas and UQLM: the whole answer is compared with whole sampled answers. Sentence-level consistency is only provided by the SelfCheck family. Sentence positions are never assumed to align.

### White-box token signals over an API

`OpenAIChatBackend` (Chat Completions) exposes per-token logprobs, so
`WhiteBoxGenerationPipeline` and `Detector(preset="openai_whitebox")` run MTP,
LNTP, and token entropy against API or OpenAI-compatible servers. Top-k is
capped at 20 by the API, so entropy stays a top-k approximation, and
hidden-state methods (EigenScore, Effective rank) still require a local model.
`PTrueScorer` reads A/B logprobs from the same window and falls back to the
verbalized estimator when the tokens are absent.

### White-box token signals

```python
import math

from hallucination.detection import DetectionExample
from hallucination.detection.pipelines.whitebox_generation import build_whitebox_generation_pipeline

example = DetectionExample(
    prompt="蓝鲸主要吃什么？",
    response="蓝鲸主要以磷虾为食。",
    token_logprobs=[math.log(0.92), math.log(0.80), math.log(0.76)],
    tokens=["蓝鲸", "主要", "以磷虾为食"],
    token_is_special=[False, False, False],
    top_logprobs=[
        {"磷虾": math.log(0.70), "海藻": math.log(0.20), "小鱼": math.log(0.10)},
    ],
)

result = build_whitebox_generation_pipeline().detect(example)
for item in result.method_scores:
    print(item.method, item.score, item.risk, item.signal_type)
```

`MTPScorer` and `LNTPScorer` accept a `TokenFilter` to exclude special tokens, punctuation, or leading/trailing tokens. `TokenEntropyScorer` reports `distribution_coverage` and treats a one-candidate step as zero entropy with a warning. Top-k entropy is an approximation when the full vocabulary distribution is unavailable.

### Response-level exact-match consistency

```python
from hallucination.detection import DetectionExample
from hallucination.detection.blackbox.exact_match import (
    ExactMatchConsistencyScorer,
)

result = ExactMatchConsistencyScorer(normalize=True).score(
    DetectionExample(response="软骨鱼纲", samples=["软骨鱼纲", "硬骨鱼纲", "软骨鱼纲"])
)
print(result.score)  # 2 / 3 literal agreement
```

`normalize=False` reproduces strict equality; the default normalizes case, whitespace, and punctuation for short Chinese answers.

### Response-level cosine and semantic entropy

```python
from hallucination.detection.backends.embeddings import SentenceTransformerEmbeddingBackend
from hallucination.detection.backends.nli import BidirectionalNLIBackend
from hallucination.detection.blackbox.response_cosine import ResponseCosineConsistencyScorer
from hallucination.detection.blackbox.response_semantic_entropy import ResponseSemanticEntropyScorer
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline

embedding = SentenceTransformerEmbeddingBackend("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
nli = BidirectionalNLIBackend("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

pipeline = BlackBoxSamplingPipeline(
    backend=my_llm_backend,
    scorers=[
        ResponseCosineConsistencyScorer(embedding),
        ResponseSemanticEntropyScorer(nli, equivalence="official", base=2.0),
    ],
    num_samples=5,
)
result = pipeline.detect_prompt("请介绍蓝鲸的食性。")
for item in result.method_scores:
    print(item.method, item.risk, item.raw.get("semantic_entropy"))
```

`ResponseSemanticEntropyScorer` reports raw entropy, normalized entropy, cluster assignments, cluster probabilities, and the number of semantic sets. The official three-way equivalence rule is the default; `strict` and `threshold` variants are available for teaching.

### Sentence-level SelfCheck

```python
from hallucination.detection.blackbox.selfcheck_ngram import SelfCheckNgramScorer
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline

pipeline = BlackBoxSamplingPipeline(
    backend=my_llm_backend,
    sentence_scorers=[SelfCheckNgramScorer(n=1), SelfCheckNLIScorer(nli)],
    num_samples=5,
)
result = pipeline.detect_prompt("请介绍蓝鲸的食性。")
for unit in result.units:
    print(unit.text, unit.risk)
```

`SelfCheckNLIScorer` uses the official input order `(sentence, sample)` by default; pass `premise="sample"` for the earlier toolkit behaviour. `SelfCheckNgramScorer` trains on the original response and samples, reports both average and maximum negative log-probability, and exposes the document-level aggregate.

### Claim-level external evidence

```python
from hallucination.detection.backends.nli import BidirectionalNLIBackend
from hallucination.detection.evidence import FActScoreScorer, LocalCorpusRetriever, NLIEvidenceVerifier, load_textbook_evidence
from hallucination.detection.pipelines import EvidenceVerificationPipeline

pipeline = EvidenceVerificationPipeline(
    FActScoreScorer(
        NLIEvidenceVerifier(BidirectionalNLIBackend()),
        retriever=LocalCorpusRetriever(load_textbook_evidence()),
    )
)
result = pipeline.detect(response="蓝鲸主要以海藻为食。")
print(result.claims[0].label)  # contradicted
```

`unknown` means the configured retrieval/verifier could not establish support or contradiction. By default it counts as **not supported**, so `risk = 1 - supported_ratio` and an answer with no supported claims is high risk. Pass `unknown_policy="contradicted"` only when you explicitly want the older "unknown is not a contradiction" behaviour. The packaged corpus only supports the textbook examples.

### Supplementary methods: P(True), CoVe, and LLM judges

```python
from hallucination.detection import (
    CoVeScorer,
    DetectionExample,
    LLMJudgeScorer,
    LLMPanelScorer,
    PTrueScorer,
)

example = DetectionExample(
    prompt="请介绍蓝鲸的食性。",
    response="蓝鲸主要以磷虾为食。",
)

p_true = PTrueScorer(backend)                       # logprob backend, or verbalized fallback
cove = CoVeScorer(backend, nli=nli, max_questions=5)  # questions -> independent answers -> conflicts
judge = LLMJudgeScorer(judge_backend, mode="binary")  # also "continuous" and "likert"
panel = LLMPanelScorer([judge, LLMJudgeScorer(other_judge, mode="likert")])
```

- `PTrueScorer` uses the A) True / B) False prompt and reads the ` A` / ` B` logprob when the backend implements `next_token_logprobs`; the default normalizes over A/B, while `normalize_ab=False` reproduces the official "read only the ` A` logprob" estimator. The official few-shot exemplars are not included; without logprobs the scorer falls back to a temperature-zero verbalized answer. The fallback is labelled in `raw["method"]`.
- `CoVeScorer` reports the fraction of verification answers that conflict with the draft. It never rewrites the answer. An NLI verifier is recommended; the lexical fallback is explicitly labelled.
- `LLMJudgeScorer` supports binary, continuous (0–100), and Likert (1–5) judgements with JSON parsing and trace. `LLMPanelScorer` averages independent judges.
- All three are self-report/judge signals; none is a factual oracle, and all should be calibrated on validation data.

## Calibration

Calibration is per score vector, not global: a semantic-entropy calibrator must not be applied to a cosine score, and a response-level calibrator must not be applied to sentence scores. `CalibrationBundle` records that binding, and `Detector` / `BenchmarkRunner` apply only matching bundles.

```python
from hallucination.detection import Detector
from hallucination.detection.calibration import CalibrationBundle

bundle = CalibrationBundle.fit(
    validation_risks,
    validation_labels,                 # True means hallucination
    score_name="response_semantic_entropy",
    granularity="response",
    probability_method="isotonic",     # or "platt" / "temperature"
    threshold_metric="f1",             # threshold is fit on calibrated risks
)
bundle.save("cal.json")

detector = Detector(preset="offline_toy", calibration="cal.json")
report = detector.detect("请介绍蓝鲸的食性。")
```

For a benchmark over several methods, build one bundle per method and pass a `CalibrationRegistry`, or pass a registry JSON to `benchmark --calibrator registry.json`. Methods without a bundle keep their raw metrics; calibrated variants are emitted only for matched methods.

The lower-level pieces remain available:

```python
from hallucination.detection.calibration import (
    CalibrationReport,
    ProbabilityCalibrator,
    ThresholdCalibrator,
)

threshold = ThresholdCalibrator(objective="fbeta", beta=2.0).fit(
    validation_risks, validation_labels, split="validation"
)
calibrator = ProbabilityCalibrator("isotonic", target="hallucination").fit(
    validation_risks, validation_labels, split="validation"
)
calibrated = calibrator.transform(test_risks)
report = CalibrationReport(test_risks, test_labels, calibrated_scores=calibrated, method="isotonic")
print(report.to_markdown())   # ECE, MCE, Brier, NLL
```

`ThresholdCalibrator` supports `f1`, `fbeta`, `balanced_accuracy`, and `cost`. `ProbabilityCalibrator` supports `platt`, `isotonic`, and `temperature`. Both refuse to fit on `split="test"`. `CalibrationBundle.fit` also refuses test data and records `n`, `positive_rate`, and `split`. Calibration uses labels and is only meaningful for the same model, task, and language as the validation data.

## Benchmarking

```python
from hallucination.detection.datasets import load_dataset
from hallucination.detection.evaluation import BenchmarkRunner

report = BenchmarkRunner(pipeline, granularity="sentence").evaluate(
    load_dataset("textbook_toy"), granularity="sentence"
)
print(report.metrics)
report.write_json("benchmark.json")
report.write_csv("benchmark.csv")
```

The report provides count, precision, recall, F1, ROC-AUC, PR-AUC, coverage, ECE, Brier, NLL, and calibrated variants for methods with a matching `CalibrationBundle`. A legacy `ProbabilityCalibrator` is treated as a wildcard bundle for backward compatibility; prefer per-method bundles. Labels are interpreted as hallucination labels: `True`, `contradicted`, `not_supported`, or `unsupported` means positive risk. Threshold and calibration selection must use validation data, never the test set.

## Dataset Loading

```python
from hallucination.detection.datasets import load_dataset

records = load_dataset("jsonl", path="annotations.jsonl", split="validation")
records = load_dataset("textbook_toy")
records = load_dataset("selfcheck_wikibio", split="train")  # requires .[datasets]
```

Local records require `id` and `response`. Optional fields include `prompt`, `samples`, `reference_answer`, `units`, `unit_labels`, `claims`, `claim_labels`, `evidence`, `response_label`, `split`, `label_source`, and `metadata`. `units`/`unit_labels` and `claims`/`claim_labels` must have matching lengths.

## Web Demo

```bash
python web_demo/server.py
```

Open `http://127.0.0.1:8765`. The built-in model works offline. For real inference, select an OpenAI-compatible chat completions API or install `.[whitebox]` and select a local Hugging Face causal model. SelfCheck NLI and response semantic entropy use `.[nli]` with a Hugging Face NLI model for real API answers.

Without a fitted `CalibrationBundle`, the displayed risk and confidence are detector signals, not calibrated probabilities of factual correctness.

## Examples

The tutorials are real-backend notebooks in `examples/detection/` (`00`-`12`), one per usable method; none ship pre-executed outputs. See the [examples index](../examples/detection/README.md).

## Tests

```bash
python -m pytest tests
```

Default tests are deterministic and do not require a network connection or model download. Calibration tests cover Platt/isotonic, ECE/Brier/NLL, split isolation, per-method bundle routing, and save/load. API/model integration tests belong to a separate suite.

## Development Roadmap

The complete delivery plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Remaining planned work: asynchronous/batched sampling and reliability-diagram plotting. MQAG, LLM-prompt SelfCheck, and trained text classifiers are intentionally out of scope.
