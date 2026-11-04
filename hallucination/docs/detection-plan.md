# Hallucination Detection Toolkit: Textbook Release Plan

## 1. Goal and Scope

`hallucination.detection` is the executable companion to the hallucination-detection chapter in *Trustworthy Large Language Models*. Its first release must let a reader:

1. load an annotated dataset or a small built-in teaching dataset;
2. run the methods introduced in the chapter under the appropriate white-box or black-box assumptions;
3. inspect intermediate artifacts rather than receive only a scalar score;
4. compare methods with reproducible metrics; and
5. understand each method's failure mode.

The package detects and reports risk. It does not rewrite, regenerate, abstain, or otherwise mitigate an answer. Those actions belong in a future mitigation package.

### 1.1 Non-goals for the first textbook release

- A universal factuality oracle.
- A single score that treats self-consistency as evidence-grounded truth.
- Code-generation, multimodal, graph-UQ, or agent-specific methods not covered by the textbook chapter.
- Bundling copyrighted or externally licensed benchmark data into the wheel.

## 2. Method Taxonomy

The package has three evidence classes. Scores from different classes must retain this provenance and must not be silently averaged.

| Evidence class | What it measures | Methods |
|---|---|---|
| `generation_confidence` | The generating model's local uncertainty | MTP, LNTP, token entropy, hidden-state methods |
| `self_consistency` | Whether independent samples from the same model support the answer | exact-match consistency, response cosine, response semantic entropy, sentence-level SelfCheck variants |
| `evidence_support` | Whether retrieved external evidence supports a claim | FActScore-style claim verification |
| `self_reported_confidence` | The model's own verbal confidence | P(True), CoVe detector |
| `judge_assessment` | A separate judge model's evaluation | LLM-as-a-Judge |

Low self-consistency risk means only that the sampled responses agree. It does not prove factual correctness. The README, result objects, examples, and reports must show this warning.

## 3. Architecture

### 3.1 One black-box sampling pipeline, not a parallel SelfCheck pipeline

SelfCheckGPT is a useful reference implementation for a family of sampled-response comparison methods. It is not a single method and should not become an independent public pipeline with its own generation and sampling logic.

The public black-box pipeline generates the original response and its independent samples once. It then routes these artifacts to response-level and unit-level scorers.

```text
prompt
  -> original response + N independently sampled responses
  -> response-level scorers
     -> exact-match consistency
     -> cosine consistency
     -> semantic entropy
  -> sentence splitter
  -> sentence-level SelfCheck scorers
     -> n-gram
     -> BERTScore
     -> NLI
  -> separately optional evidence-verification path
     -> claim decomposition -> retrieval -> verification -> FActScore
  -> DetectionResult with separate signal groups
```

Exact-match consistency, cosine consistency, and semantic entropy are response-level: the whole answer is compared with whole sampled answers, matching the textbook formulas and the UQLM scorer definitions. Sentence-level consistency is provided only by the SelfCheck family. Sentence positions are never assumed to align. FActScore is separate because it uses an external corpus and produces evidence support, not model self-consistency.

### 3.2 Source layout

```text
src/hallucination.detection/
  api/
    detect.py          # Detector facade
    report.py          # DetectionReport
  core/
    base.py
    results.py
    config.py
    thresholds.py
  backends/
    llm.py
    api.py
    demo.py
    hf_whitebox.py
    embeddings.py
    nli.py
    retrieval.py
  whitebox/
    tokens.py
    token_probability.py
    entropy.py
    eigenscore.py
    effective_rank.py
  blackbox/
    exact_match.py
    response_cosine.py
    response_semantic_entropy.py
    sentence.py
    selfcheck_ngram.py
    selfcheck_bertscore.py
    selfcheck_nli.py
  calibration/
    threshold.py
    probability.py
    metrics.py
    report.py
    calibrator.py
  evidence/
    claims.py
    verification.py
    factscore.py
    retrieval.py
    corpus.py
  datasets/
    schema.py
    registry.py
    loaders.py
    adapters/
      selfcheck_wikibio.py
    cards/
  evaluation/
    benchmark.py
    metrics.py
  pipelines/
    short_answer.py
    whitebox_generation.py
    evidence_verification.py
    long_form.py
    ensemble.py
  cli.py
  __main__.py
  utils/
    text.py
    math.py
    async_utils.py
```

`blackbox/selfcheck.py` and the public `SelfCheckGPTScorer` name are removed. A deprecated import alias may remain for one minor release only, with a migration warning.

## 4. Common Data Contracts

### 4.1 Input

`DetectionExample` remains the single-example contract but adds optional unit metadata:

```python
DetectionExample(
    prompt="...",
    response="...",
    samples=["...", "..."],
    reference_answer="...",
    token_logprobs=[...],
    top_logprobs=[...],
    hidden_states=...,
    units=["..."],              # optional pre-segmented sentences or claims
    unit_type="sentence",        # "sentence" or "claim"
    retrieved_evidence=[...],
    metadata={...},
)
```

### 4.2 Output

All public methods follow one direction: `score` is higher when the signal is more favorable; `risk = 1 - score` is higher when the signal is unfavorable.

```python
ScoreResult(
    method="selfcheck_nli",
    signal_type="self_consistency",
    score=0.72,
    risk=0.28,
    level="sentence",
    raw={...},
    assumptions=[...],
    warnings=[...],
    provenance={...},
)

UnitResult(
    text="...",
    index=0,
    unit_type="sentence",
    score=0.72,
    risk=0.28,
    label="low_risk",
    method_scores=[...],
)
```

`DetectionResult` contains grouped response-level results, `units`, claims, evidence, and full run metadata. The aggregate score is only available for scorers of the same `signal_type` unless users explicitly configure a cross-signal ensemble.

### 4.3 Provenance required in every nontrivial result

- scorer version and configuration;
- model name/version for embedding, NLI, judge, and generator;
- sample count, decoding parameters, and random seed;
- dataset ID/version/split when applicable;
- external evidence source and retrieval parameters;
- latency, token usage, and recoverable failures.

## 5. Dataset and Benchmark Support

Dataset support is a first-class capability, not a final-phase feature.

### 5.1 Dataset record

```python
DatasetRecord(
    id="...",
    prompt="...",
    response="...",
    samples=[...],
    reference_answer=None,
    units=[...],
    unit_labels=[...],
    response_label=None,
    split="validation",
    label_source="human",
    claims=[...],
    claim_labels=[...],
    evidence=[...],
    metadata={...},
)
```

`unit_labels` and `claim_labels` use explicit label vocabularies defined in the dataset card. The loader validates that text and labels have matching lengths and does not silently infer missing labels.

### 5.2 Loaders

```python
from hallucination.detection.datasets import list_datasets, load_dataset

dataset = load_dataset("selfcheck_wikibio", split="test")
local = load_dataset("jsonl", path="my_annotations.jsonl")
```

Required loaders:

- local JSON, JSONL, and CSV;
- Hugging Face Hub datasets through the optional `datasets` dependency;
- a registry with lazy remote download, cache directory, offline mode, checksum, and dataset version;
- a tiny locally packaged `textbook_toy` dataset for fully offline examples and tests.

The first remote adapter is `potsawee/wiki_bio_gpt3_hallucination`. It is loaded from its original source after the reader accepts its license; it is not copied into this repository. Its adapter maps generated passages, sampled passages, sentence splits, and human sentence labels into `DatasetRecord`.

### 5.3 Dataset cards

Every registered dataset needs a Markdown card covering source URL, version, license, language, task, instance count, label meaning, field mapping, known limitations, compatible methods, and citation.

### 5.4 Benchmark runner

```python
report = BenchmarkRunner(
    pipeline=pipeline,
    granularity="sentence",
).evaluate(dataset)
```

Required report fields:

- PR-AUC, ROC-AUC, F1, precision, recall, and coverage;
- response and unit-level metrics reported separately;
- number of omitted, failed, and unparsable records;
- average latency, model calls, and token usage;
- scorer configuration, data version, and random seed;
- JSON and CSV export.

Calibration and threshold fitting must use validation data only. Test labels may never be used to select a threshold or ensemble weight.

## 6. Methods

### 6.1 White-box methods

| Method | Implementation | Teaching output |
|---|---|---|
| MTP | selected token probability minimum | token, probability, lowest-risk contributors |
| LNTP | geometric mean of token probabilities | mean logprob, normalized probability |
| Token entropy | full-distribution or marked top-k approximation | per-token entropy timeline and aggregation |
| EigenScore | experimental hidden-state spectral method | selected layer, pooling, spectrum |
| Effective Rank | experimental SVD-based diversity | singular values, effective rank |

Requirements:

- token filtering controls for special tokens, punctuation, and initial tokens;
- `mean`, `max`, and `p95` entropy aggregation;
- explicit `distribution_coverage`/truncation metadata for top-k entropy;
- HF backend support for repeated sampled generations and retained hidden-state collections;
- hidden-state methods remain `experimental` until benchmarked and calibrated per model/task.

### 6.2 Black-box sampled consistency by fixed granularity

| Method | Inputs | Status and contract |
|---|---|---|
| `ReferenceMatchScorer` | response + answer key | closed-answer correctness proxy |
| `ExactMatchConsistencyScorer` | response + samples | response-level literal sample agreement only, not reference correctness |
| `ResponseCosineConsistencyScorer` | response + samples + embedding backend | response-level mean cosine over whole answers |
| `ResponseSemanticEntropyScorer` | response + samples + bidirectional NLI backend | response-level semantic clustering and entropy |

Split the current `ExactMatchScorer`: matching a reference answer and agreement with sampled answers are different signals and must have different names.

Response cosine and semantic entropy follow the textbook formulas and UQLM: the unit of comparison is the complete answer. They do not perform sentence alignment. Sentence-level consistency belongs exclusively to the SelfCheck family in §6.3.

Response semantic entropy must report raw entropy, normalized entropy, cluster assignments, cluster probabilities, and the number of semantic sets. A sampled passage that omits a fact is not equivalent to one that directly contradicts it.

Production embedding and NLI backends are required for real use. The keyword/hashing embedding and demo NLI implementations remain only under `demo`/offline namespaces; public presets do not silently use them for real API answers.

### 6.3 SelfCheck-inspired unit-level methods

All variants reuse the response and samples produced by `BlackBoxSamplingPipeline`. They score each original sentence or claim against each sampled passage, then aggregate across passages and units.

| Scorer | Risk construction | Phase |
|---|---|---|
| `SelfCheckNgramScorer` | token n-gram negative log probability under the original response and samples | R1 baseline |
| `SelfCheckBERTScoreScorer` | `1 -` best-sentence BERTScore support in each sampled passage | R1 |
| `SelfCheckNLIScorer` | mean (or max) contradiction probability in official `(sentence, sample)` order | R1 default |

For every unit scorer, `raw` must expose per-sample values, aggregation, skipped samples, and the most contradictory/supporting sample. NLI must preserve the distinction between contradiction, neutral, and entailment.

MQAG and LLM-prompt SelfCheck are **intentionally excluded** from this toolkit: both are official SelfCheckGPT variants, but they require additional question-generation/answering or LLM-judge infrastructure and are not part of the textbook detection main line. The README documents their absence.

### 6.4 External-evidence verification: FActScore-style

This is a different path from SelfCheck:

```text
response -> atomic claims -> evidence retrieval -> claim verification
         -> supported / contradicted / unknown
```

Components:

- `ClaimDecomposer`: pre-segmented claims, deterministic sentence fallback, and optional LLM atomic-fact decomposition;
- `Retriever`: local corpus implementation for offline teaching, plus optional Wikipedia/web/vector-store adapters;
- `ClaimVerifier`: NLI and LLM variants;
- `FactScoreScorer`: `supported_ratio`, `contradicted_ratio`, `unknown_ratio`, and evidence trace for every claim.

Do not call every missing-evidence case a hallucination. `unknown` means the configured retrieval system could not establish support or contradiction.

### 6.5 Textbook supplementary methods

| Method | Status | Boundary |
|---|---|---|
| P(True) | implemented | structured self-report; show calibration and overconfidence |
| CoVe detector | implemented | generate verification questions and report conflicts; no answer rewrite |
| LLM-as-a-Judge | implemented | structured binary/continuous/Likert judgement with trace; panel supported |

A trained text-classifier baseline is intentionally excluded: it requires user-owned labels and a domain-specific checkpoint, and no universal bundled classifier is scientifically defensible.

## 7. Pipelines and Presets

### 7.1 `BlackBoxSamplingPipeline`

This is the single generation-and-sampling owner for black-box methods.

```python
pipeline = BlackBoxSamplingPipeline(
    backend=llm_backend,
    scorers=[
        ExactMatchConsistencyScorer(),
        ResponseCosineConsistencyScorer(embedding_backend),
        ResponseSemanticEntropyScorer(nli_backend),
    ],
    sentence_scorers=[
        SelfCheckNgramScorer(n=1),
        SelfCheckNLIScorer(nli_backend),
    ],
    num_samples=5,
)
```

Rules:

- it never invokes each scorer's own sampler;
- callers can provide existing `response` and `samples` to avoid regeneration;
- exact-match consistency, cosine consistency, and semantic entropy declare response-level operation only;
- SelfCheck n-gram, BERTScore, and NLI declare sentence-level operation only;
- every sampled passage is independently sentence-segmented before SelfCheck scoring;
- any response-level value derived from sentence scores is labelled as an aggregation, not as a response-level scorer;
- evidence verification remains an explicit optional stage, avoiding unintended network calls or costs.

### 7.2 `WhiteBoxGenerationPipeline`

Generates once through the HF or logprob-capable backend and runs MTP, LNTP, and entropy. It exposes token diagnostics and never presents an uncalibrated threshold as factual truth.

### 7.3 `EvidenceVerificationPipeline`

Takes a response or claims and runs the FActScore-style path. It can be composed with a black-box pipeline but returns an `evidence_support` group separately.

### 7.4 Ensemble policy

`DetectionPipeline` must replace equal-weight-only averaging. It supports:

- aggregation within one signal type: mean, min, weighted mean;
- cross-signal combination only when user supplies explicit weights and a validation-calibrated configuration;
- no default combination of self-consistency and evidence-support scores;
- saved configuration with scorer versions and calibration provenance.

## 8. Dependencies

Base installation remains small: `numpy`, `pyyaml`, and standard library data structures.

| Extra | Dependencies and capability |
|---|---|
| `datasets` | Hugging Face `datasets`, remote dataset adapters |
| `embeddings` | `sentence-transformers`, production cosine backend |
| `nli` | `torch`, `transformers`, bidirectional NLI and SelfCheck-NLI |
| `selfcheck-bertscore` | `bert-score` and model requirements |
| `judge` | OpenAI-compatible or local judge adapter (planned) |
| `evidence` | retrieval/indexing adapters |
| `whitebox` | `torch`, `transformers` |
| `evaluation` | `scikit-learn`, plotting optional |
| `all` | every supported optional feature |

Heavy models are never downloaded at import time. Every remote model, dataset, API request, or search source is opt-in.

## 9. Documentation, Examples, and Tests

### 9.1 Examples

Each example follows the textbook sequence: formula/assumption, runnable code, intermediate artifacts, known failure case, and a reader exercise.

```text
examples/
  00_result_contract_and_signal_types.ipynb
  01_whitebox_token_probability.ipynb
  02_whitebox_hidden_state_experimental.ipynb
  03_response_level_exact_match.ipynb
  04_response_consistency_cosine_and_entropy.ipynb
  05_selfcheck_sentence_baselines.ipynb
  06_blackbox_sampling_pipeline_and_backends.ipynb
  07_datasets_local_and_remote_adapter.ipynb
  08_sentence_benchmark.ipynb
  09_factscore_local_evidence.ipynb
  10_long_form_separated_workflow.ipynb
  11_optional_huggingface_embedding_nli_backends.ipynb
  12_threshold_calibration.ipynb
  13-22 real-backend integration notebooks
```

The first thirteen examples need an offline mode using `textbook_toy`; external-model variants are explicitly labeled with cost and setup requirements.

### 9.2 Tests

Tests verify the package, not merely demonstrate it. They must be deterministic, fast, and offline by default.

- unit tests: formulas, score direction, empty/missing input, parser failures, aggregation;
- backend tests: mocked embeddings/NLI/judges/retrievers, batch behavior, provenance;
- dataset tests: local formats, field mapping, labels, cache/offline behavior, dataset-card registry;
- benchmark tests: label alignment, metrics, filtering failures, validation/test separation;
- pipeline smoke tests: `DatasetRecord -> DetectionExample -> result -> benchmark report` using fixtures;
- integration tests: real downloadable models, APIs, and remote datasets, marked separately and not run by default.

No README command may call an unfinished scorer. Every claimed feature needs at least one test and one example.

## 10. Delivery Phases

### R0: Repair the current public contract

Status: complete.

Deliverables:

- remove/replace the empty `SelfCheckGPTScorer` and disable the nonfunctional long-form default;
- add `signal_type`, assumptions, warnings, provenance, and `UnitResult`;
- split reference match from sampled exact-match consistency;
- move demo embedding/NLI implementations out of public defaults;
- add `textbook_toy` fixtures, offline tests, and honest README status labels.

Implemented in this repository:

- `signal_type`, assumptions, warnings, provenance, `UnitResult`, and grouped signal results;
- `ReferenceMatchScorer` and `ExactMatchConsistencyScorer`, with a deprecated legacy wrapper;
- a black-box preset that requires explicitly supplied non-demo backends;
- local JSON/JSONL/CSV loading plus the packaged `textbook_toy` dataset and dataset card;
- removal of the empty public `SelfCheckGPTScorer` entry point;
- offline test coverage for the revised contracts, data loading, and preset behavior.

Exit criteria:

- no public preset raises `NotImplementedError`;
- no public default uses a demo backend silently;
- all core tests run without network or model downloads.

### R1: Dataset foundation and essential sampled black-box methods

Status: complete.

Deliverables:

- dataset schema, local loaders, Hugging Face loader, registry, card format, and `selfcheck_wikibio` adapter;
- benchmark runner and core sentence-level metrics;
- reusable sentence splitter and `BlackBoxSamplingPipeline` response/unit routing;
- production embedding backend and bidirectional NLI backend;
- response-level exact-match consistency;
- response-level cosine consistency and semantic entropy;
- SelfCheck n-gram, BERTScore, and NLI implementations.

Implemented in this repository:

- `ResponseCosineConsistencyScorer`, `ResponseSemanticEntropyScorer`, `SelfCheckNgramScorer`, `SelfCheckBERTScoreScorer`, and `SelfCheckNLIScorer`;
- greedy bidirectional-entailment clustering with official three-way equivalence for response semantic entropy;
- independent sentence segmentation and diagnostics for sampled passages in SelfCheck scorers;
- optional `SentenceTransformerEmbeddingBackend`, `BERTScoreSimilarityBackend`, and `BidirectionalNLIBackend`;
- the `selfcheck_wikibio` Hugging Face adapter and dataset card;
- `BenchmarkRunner` with offline sentence-level PR-AUC, ROC-AUC, F1, precision, recall, coverage, JSON, and CSV reports;
- deterministic offline examples and tests for the R1 workflow.

Exit criteria:

- reader can load `textbook_toy` offline and run all R1 examples;
- reader can opt in to the external WikiBio dataset and reproduce a sentence-level comparison report;
- all three SelfCheck variants return unit and response aggregates with diagnostic traces;
- response-level cosine and semantic entropy are separate from sentence-level SelfCheck and are documented as such.

### R2: External verification and long-form separation

Status: complete. MQAG and LLM-prompt SelfCheck were removed from scope by design.

Deliverables:

- claim decomposition, local-corpus retrieval, verifier interface, and FActScore-style result;
- local evidence corpus matching textbook examples;
- optional remote retrieval adapters;
- long-answer preset combining black-box sampling and optional evidence verification without mixing score semantics.

Implemented in this repository:

- `ClaimDecomposer`, with supplied-claim priority and an explicit deterministic sentence fallback;
- `LocalCorpusRetriever` plus a packaged Chinese textbook evidence corpus and transparent lexical-ranking trace;
- `NLIEvidenceVerifier`, which preserves `supported`, `contradicted`, and `unknown` as distinct verdicts;
- `FActScoreScorer` and `EvidenceVerificationPipeline`, returning claim evidence traces under the separate `evidence_support` signal;
- `LongFormVerificationPipeline`, which returns sampling consistency and evidence support as separate results after one sampling pass;
- deterministic offline example and tests covering support, contradiction, unknown, retrieval provenance, and sentence fallback.

Exit criteria:

- a long answer yields sentence risk, atomic-claim verdicts, evidence traces, and clearly separated self-consistency/evidence-support sections;
- `unknown` claim status is preserved;
- external services remain optional.

### R3: Calibration and textbook supplementary methods

Status: calibration and supplementary methods complete; weighted ensemble tuning remains planned.

Deliverables:

- threshold fitting, Platt scaling, isotonic regression, ECE/Brier/reliability report;
- P(True), CoVe detector, and LLM judge;
- weighted/calibrated ensemble configuration and serialization;
- reproducible comparison notebook and instructor-oriented exercise notes.

Implemented in this repository:

- `ThresholdCalibrator` with `f1`, `fbeta`, `balanced_accuracy`, and `cost` objectives;
- `ProbabilityCalibrator` with Platt, isotonic, and temperature scaling;
- `CalibrationReport` with ECE, MCE, Brier, NLL, and reliability tables;
- validation/test isolation enforced in code;
- `BenchmarkRunner` raw and calibrated metrics;
- `Detector` facade and `hallucination-detection` CLI;
- `PTrueScorer` with logprob and verbalized fallback;
- `CoVeScorer` with independent verification answers and NLI or lexical conflict checks;
- `LLMJudgeScorer` (binary/continuous/Likert) and `LLMPanelScorer`;
- `HuggingFaceWhiteBoxBackend.next_token_logprobs` for P(True);
- calibration tutorial (`examples/12_threshold_calibration.ipynb`) and tests.

Exit criteria:

- threshold/weight selection requires a validation split;
- results report calibration and data provenance;
- no method is described as universal or as a factual proof without external evidence.

### R4: Experimental white-box representation methods

Status: experimental core complete; independent benchmarking remains open.

Deliverables:

- repeated-generation hidden-state collector;
- revalidated EigenScore and Effective Rank implementations;
- benchmark comparisons with MTP/LNTP/entropy;
- model-specific configuration and warning documentation.

Implemented in this repository:

- `HuggingFaceWhiteBoxBackend.generate_hidden_states` for `[n_samples, hidden_dim]` collection;
- `EigenScoreScorer` with `mean_log` and `logdet` variants and `alpha=1e-3`;
- `EffectiveRankScorer` exposing spectral entropy, effective rank, and a normalized display value;
- token filtering and entropy coverage metadata.

Exit criteria:

- methods are reproducible on at least one documented local model and dataset;
- experimental status stays visible until independently benchmarked across settings.

## 11. Release Gates

Before the textbook release:

1. all advertised methods, datasets, examples, and optional extras have accurate availability labels;
2. a clean installation can run offline core tests and the toy tutorial;
3. every remote asset has source, license, version, and cache documentation;
4. every result exposes enough artifacts to reconstruct its score;
5. all methods document required model access, latency/cost, language scope, and failure cases;
6. benchmark reports distinguish factual support from model self-consistency;
7. package documentation matches the textbook terminology and equations.

## 12. First Implementation Slice

Implement R0 before adding methods:

1. revise result schemas and score grouping;
2. replace the old long-form `SelfCheckGPTScorer` path with the scaffold for `BlackBoxSamplingPipeline` unit scorers;
3. split `ExactMatchScorer` semantics;
4. add `DatasetRecord`, local JSONL loader, and `textbook_toy` data;
5. add offline tests for the new contracts;
6. update README so its runnable examples describe only available functionality.

The next slice is R1 SelfCheck-NLI plus the dataset benchmark example, since it best demonstrates the textbook's black-box consistency progression while reusing a durable pipeline architecture.
