# Changelog

All notable changes to `hallucination-toolkit` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- Web demo **tutorial mode** (`web_demo/`): every implemented detection method
  (14) and mitigation method (5)
  is runnable on preset teaching scenarios with
  zero model calls — the toolkit's real scorers and decoding math
  (`dola.jsd`/`apc_mask`/`contrast_logits`, `cad.combine_context_logits`,
  the real CoVe chain) execute on canned data. The UI mirrors the book
  chapter order (white-box → black-box signals → DoLa/ITI → CAD → steering &
  verification) with per-method mechanism panels, baseline-vs-mitigated
  comparison cards, and token-level annotations. New endpoints
  `GET /api/catalog` and `POST /api/method`; the former interactive
  workbench (`POST /api/generate`) was removed together with its tests.
  Tests: `tests/detection/test_web_demo_tutorial.py`.

### Removed

- **Breaking:** `ReferenceMatchScorer` (`reference_match`) and the deprecated
  `ExactMatchScorer` wrapper. Comparing a response against an external answer
  key is benchmark scoring rather than hallucination detection, so it no
  longer ships as a scorer. Consequently
  `build_blackbox_short_answer_pipeline` lost its `include_reference_match`
  argument, and detection notebook `02` was renamed to
  `02_real_exact_match_consistency_openai.ipynb` (exact-match consistency
  only). `DetectionExample.reference_answer` remains part of the dataset
  schema.

## 0.1.0 (2026-10-26)

Merge of the two former standalone packages into one distribution:

- `hallucination-detection` 0.3.0 → `hallucination.detection`
- `hallucination-mitigation` 0.1.0 → `hallucination.mitigation`

Breaking changes relative to the former packages:

- Import paths changed from `hallucination_detection` / `hallucination_mitigation`
  to `hallucination.detection` / `hallucination.mitigation`.
- `requires-python` raised to `>= 3.10` (the former detection package allowed 3.9).
- The mitigation `detect` extra is gone: the detection subpackage ships in the
  same distribution.
- CLI entry points keep their names (`hallucination-detection`,
  `hallucination-mitigation`); module execution moved to
  `python -m hallucination.detection` / `python -m hallucination.mitigation`.
- Single version line for the whole toolkit; one CHANGELOG, one release tag
  scheme (`vX.Y.Z`).

Feature set is the union of both former packages at the versions above; see
[docs/detection.md](docs/detection.md) and [docs/mitigation.md](docs/mitigation.md).

### Detection (from `hallucination-detection` 0.3.0)

- White-box MTP, LNTP, token entropy, experimental EigenScore / effective rank.
- Response-level exact-match, cosine, semantic entropy; sentence-level
  SelfCheck n-gram / BERTScore / NLI; FActScore-style evidence verification.
- P(True), CoVe detector, LLM-as-a-Judge and judge panels.
- Per-method calibration bundles (threshold, Platt/isotonic/temperature,
  ECE/Brier/NLL), offline benchmark runner, `Detector` facade, CLI,
  dataset adapters, 23 tutorials, local web demo.

### Mitigation (from `hallucination-mitigation` 0.1.0)

- DoLa layer-contrast decoding; CAD context-aware decoding with dual KV caches.
- CAA activation steering with JSON/NPZ artifacts; ITI per-head probe steering.
- CoVe verification-and-revision (HF, OpenAI-compatible, callable backends).
- CLI (`run`, `fit-steering`, `fit-iti`, `list-methods`, `list-presets`,
  `doctor`, `benchmark`), model presets, baseline-vs-mitigated comparison.
- Statistical validation: paired win/tie/loss significance (exact
  binomial test, Wilson CI, Holm correction), standard benchmark adapters
  (TruthfulQA generation, NQ-Open, NQ-Swap) with deterministic seeded
  subsampling, and a paired validation study on Qwen2.5-0.5B-Instruct
  (docs/mitigation-validation.md).
