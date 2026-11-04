# Mitigation validation study

This document reports a **paired, significance-tested validation** of the
inference-time mitigation methods against regular decoding, on standard
hallucination benchmarks, with a real model. It replaces the earlier
2–3-example smoke runs as the toolkit's effectiveness evidence.

## Methodology

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`, CPU, float32, greedy decoding
  (`do_sample=False`), fixed seed. Every example is generated twice — once
  with regular decoding (baseline) and once with the mitigation — so the
  comparison is paired and model/sampling noise cancels.
- **Datasets** (deterministic 40-example subsets, `seed=0`, via
  `hallucination.mitigation.evaluation.datasets`):
  - **TruthfulQA generation** (`truthfulqa/truthful_qa`): adversarial
    factuality; references are the curated correct answers.
  - **NQ-Open** (`google-research-datasets/nq_open` validation): open-domain
    factoid QA; references are the gold short answers.
  - **NQ-Swap** (`pminervini/NQ-Swap` mirror of the official substitution
    set): the context's answer has been swapped, so following the context
    contradicts parametric memory. This is the designed benchmark for CAD;
    both baseline and CAD see the same swapped context.
- **Metric**: `contains` — a normalized gold answer must appear in the
  generation. Crude (paraphrases score 0) but symmetric between baseline and
  mitigation, which is what a paired design needs.
- **Statistics** (`evaluation/stats.py`): ties are dropped; the decisive win
  rate is reported with a Wilson 95% interval and an exact two-sided binomial
  test (identical to McNemar's exact test for binary per-example outcomes).
  Holm adjustment is applied across the three comparisons. No normal
  approximations; n=40 keeps these studies **exploratory**, not confirmatory.

## Results (Qwen2.5-0.5B-Instruct, CPU, n=40 per dataset)

| Method | Dataset | Baseline | Mitigated | Δ | win/tie/loss | Decisive win rate [95% CI] | Exact binomial p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **CAD** (α=1.0) | NQ-Swap | 0.175 | **0.425** | **+0.250** | 13/24/3 | 0.812 [0.570, 0.934] | 0.0213 | 0.064 | significant per-comparison; borderline under Holm |
| DoLa | NQ-Open | 0.075 | 0.100 | +0.025 | 2/37/1 | 0.667 [0.208, 0.939] | 1.000 | 1.000 | no significant difference |
| DoLa | TruthfulQA | 0.075 | 0.025 | −0.050 | 1/36/3 | 0.250 [0.046, 0.699] | 0.625 | 1.000 | no significant difference |

**What this says:**

- **CAD works in its designed setting.** On NQ-Swap — where the context's
  answer was swapped against parametric memory — CAD raised contains-accuracy
  from 17.5% to 42.5% (+25 points), winning 13 of 16 decisive comparisons
  (p = 0.021). This is the first statistically-grounded positive result for
  the toolkit's headline context-faithfulness method. Under Holm correction
  across the three comparisons it is borderline (p_holm = 0.064).
- **DoLa shows no significant effect on factuality**, and its TruthfulQA
  direction is numerically *negative* (7.5% → 2.5%) — consistent with the
  earlier 1.5B smoke run where DoLa degraded a correct baseline. The toolkit's
  own warning ("layer buckets must be validated per task") is corroborated:
  DoLa should not be treated as a safe default.
- Only 3–4 decisive pairs formed for DoLa: the 0.5B model + contains metric
  produces few informative comparisons, and the wide CIs say the DoLa studies
  are underpowered rather than evidence of "no effect".

Raw per-example outputs, win/tie/loss and full significance blocks:
`runs/validation/{dola,cad}_*_{truthfulqa_gen,nq_open,nq_swap}_qwen05b.json`.

## Reproduce

```bash
python -m pip install -e ".[hf,datasets]"
python examples/mitigation/validation_study.py     # ~35 min on 8-core CPU
```

Subsets are rebuilt deterministically from the adapters (`limit=40, seed=0`);
raw per-example outputs, win/tie/loss and full significance blocks are in
`runs/validation/*_qwen05b.json`.

## Scope and limitations

- CAA and ITI are **not** validated here: both need fitted artifacts on
  labeled true/false pairs, and no curated training set ships with the
  toolkit. CoVe needs an API key. Building those studies is future work.
- One 0.5B model on CPU, `contains` scoring, n=40: effects that are real but
  small will not reach significance. "No significant difference" is not
  evidence of no effect.
- Multiple methods × datasets are compared; per-comparison p-values are
  Holm-corrected, but the study family is still small.
