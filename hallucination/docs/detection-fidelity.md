# Fidelity notes: textbook, official code, and this toolkit

This document records the reference implementation and the intentional deviations for every implemented detection method. The goal is to make teaching claims auditable, not to claim byte-for-byte reproduction.

## Reference repositories

| Method family | Official repository | Commit used for review |
|---|---|---|
| SelfCheckGPT (BERTScore, MQAG, n-gram, NLI, LLM prompt) | https://github.com/potsawee/selfcheckgpt | `19b492a` |
| FActScore | https://github.com/shmsw25/FActScore | `f28272d` |
| Semantic entropy / P(True) | https://github.com/jlko/semantic_uncertainty | `a8d9aa8` |
| EigenScore / INSIDE | https://github.com/alibaba/eigenscore | `ea8062a` |
| Effective Rank | https://github.com/1240148048/Effective-Rank-based-Uncertainty | `d585b99` |
| Engineering reference | https://github.com/cvs-health/uqlm | v0.6.7 (`694ea3e`) |

MQAG, LLM-prompt SelfCheck, and trained text classifiers are intentionally not implemented; see the README.

## Method-by-method deviations

### MTP / LNTP

- Official: minimum and length-normalized token probability over the generated response.
- This toolkit: same formulas, plus an optional `TokenFilter` for special tokens, punctuation, and leading/trailing tokens.
- Deviation: when token metadata is absent, filtering is skipped rather than guessed.

### Token entropy

- Textbook: Shannon entropy over the next-token distribution.
- This toolkit: entropy over the available top-k distribution, normalized by `log(k)` when `k > 1`; `distribution_coverage` records retained mass.
- Deviation: full-vocabulary entropy is only available when the backend supplies it; top-k is an approximation. A one-candidate step is skipped by default because the backend truncated the distribution; `single_candidate_policy="zero"` preserves the legacy zero-entropy behaviour for explicit comparisons.

### EigenScore

- Official: the INSIDE paper defines EigenScore as `(1/K) sum log(lambda_i)` over the `K x K` covariance of sentence embeddings. The INSIDE repository computes `mean(log10(lambda + alpha))` on `np.cov` (a `d x d` matrix); the Effective Rank repository computes `sum(log(lambda + alpha))` (natural log) on `np.cov`.
- This toolkit: `variant="mean_log"` (default) or `"logdet"`, `alpha=1e-3`, Gram-matrix spectrum, optional first-principal-component removal.
- Deviation: `mean_log` uses log10 like the INSIDE repository while the paper uses natural log; a constant log-base factor preserves ranking but changes the raw score and sigmoid risk. `logdet` uses natural log like the Effective Rank repository, but sums over `K` Gram eigenvalues rather than `d` covariance eigenvalues. Raw scores are therefore not numerically comparable across implementations. The `feature_clip` option removes the first principal component, not the INSIDE test-time activation clipping; use it as a documented preprocessing choice, not as an official reproduction.

### Effective Rank

- Official: `H = -sum p_i log p_i` with `p_i = sigma_i / sum sigma_j`; the reference returns `H`.
- This toolkit: reports both `spectral_entropy` (`H`, natural log) and `effective_rank = exp(H)`; the normalized rank is a display convenience.
- Deviation: the official code uses a mid-layer window across generations; the toolkit accepts any `[n_samples_or_tokens, hidden_dim]` matrix and documents the expected protocol.

### Exact-match consistency

- Textbook / official: strict equality with each sample.
- This toolkit: `normalize=True` by default (case, whitespace, punctuation), with `normalize=False` for strict equality; `raw.normalized` records the choice.

### Response cosine consistency

- Textbook / UQLM: mean cosine between the whole response and each whole sample.
- This toolkit: same, with `(cos + 1) / 2` mapping by default; raw cosines are retained.
- Deviation: UQLM maps to `[0, 1]`; this toolkit exposes both raw and mapped values. It does not perform sentence alignment.

### Response semantic entropy

- Official: greedy representative clustering, bidirectional three-way equivalence, entropy over cluster frequencies.
- This toolkit: same greedy clustering and official three-way rule; `strict` and probability-threshold variants are available; `include_response=True` matches UQLM's `m+1` candidates, while `include_response=False` uses only samples.
- Deviations: cluster probabilities are frequency-based (no token-probability weighting); entropy is reported in bits by default; normalized confidence is `1 - H / log_base(n)`.

### Sentence-level SelfCheck

- N-gram: official trains on the original response and samples, returns average and maximum negative log-probability. This toolkit does the same and adds a numerical floor for unseen n-grams; the `[0,1]` score is a monotone display mapping of the raw NLL.
- BERTScore: official takes the best matching sampled sentence per sample. This toolkit matches that; `min_sentence_length` is configurable.
- NLI: official feeds `(sentence, sample)` to the NLI model. This toolkit defaults to that order; `premise="sample"` preserves the earlier toolkit behaviour.

### P(True)

- Official: few-shot brainstormed answers + `A) True` / `B) False`; `get_p_true` reads only the logprob of the ` A` continuation, and `p_false_fixed = 1 - exp(p_true)` confirms no A/B normalization.
- This toolkit: same A/B prompt skeleton without the official few-shot exemplars; `normalize_ab=True` (default) normalizes over A/B, while `normalize_ab=False` reproduces the official "only ` A`" estimator. A logprob backend (`next_token_logprobs`) is used when available, otherwise a temperature-zero verbalized True/False answer is parsed and labelled `raw["method"]="verbalized"`.
- Deviation: no few-shot exemplars; the verbalized fallback is not the same estimator as logprob P(True); both are uncalibrated self-reports.

### CoVe detector

- Official: generate verification questions, answer them independently, compare with the draft, and optionally revise.
- This toolkit: detection only; reports the fraction of verification answers that conflict with the draft. No answer rewrite.
- Deviation: the official revision step is excluded, and the default verifier is an optional NLI backend; without one, a labelled lexical-overlap heuristic is used.

### LLM-as-a-Judge

- Official/reference: UQLM-style categorical, continuous, and Likert judges; panels average multiple judges.
- This toolkit: JSON-only judge prompts, parsing with trace, binary/continuous/Likert modes, and an averaging panel.
- Deviation: no self-consistency sampling of judge outputs and no judge-specific calibration; judge outputs are not factual probabilities.

### FActScore-style verification

- Official: LLM atomic-fact decomposition, Wikipedia retrieval, ChatGPT True/False verification, length penalty `gamma=10`.
- This toolkit: deterministic claim fallback or supplied claims, local lexical retrieval, NLI three-way verification, supported/contradicted/unknown ratios. By default `risk = 1 - supported_ratio` (unknown is not supported); `unknown_policy="contradicted"` keeps the older "unknown is not a contradiction" behaviour.
- Deviation: this is a teaching-grade, offline FActScore-style path; it is not the official FActScore implementation and does not include the length penalty or LLM verifier. Treating unknown as unsupported follows the textbook's binary Supported/Not-supported definition rather than the official ChatGPT verifier's granularity.

### Reference matching

- Textbook EMR: strict equality between the response and a candidate answer.
- This toolkit: `ExactMatchConsistencyScorer` implements sample-level EMR (strict or normalized). Comparing a response against an external answer key is benchmark scoring, not a detection method, and is out of scope for the scorers.

### Calibration

- UQLM reference: one `ScoreCalibrator` per scorer, Platt/isotonic calibration and ECE evaluation; `Tuner` tunes ensemble weights and a threshold.
- This toolkit: threshold selection (`f1`, `fbeta`, `balanced_accuracy`, `cost`), Platt/isotonic/temperature calibration, ECE/MCE/Brier/NLL, reliability tables, validation/test isolation. `CalibrationBundle` binds a calibrator to a method name and granularity; `CalibrationRegistry` holds one bundle per method, mirroring UQLM's per-scorer dictionary. `Detector(calibration=...)` and `BenchmarkRunner(calibration=...)` apply only matching bundles and refuse cross-method application.
- Deviation: no plotting dependency is required; reliability tables are returned as data. No ensemble-weight tuner is included; aggregation remains mean/logit-mean over calibrated risks.
