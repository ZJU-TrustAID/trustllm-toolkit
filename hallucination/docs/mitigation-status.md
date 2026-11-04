# Status: 0.1.0 implementation

## Implemented

- Package scaffold with `src/` layout, hatchling build, CLI entry point.
- `Mitigator` facade with `run` and `compare`.
- `HFEngine`: prefill/step KV cache, hidden states, sampling, hooks.
- DoLa layer-contrast decoding.
- CAD context-aware decoding with two KV caches.
- CAA fit/apply with JSON/NPZ steering artifacts.
- ITI fit/apply with per-head probes (optional `[iti]`).
- CoVe verification/revision with HF, OpenAI-compatible, and callable backends.
- CLI: `run`, `fit-steering`, `fit-iti`, `benchmark`, `list-methods`, `list-presets`, `doctor`.
- Presets for Qwen2.5-1.5B, TinyLlama, Llama-2, OpenAI, vLLM.
- JSONL benchmark with exact/contains metrics, win/tie/loss, and paired
  significance (exact binomial test, Wilson CI, Holm correction).
- Standard benchmark adapters: TruthfulQA generation, NQ-Open, NQ-Swap
  (Hub mirror or official test.json); deterministic seeded subsampling.
- Paired validation study on Qwen2.5-0.5B-Instruct (n=40 per dataset):
  see docs/mitigation-validation.md.
- Real Qwen2.5-0.5B-Instruct CPU smoke:
  - DoLa 8 tokens: ~4.6 s; CAD 8 tokens: ~3.4 s; CAA run: ~2.1 s (fit separate); CoVe: 15.1 s / 5 calls / 2 verification questions; ITI run: ~1.5 s (fit separate; demo data is only 12 samples, accuracy near chance).
- Real Qwen2.5-1.5B-Instruct CPU comparison (64 tokens):
  - DoLa: baseline correct, mitigated hallucinated fish/marine mammals (58.6 s) — degradation.
  - CAD: baseline rambling, mitigated correctly answered "3 World Cups" (46.4 s) — improvement.
  - CAA: baseline correct "small crustaceans", mitigated "large crustaceans" (37.3 s) — over-correction.
  - CoVe: baseline factual error, mitigated correct facts but answered in English (84.5 s / 9 calls).
  - Conclusion: methods can help or hurt; baseline-vs-mitigated reporting and validation are mandatory.
- Held-out micro-benchmarks (2–3 examples, 16 tokens; not statistically meaningful):
  - CAD factual context: baseline 1.000 vs CAD 1.000 (ceiling effect).
  - CAD conflict context: baseline 0.333 vs CAD 0.333 (strong priors not overridden).
  - CoVe general QA: baseline 0.500 vs CoVe 0.500 after adding same-language prompts (initial run was 0.000 because the answer switched to English).
- Fixed issues found by real runs: HookManager did not propagate `active` to hooks; CoVe did not parse Chinese fullwidth `？`; ITI assumed equal chat-sequence lengths.
- Offline tests: 44 passed (including tiny Llama engine, DoLa, CAD, CAA, ITI, CoVe).
- Wheel builds and includes preset YAML files.

## Pending for a full 0.1.0 release

- CNN-DM adapter (deliberately deferred: faithful summarization scoring
  needs ROUGE or an LLM judge, not the contains metric).
- `demo --offline` command (not an acceptance requirement).
- Real 7B/8B GPU validation and latency/VRAM benchmarks.
- Docs (`quickstart`, `backends`, `methods`, `fidelity`, `troubleshooting`).
- GitHub Actions CI, coverage, build/PyPI release workflow.
- More tests for API error mapping and real model smoke.
- `bake=True` for ITI (currently rejected explicitly).
- `pyproject` package-data policy review and clean-venv wheel test.
