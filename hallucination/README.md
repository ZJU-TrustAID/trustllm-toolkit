# Hallucination Toolkit

`hallucination-toolkit` is the executable companion to the hallucination
chapters of *Trustworthy Large Language Models*. It ships two subpackages with
one install:

- **`hallucination.detection`** — quantifies hallucination risk: white-box
  confidence signals (MTP, LNTP, token entropy, EigenScore, effective rank),
  black-box consistency (exact match, cosine, semantic entropy, SelfCheck),
  evidence verification (FActScore-style), P(True), CoVe detector and LLM
  judges, plus per-method calibration bundles, benchmarking, a CLI and a local
  web demo. It **detects and reports risk; it never rewrites answers**.
- **`hallucination.mitigation`** — inference-time methods that **change the
  model output**: DoLa (layer-contrast decoding), CAD (context-aware
  decoding), CAA (activation steering), ITI (per-head probe steering) and CoVe
  (verification + revision), with a `Mitigator` facade, CLI, presets and
  baseline-vs-mitigated comparison reporting.

Detailed guides: [docs/detection.md](docs/detection.md) ·
[docs/mitigation.md](docs/mitigation.md) ·
[fidelity notes](docs/detection-fidelity.md) ·
[mitigation status & real-model smoke results](docs/mitigation-status.md)

> Mitigation is not a guarantee of factual correctness. A paired, significance-tested
> validation study (Qwen2.5-0.5B, n=40 per benchmark) shows CAD significantly improving
> context-faithfulness on NQ-Swap while DoLa shows no significant effect on factuality
> (see [docs/mitigation-validation.md](docs/mitigation-validation.md)); the report always
> includes warnings, and baseline-vs-mitigated validation is mandatory.

## Install

Requires Python >= 3.10.

```bash
python -m pip install -e .                     # light core: numpy, pyyaml, tqdm
python -m pip install -e ".[dev]"              # + pytest, ruff, build, twine
```

Optional extras (install by capability):

| Extra | Provides |
|---|---|
| `whitebox` / `nli` | local Hugging Face causal LM / NLI backends (torch, transformers) |
| `embeddings` | sentence-transformer cosine consistency |
| `selfcheck-bertscore` | SelfCheck BERTScore |
| `calibration` / `pca` / `iti` | scikit-learn calibration, CAA PCA, ITI probes |
| `hf` | full local mitigation stack (torch, transformers, accelerate) |
| `openai` | OpenAI / OpenAI-compatible API backends |
| `datasets` | remote dataset adapters (SelfCheckWikiBio) |
| `dataframe` | `DetectionReport.to_dataframe()` |
| `plots` | reliability diagrams and plots |
| `quant` | bitsandbytes quantized loading |

## Detection quickstart

```python
from hallucination.detection import Detector

detector = Detector(preset="offline_toy")
report = detector.detect("请介绍蓝鲸的食性。")
print(report.to_markdown())
```

CLI (deterministic, offline):

```bash
hallucination-detection detect --preset offline_toy --prompt "请介绍蓝鲸的食性。"
hallucination-detection benchmark --dataset textbook_toy
hallucination-detection list-methods
```

All scorers return `score`/`risk` in `[0, 1]` tagged with a `signal_type`
(generation confidence, self-consistency, evidence support, self-reported
confidence, judge assessment). Self-consistency is not factual verification.
See [docs/detection.md](docs/detection.md) for the full method/granularity
matrix, calibration bundles and result semantics.

## Mitigation quickstart

```bash
hallucination-mitigation run --method dola \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "请介绍蓝鲸的食性。"

hallucination-mitigation run --method cad \
  --model /models/Qwen2.5-1.5B-Instruct --local-files-only \
  --context "阿根廷在 1978、1986 和 2022 年赢得了世界杯。" \
  --prompt "阿根廷赢过多少次世界杯？"
```

Python API:

```python
from hallucination.mitigation import Mitigator

m = Mitigator(method="dola", model="Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
report = m.mitigate("请介绍蓝鲸的食性。")
print(report.to_markdown())
```

DoLa/CAA/ITI/CAD need a local white-box model; API models are supported only
by CoVe. Method presets, steering fits (`fit-steering`, `fit-iti`) and the
JSONL benchmark are documented in [docs/mitigation.md](docs/mitigation.md).

## Web demo

```bash
python web_demo/server.py    # http://127.0.0.1:8765
```

A MarkSec-style **教程模式** tutorial across three pages (白盒检测 /
黑盒检测 / 缓解方法): click the prompt dialog to pick a prompt, choose a
method in the right-hand option group, and 生成 simulates the streamed
generation before rendering the method's result. The page never calls a
model — real scorers and decoding math run on packaged teaching data
(14 detection + 5 mitigation methods); run the CLI/notebooks locally for
real inference. See [web_demo/README.md](web_demo/README.md).

## Examples

`examples/detection/` holds 13 real-backend tutorial notebooks (`00`–`12`),
one per API- or local-model-capable detection method, each following the same
shape: configure a real backend, build the pipeline/scorers, run interactively
(`input()`), and read the structured results.
`examples/mitigation/` holds 5 real-model tutorials (DoLa, CAD, CAA, ITI,
CoVe) plus the demo JSONL datasets for steering fits and benchmarks.
Regenerate either set with the `build_*_notebooks.py` scripts.

## Tests

```bash
python -m pytest tests
```

Default tests are deterministic, offline and require no model downloads.

## Scope

- v0.1 targets Llama-family and Qwen2.5 causal LMs on a single device.
- Not included: MQAG, LLM-prompt SelfCheck, trained text-classifier baselines.
- See [CHANGELOG.md](CHANGELOG.md) for how the former `hallucination-detection`
  0.3.0 and `hallucination-mitigation` 0.1.0 packages map onto this release.
