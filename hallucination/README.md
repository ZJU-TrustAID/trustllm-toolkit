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

**设计目标：同一页面，双模式可切换（同越狱攻击工具包）。** 页面顶部固定一条
「选择模式」横条，要求在**同一个页面**内提供两个版本并随时切换：

- **01 预设回放**（离线）— 浏览预置教学场景，不加载模型、不占 GPU，方法结果
  全部来自离线预设目录；
- **02 实时模型**（在线）— 真后端工作台：与真实模型对话，并对其回答运行检测 /
  缓解；模型服务未启动时该按钮小字明确标示（每 5 秒轮询 `GET /api/real/status`：
  模型名 / 待加载 / 未配置模型 / 模型服务离线）。

当前状态：模式横条已就位且默认停在「预设回放」，但**切换尚未接通功能**——
点击只更新右侧说明文字，页面主体仍是实时工作台。把预设回放版本接入同一页面
是下一步的既定工作，所需素材均已保留并在服务端可用（`static/tutorial.js`
预设前端 + `GET /api/catalog`、`POST /api/method` 离线接口，零模型调用），
届时只需挂载素材并在两种面板间切换，布局模板见
[web_demo/README.md](web_demo/README.md)。

实时工作台由五个卡片构成（模型问答 / 选择方法 / 参数设置 / 输出结果 /
检测结果）：

1. **模型问答**（left, spans two rows）— streaming chat with the configured
   model. Every answer is registered as a record and its bubble can be clicked
   as the detection target; otherwise the latest answer is used. A context
   textarea sits **above the prompt** (shown in the mitigation category only):
   selecting CAD turns the card green, marks the context as required with a
   hint, and blocks the run with a toast before hitting the backend (the
   backend rejects empty context); other mitigation methods may leave it empty.
2. **选择方法**（top right）— the three categories (白盒检测 / 黑盒检测 /
   幻觉缓解) are collapsible groups styled like the Jailbreak toolkit's method
   groups (01/02/03, accordion: expanding one switches the category and resets
   the selected method). **On first load / refresh all three start collapsed.**
   Inside each group, methods are grouped by the book's taxonomy. Plus the
   gated 「开始检测 / 缓解」button (unlocks once a method is selected, numeric
   params are filled and any required sampling is done) and a paper reference
   line linking to the matching textbook section, e.g.
   `…/chapter-6/6-5/#基于词元概率检测`.
3. **参数设置**（bottom right）— sampling fields (样本数 / 隐状态层) with the
   「抽样」button and per-method extra params; defaults are light-grey
   placeholders (visibly lighter than filled values) and are never auto-filled
   or silently sent. Methods without extra params show a centered hint.
4. **输出结果**（full width, shown only for sampling-based methods）— the
   sampled independent answers.
5. **检测结果**（full width）— results are archived per answer record with
   pill tabs to switch between methods of the same answer; switching answers
   switches the result set, switching tabs never clears it.

The run button's hint line doubles as guidance: with a prompt already sent and
a method selected, if the button is disabled only because numeric params are
missing it reads 「先在下方设置参数，再开始运行」.

The workbench calls `api/real/*` with a real model (local path / HF repo /
OpenAI-compatible API) and returns results identical to the toolkit CLI.

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
