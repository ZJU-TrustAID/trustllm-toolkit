# Hallucination Mitigation Toolkit

`hallucination.mitigation` is the mitigation companion to `hallucination.detection`.
It provides inference-time methods that change the model output instead of only scoring it.

The final acceptance target is **real usage**: a Hugging Face model id, a local model
directory, or a real model API. `demo --offline` is only an installation smoke test.

## Methods

| Method | Type | Needs fit | Needs white-box | API models |
|---|---|---|---|---|
| DoLa | layer-contrast decoding | no | yes | no |
| CAD | context-aware decoding | no | yes | no |
| CAA | layer-wise activation steering | yes (mean diff) | yes | no |
| ITI (optional) | per-head probe steering | yes | yes | no |
| CoVe | verification + revision | no | no | yes |

## Install

```bash
python -m pip install -e ".[hf]"
# optional:
python -m pip install -e ".[iti]"      # ITI probes
python -m pip install -e ".[openai]"   # CoVe via API
```

## Quickstart: real Hugging Face model

```bash
hallucination-mitigation run --method dola \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "请介绍蓝鲸的食性。"
```

## Quickstart: local model, offline

```bash
hallucination-mitigation run --method cad \
  --model /models/Qwen2.5-1.5B-Instruct --local-files-only \
  --context "阿根廷在 1978、1986 和 2022 年赢得了世界杯。" \
  --prompt "阿根廷赢过多少次世界杯？"
```

## Quickstart: model API (CoVe only)

```bash
export OPENAI_API_KEY="sk-..."
hallucination-mitigation run --method cove --backend openai \
  --model gpt-4o-mini --prompt "请介绍蓝鲸的食性。"
```

OpenAI-compatible local servers are supported for CoVe:

```bash
export HM_API_KEY="local-key"
hallucination-mitigation run --method cove --backend openai-compatible \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-base http://localhost:8000/v1 --api-key-env HM_API_KEY \
  --prompt "请介绍蓝鲸的食性。"
```

## Fit CAA

```bash
hallucination-mitigation fit-steering \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data examples/data/hallucination_pairs_demo.jsonl \
  --layer 12 --chat --out runs/caa.npz

hallucination-mitigation run --method caa \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steering runs/caa.npz --prompt "请介绍蓝鲸的食性。"
```

## Benchmark

Bring your own JSONL with `prompt`, optional `context`, and `reference` / `references`, or run standard benchmarks (`--dataset truthfulqa_gen|nq_open|nq_swap`) with paired significance output; the shipped validation study is in `docs/mitigation-validation.md`:

```bash
hallucination-mitigation benchmark \
  --data examples/data/context_demo.jsonl \
  --method cad --model Qwen/Qwen2.5-1.5B-Instruct \
  --metric contains --limit 10 --out runs/cad.md
```

## Python API

```python
from hallucination.mitigation import Mitigator

m = Mitigator(method="dola", model="Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
report = m.mitigate("请介绍蓝鲸的食性。")
print(report.to_markdown())
report.to_json("runs/dola.json")
```

## Status

See [STATUS.md](STATUS.md) for implemented features and pending release work, and
[SMOKE.md](SMOKE.md) for real Qwen2.5-0.5B CPU smoke results.

## Scope

- v0.1 targets Llama-family and Qwen2.5 causal LMs on a single device.
- DoLa/CAA/ITI/CAD require a local white-box model; API models are supported only by CoVe.
- Mitigation is not a guarantee of factual correctness. The report always includes warnings.
