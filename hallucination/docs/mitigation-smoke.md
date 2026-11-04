# Real-model smoke (Qwen2.5-0.5B and 1.5B, CPU)

Date: 2026-10-25  
Models: `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` (local directories)  
Runtime: CPU, `float32`, `transformers 5.20.2`, `torch 2.14.1+cpu`  
Purpose: verify real generation, KV cache, hooks and reporting. These are **smoke
results, not a quality benchmark**.

## Commands

```bash
# DoLa
hallucination-mitigation run --method dola \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --device cpu --dtype float32 \
  --prompt "请介绍蓝鲸的食性。" --max-new-tokens 8 \
  --format json --out runs/real_dola_qwen.json

# CAD
hallucination-mitigation run --method cad \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --device cpu --dtype float32 \
  --context "阿根廷在 1978、1986 和 2022 年赢得了世界杯。" \
  --prompt "阿根廷赢过多少次世界杯？" --max-new-tokens 8 \
  --format json --out runs/real_cad_qwen.json

# CAA fit + run
hallucination-mitigation fit-steering \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --device cpu --dtype float32 \
  --data examples/data/hallucination_pairs_demo.jsonl \
  --layer 12 --chat --out runs/real_caa_qwen.npz

hallucination-mitigation run --method caa \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --device cpu --dtype float32 --steering runs/real_caa_qwen.npz \
  --layer 12 --multiplier 1.0 \
  --prompt "请介绍蓝鲸的食性。" --max-new-tokens 8 \
  --format json --out runs/real_caa_qwen_run.json

# CoVe
hallucination-mitigation run --method cove \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --device cpu --dtype float32 \
  --prompt "请介绍蓝鲸的食性。" --max-new-tokens 16 \
  --format json --out runs/real_cove_qwen.json

# ITI fit + run (optional extra)
hallucination-mitigation fit-iti \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --device cpu --dtype float32 \
  --data examples/data/iti_demo.jsonl --top-k 8 --alpha 1.0 --chat \
  --out runs/real_iti_qwen.json

hallucination-mitigation run --method iti \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --device cpu --dtype float32 --probe runs/real_iti_qwen.json \
  --alpha 1.0 --prompt "请介绍蓝鲸的食性。" --max-new-tokens 8 \
  --format json --out runs/real_iti_qwen_run.json
```

## Results

### DoLa

- baseline: `蓝鲸是海洋哺乳动物`
- mitigated: `蓝鲸是海洋哺乳动物`
- latency: 4.58 s; 8 generated tokens; 9 forward passes
- trace: premature layer changes per step (22, 18, 16, 22, 10, …) with JSD ≈ 0.68–0.69

The 8-token budget is too short to change the answer; the run verifies hidden-state
projection, JSD selection, APC masking and logits contrast on a real model.

### CAD

- context: Argentina won the World Cup in 1978, 1986 and 2022
- baseline: `2022 年，` (truncated)
- mitigated: `这个问题的答案是 3`
- latency: 3.37 s; 8 generated tokens; 18 forward passes (2 per token)

CAD produced the correct short answer while the baseline was truncated in the same
budget. This is a single example, not a benchmark.

### CAA

- fit: 6 positive/negative pairs, layer 12, chat template
- artifact: `runs/real_caa_qwen.npz` (norm 1.0)
- baseline/mitigated: `蓝鲸是海洋哺乳动物`
- latency: 2.06 s; 8 hook calls; 8 generated tokens

The run verifies fitting, artifact loading, layer hook activation and reporting.
Multiplier 1.0 with 6 pairs is a smoke setting, not a recommended production value.

### CoVe

- baseline: `蓝鲸是海洋哺乳动物，属于鲸目，属于齿鲸科。`
- questions: `蓝鲸的食性是什么？`, `蓝鲸属于哪类动物？`
- verification answers: `蓝鲸的食性是肉食性。`, `蓝鲸属于哺乳动物。`
- response: same as baseline
- latency: 15.13 s; 5 calls; 2 verification questions

The first run failed to parse the plan because it used Chinese fullwidth `？`; the
parser now accepts both `?` and `？`. The final response was unchanged because the
baseline was already acceptable for the short budget.

### ITI (optional)

- fit: 12 labeled samples, 8 heads, chat template
- best validation accuracy: **0.333** (near chance; the demo data is far too small)
- baseline/mitigated: `蓝鲸是海洋哺乳动物`
- latency: 1.53 s; 8 hook calls; 8 generated tokens

This verifies the end-to-end probe artifact and per-head hook path. It is **not** a
quality result; a real ITI run needs a few hundred TruthfulQA-style examples per model.

## Qwen2.5-1.5B comparison (64 tokens, CPU)

Model: `Qwen2.5-1.5B-Instruct` (2.87 GB safetensors), local directory, CPU float32,
64 new tokens. CPU cost is high; this is a behavior check, not a quality benchmark.

| Method | Baseline | Mitigated | Latency | Verdict |
|---|---|---|---|---|
| **DoLa** | correct: krill / small crustaceans | **worse**: fish and small marine mammals | 58.6 s | over/contrast changed a correct answer into a hallucination |
| **CAD** | rambling, ends with dates | **correct**: Argentina won 3 World Cups | 46.4 s | clear improvement in this example |
| **CAA** | correct: small crustaceans / krill | **wrong**: "large crustaceans" | 37.3 s | over-correction with 6 pairs, layer 14, multiplier 1.0 |
| **CoVe** | wrong: "small marine mammals like krill" | **correct facts**, but answered in English | 84.5 s / 9 calls | facts improved, language consistency regressed |

This is the most useful result so far: **the methods can help or hurt**, and the default
hyperparameters are not universally safe. CAD and CoVe helped on their example; DoLa and
CAA degraded a correct baseline. This is why the toolkit must report baseline vs
mitigated and why the benchmark/validation path is mandatory.

ITI was not run at 1.5B because the demo dataset has only 12 samples; its 0.5B artifact
was already near chance. A real ITI evaluation needs a few hundred labeled examples.

### Held-out batches (small, CPU)

These are deliberately small (2–3 examples, 16 tokens) because CPU inference is the
bottleneck. They are **not statistically meaningful**, but they show where the methods
do and do not help.

| Set | Method | n | baseline | mitigated | win/tie/loss |
|---|---|---:|---:|---:|---|
| `heldout_context_qa.jsonl` (factual context) | CAD | 3 | 1.000 | 1.000 | 0/3/0 |
| `heldout_conflict.jsonl` (context conflicts with prior) | CAD | 3 | 0.333 | 0.333 | 0/3/0 |
| `heldout_qa.jsonl` (general QA) | CoVe | 2 | 0.500 | 0.500 | 0/2/0 |

Interpretation:

- **CAD context set:** the Qwen 1.5B baseline already followed the supplied context, so
  there was no headroom for CAD to improve. This is a ceiling effect, not a CAD failure.
- **CAD conflict set:** the baseline followed the Argentina context but ignored the false
  "water boils at 50°C" and "Earth is flat" contexts; CAD did the same. CAD does not
  automatically override a strong parametric prior.
- **CoVe general QA:** the first run scored 0.500 vs 0.000 because CoVe switched to English.
  After adding explicit "same language" instructions, the first answer stayed Chinese and
  the score tied at 0.500; the second answer was still English/truncated at 16 tokens. This
  is a language-consistency limitation, not a factual failure.
- CoVe verification answers default to a large token budget; the CLI now exposes
  `--max-questions` and `--max-answer-tokens` for CPU-friendly runs.

JSON results are in `runs/heldout_cad_qwen.json`,
`runs/heldout_cad_conflict_qwen.json`, and `runs/heldout_cove_qwen.json`.

## Findings fixed by this smoke

1. `HookManager.active` was not propagated to individual hooks, so CAA/ITI ran with
   `hook_calls == 0`. Fixed with an `active` property that updates every hook.
2. `CoVe` question parsing only accepted ASCII `?`; now accepts fullwidth `？`.
3. ITI activation collection assumed all chat-formatted samples had equal sequence
   length; fixed by selecting the last content token per sample before stacking.

## Still pending

- Real 7B/8B smoke with GPU and 128+ token outputs.
- Real OpenAI/OpenAI-compatible API smoke with a user key.
- Held-out benchmark (TruthfulQA/NQ-Swap/CNN-DM adapters).
- Clean-venv wheel install test and CI.
