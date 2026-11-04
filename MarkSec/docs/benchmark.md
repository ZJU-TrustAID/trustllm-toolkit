# MarkSec

MarkSec is an open-source toolkit for benchmarking attacks against LLM
watermarking systems. It includes a configuration-driven experiment runner,
watermark attack implementations, evaluation utilities, benchmark matrix
definitions, and public dataset slices for quick reproduction checks.

This repository is the clean toolkit release tree. It intentionally excludes
development notes, paper drafts, dashboards, notebooks, experiment schedulers,
generated experiment state, runtime logs, model caches, and local outputs.

## Included Benchmark Scope

The public benchmark matrix is defined in `src/evaluation/benchmark_matrix.py`.

Datasets:

| key | default path | notes |
| --- | --- | --- |
| `c4` | `data/c4/processed_c4.json` | Shipped public slice. The file is JSONL despite the `.json` suffix and contains `prompt` and `natural_text`. |
| `dolly` | `data/databricks-dolly-15k.jsonl` | Shipped instruction-following slice. |
| `bookreport` | built in | Prompt-generation dataset, no file path required. |

The full Hugging Face C4 source is `allenai/c4`:
https://huggingface.co/datasets/allenai/c4. Use config `en` for the full
English C4 release. The shipped public slice is small and news-like; if you
want the closest HF source variant for regenerating a larger C4-style slice,
use config `realnewslike`.

```python
from datasets import load_dataset

# Full English C4. Prefer streaming unless you explicitly want a local copy.
c4_en = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Smaller news-like C4 variant, useful for rebuilding a MarkSec-style slice.
c4_news = load_dataset(
    "allenai/c4", "realnewslike", split="train", streaming=True
)
```

HF C4 records have `text`, `timestamp`, and `url` fields. MarkSec's C4 loader
expects JSONL records with `prompt` and `natural_text`, so convert downloaded HF
records to that schema before overriding the default C4 path.

Direct-removal attacks:

| attacks | watermark families |
| --- | --- |
| `LLMP`, `SIRA`, `DIPPER`, `RW`, `SA` | `KGW`, `MPAC`, `Unigram`, `SynthID`, `Unbiased`, `DIP`, `SIR` |

Steal-informed / lifecycle attacks:

| attacks | watermark settings |
| --- | --- |
| `B4`, `MIP`, `DEMARK`, `SCTS`, `JSV` | `KGW_h1_g0.25`, `KGW_h1_g0.5`, `KGW_h3_g0.25`, `KGW_h3_g0.5` |

Base models:

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## SA Unsupported Settings

`SA` is included in the direct-removal benchmark, but these three primary
settings are intentionally excluded and should not be run:

- `SA + KGW + mistralai/Mistral-7B-Instruct-v0.3`
- `SA + Unigram + mistralai/Mistral-7B-Instruct-v0.3`
- `SA + SIR + mistralai/Mistral-7B-Instruct-v0.3`

This is an algorithm-design limitation, not a missing implementation. For
distortionary watermark families (`KGW`, `Unigram`, and `SIR`), `SA` uses a
reference distribution in the same tokenizer token-id space. The Mistral threat
model currently maps to the cross-family surrogate
`meta-llama/Llama-3.2-1B-Instruct`, so these Mistral distortionary cells do not
have a validated same-tokenizer reference model. The benchmark matrix records
this exclusion in `EXCLUDED_PRIMARY_SETTINGS`.

## Install

```bash
git clone https://github.com/ZJU-TrustAID/MarkSec.git MarkSec
cd MarkSec

conda create -n wm python=3.11 -y
conda activate wm
pip install -r requirements.txt
```

If you already have the `wm` environment, activate it and run only the final
`pip install` command.

After installation, the CLI entrypoint is available as:

```bash
marksec --help
```

## Model Loading

By default, MarkSec uses Hugging Face model IDs and lets Transformers/vLLM load
from the Hub or local HF cache. For offline or region-constrained review, use a
pre-downloaded mirror root:

```bash
export HF_MODELS_ROOT="$PWD/models"
export HF_TOKEN="<token-for-gated-models-if-needed>"
```

The download helper defaults to `https://hf-mirror.com`. Use
`--endpoint https://huggingface.co` for the official Hub endpoint.
Its dry run prints the exact model list discovered from the release configs,
including base models, threat-model surrogates, DIPPER/RW/SCTS/B4 auxiliary
models, and quality models.

```bash
python scripts/download_hf_models.py \
  --models-root "$HF_MODELS_ROOT" \
  --endpoint https://hf-mirror.com \
  --dry-run

python scripts/download_hf_models.py \
  --models-root "$HF_MODELS_ROOT" \
  --endpoint https://hf-mirror.com \
  --mode missing
```

To force local-only loading during a run:

```bash
python src/main.py \
  -g configs/general.yaml \
  -c configs/examples/c4_rw_kgw_qwen_generate.yaml \
  -o runner_settings.models_local_files_only=true
```

When local-only mode is enabled, missing model assets fail fast instead of
silently falling back to the network.

Additional non-HF PSP assets are expected at:

```text
$HF_MODELS_ROOT/paraphrase-at-scale-english/model.para.lc.100.pt
$HF_MODELS_ROOT/paraphrase-at-scale-english/paranmt.model
```

## Watermark Backends

MarkSec does not vendor MarkLLM. KGW and SynthID generation use the
watermarking implementations shipped by Hugging Face Transformers
(`WatermarkingConfig` and `SynthIDTextWatermarkingConfig`). MarkSec wraps those
APIs behind a small runner-compatible interface and provides lightweight
compatibility backends for the remaining benchmark watermark names.

## Open-Source Scope

The release tree keeps only the pieces needed to use MarkSec as a toolkit:

- `src/`: core attack, configuration, evaluation, logging, and data-loading code.
- `configs/`: default runtime settings, attack configs, database examples, and
  KGW watermark settings used by the benchmark matrix.
- `libs/psp/`: PSP metric runtime code and SentencePiece asset.
- `data/`: small public C4 and Dolly slices.
- `scripts/`: model download and stealing-corpus preparation helpers.
- `configs/examples/`: small experiment YAMLs that show the expected runtime
  config shape.

Excluded material includes experiment schedulers, scheduler development
backfills, web dashboards, generated orchestration state, logs, experiment outputs,
notebooks, analysis exports, agent configuration, local memory, and
machine-specific runtime configs.

## Quick Sanity Run

The example config runs `RW + KGW + Qwen` on the shipped C4 slice.
`pipeline_mode=generate` avoids the heavy quality-evaluation phase and is
intended as an environment sanity check.

```bash
python src/main.py \
  -g configs/general.yaml \
  -c configs/examples/c4_rw_kgw_qwen_generate.yaml
```

Outputs are created locally under `output/` and are not part of the release
tree.

## Reproduce The C4 Benchmark Matrix

MarkSec no longer ships an experiment scheduler. Use
`src/evaluation/benchmark_matrix.py` as the source of truth for supported
dataset, attack, watermark, and base-model combinations, then run the generated
or hand-written YAMLs with `src/main.py`.

Direct-removal scope:

```text
datasets:   c4
attacks:    LLMP SIRA DIPPER RW SA
watermarks: KGW MPAC Unigram SynthID Unbiased DIP SIR
models:     meta-llama/Llama-3.1-8B-Instruct
            Qwen/Qwen2.5-7B-Instruct
            mistralai/Mistral-7B-Instruct-v0.3
```

Steal-informed / lifecycle scope:

```text
datasets:   c4
attacks:    B4 MIP DEMARK SCTS JSV
watermarks: KGW_h1_g0.25 KGW_h1_g0.5 KGW_h3_g0.25 KGW_h3_g0.5
models:     meta-llama/Llama-3.1-8B-Instruct
            Qwen/Qwen2.5-7B-Instruct
            mistralai/Mistral-7B-Instruct-v0.3
```

For benchmark-scale sweeps, use your own orchestration layer to create YAMLs and
call `python src/main.py -g configs/general.yaml -c <experiment.yaml>`.

## Stealing Corpus Prep

`JSV` requires a prebuilt paired corpus. `B4` and `MIP` can also reuse the same
prebuilt corpus for consistent lifecycle comparisons. Put corpus files under
`output/corpus/`; the attacks auto-discover files whose
names contain the model, dataset, and exact KGW setting.

Template for one KGW setting:

```bash
python scripts/data_gen/generate_watermark_corpus_vllm.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset_path data/c4 \
  --alg_name KGW \
  --alg_config configs/watermark/KGW/h1_g0.25.json \
  --gamma 0.25 \
  --delta 2.0 \
  --num_samples 30000 \
  --output_file output/corpus/KGW_h1_g0.25_Llama-3.1-8B-Instruct_c4_30k.jsonl \
  --overwrite_output
```

Repeat for the exact KGW settings and base models you plan to reproduce.

## Results

Runtime artifacts are generated locally:

```text
output/marksec.db              sqlite run registry and benchmark summary
output/result/                 raw per-run result artifacts
output/log/                    runtime logs
```

`output/marksec.db` is created on first run. It is the canonical local record
for completed cells after reproduction; stale CSV exports are not used in this
release branch.

## Repository Layout

```text
MarkSec/
├── configs/                  # runtime defaults, attack JSON, KGW setting JSON
├── data/                     # shipped public dataset slices
├── libs/psp/                 # PSP metric runtime code and SentencePiece asset
├── scripts/                  # model download and stealing-corpus prep helpers
├── src/                      # attack implementations and evaluation pipeline
└── output/                   # generated locally, ignored by git
```

Vendored dependency licenses are preserved in `libs/psp/LICENSE`.
