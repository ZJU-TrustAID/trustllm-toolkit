# Alignment Toolkit 快速启动

用 YAML 描述 SFT、DPO 或 RLHF 多阶段对齐实验，统一管理数据集、方法和训练后端配置。

**默认训练后端是 Hugging Face TRL。** 配置解析和 `dry-run` 不需要 GPU；执行 `train` 时会通过 TRL 加载 Hugging Face 模型和数据集，并依次运行 SFT、Reward Model、DPO 或 PPO。

## 1. 安装

需要 Python 3.10+，基础依赖只有 PyYAML。仓库提供 `environment.yml`，可用 Conda 自动创建 Python 3.11 环境。以下所有命令均在本 toolkit 根目录执行；项目位于其他路径时，请修改 `cd` 路径。

### 使用 Conda（推荐）

```bash
cd /path/to/alignment_toolkit
bash scripts/setup_env.sh
conda activate alignment-toolkit
```

脚本会创建或更新 Conda 环境，安装 PyTorch CUDA 12.4、Transformers、Datasets、Accelerate、TRL、PEFT 和 bitsandbytes，并运行 `pip check` 与 CUDA 可用性检查。服务器必须预先安装兼容的 NVIDIA 驱动。

如果只想手动创建基础环境：

```bash
conda env create -f environment.yml
conda activate alignment-toolkit
python -m pip install -e '.[train,test]'
python -m pip check
```

需要安装 veRL 或 OpenRLHF 时，可以在同一脚本中显式启用对应后端：

```bash
bash scripts/setup_env.sh --all-backends
```

OpenRLHF 从 GitHub 安装，veRL 和 OpenRLHF 可能与其他训练框架产生版本冲突；生产环境建议为不同后端使用独立 Conda 环境。

### 使用 Python 虚拟环境

```bash
cd /path/to/alignment_toolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,test]'
```

Python 虚拟环境可以安装训练 Python 包，但不会安装 NVIDIA 驱动或 Conda 的 CUDA runtime；GPU 用户建议使用上面的 Conda 脚本。

安装后可使用 `alignment` 命令，也可用 `python -m alignment_toolkit.cli` 调用同一个入口。

## 2. 查看交互式文档

项目内包含 Alignment 专属的只读文档展示：

```bash
conda activate alignment-toolkit
python docs/server.py --host 127.0.0.1 --port 8768
```

然后访问 <http://127.0.0.1:8768>。默认入口是完整指南；页面顶部和首页按钮都可以进入真实训练控制台 `/console`。

需要在已经安装 `.[train]` 的 Conda 环境中启动 `docs/server.py`，`/console` 才能启动真实训练进程。

## 3. 运行第一个示例

```bash
alignment dry-run --config configs/experiments/sft.yaml
```

命令先打印解析后的 JSON，最后输出：

```text
dry-run ok: 1 stage(s)
```

其他内置示例：

```bash
# DPO：只输出解析后的 JSON
alignment resolve --config configs/experiments/dpo.yaml

# RLHF：SFT → Reward Model → PPO
alignment dry-run --config configs/experiments/rlhf.yaml

# 使用 TRL 执行配置中的训练阶段（需要安装 .[train]）
alignment train --config configs/experiments/rlhf.yaml
```

RLHF 示例最后输出 `dry-run ok: 3 stage(s)`。`resolve` 和 `dry-run` 只检查方法、后端和数据集能力；`train` 会下载模型/数据集并将阶段产物写入 `output.root` 下的 pipeline 目录。

## 4. Python 使用样例

在当前目录启动 Python，或将以下代码保存为脚本运行：

```python
from alignment_toolkit import AlignmentExperiment, load_experiment

spec = load_experiment("configs/experiments/rlhf.yaml")
experiment = AlignmentExperiment(spec)

print(experiment.dry_run().name)
for stage in spec.stages:
    print(stage.stage_id, stage.method, stage.backend, stage.dataset)
```

预期输出：

```text
rlhf
sft sft trl ultrachat_sft
reward_model reward_model trl ultrafeedback_dpo
ppo ppo trl ultrachat_prompts
```

## 5. 编写自己的配置

将下面内容保存为项目根目录的 `my_sft.yaml`：

```yaml
version: 1
models:
  base:
    source: huggingface
    id: Qwen/Qwen2.5-7B-Instruct
    revision: main
datasets:
  ultrachat_sft:
    kind: instruction
    adapter: hf_instruction
    source:
      type: huggingface
      id: HuggingFaceH4/ultrachat_200k
      split: train_sft
    mapping:
      messages: messages
pipeline:
  name: my_sft
  stages:
    - id: sft
      method:
        name: sft
        backend: trl
      dataset:
        name: ultrachat_sft
      training:
        learning_rate: 2.0e-5
        epochs: 1
output:
  root: models
```

```bash
alignment dry-run --config my_sft.yaml
```

替换模型 ID、数据集名称和训练参数即可描述自己的实验；`training` 字段会原样传给 runner，具体参数支持与含义由接入的后端负责。
模型通过 Hugging Face 的 `source`、`id` 和 `revision` 指定；`revision` 可以填写分支名、tag 或 commit hash。

需要教师/参考模型的 LoRA 阶段可以配置：

```yaml
training:
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj]
  adapter_disabled: true
```

`adapter_disabled: true` 会让 TRL 不再加载第二个 reference model，而是在同一个 base model 上临时关闭 LoRA adapter 计算参考策略。该选项必须和 `lora` 一起使用；它适用于 DPO 和 PPO 等需要参考策略的阶段。

LoRA 默认在训练结束后 merge 到 base model。Reward Model 会自动把 `score` 头加入 `modules_to_save`，因此输出目录保存的是可直接加载的完整 Reward Model，而不是只有 adapter 的目录；可以设置 `merge_lora: false` 保留 adapter。

| 方法 | 数据集类型 | 默认后端 |
| --- | --- | --- |
| `sft` | `instruction` | `trl` |
| `dpo` | `preference` | `trl` |
| `reward_model` | `preference` / `reward` | `trl` |
| `ppo` | `prompt` | `trl` |

每个实验 YAML 可以直接声明自己的 `datasets` 数据源；旧配置仍可通过 `datasets.registry` 复用 [configs/datasets.yaml](configs/datasets.yaml)。新增数据时需声明其 `kind`、`adapter`、`source` 和字段 `mapping`，并在实际训练集成中实现对应加载逻辑。

## 6. 运行和扩展训练

安装 `.[train]` 后，CLI 会使用内置的 `TRLRunner`：

```bash
alignment train --config configs/experiments/sft.yaml
alignment train --config configs/experiments/dpo.yaml
alignment train --config configs/experiments/rlhf.yaml
```

RLHF 配置按以下顺序运行：SFT → Reward Model → PPO。每个阶段的 checkpoint 默认保存到 `output.root/<pipeline>/<stage>/`，后一阶段通过 `input` 引用前一阶段的产物。

Python API 也可以直接使用 TRL runner：

```python
from alignment_toolkit import AlignmentExperiment, TRLRunner, load_experiment


experiment = AlignmentExperiment(
    load_experiment("configs/experiments/rlhf.yaml"),
    runner=TRLRunner(),
)
print(experiment.run())
```

TRL runner 支持 `SFTTrainer`、`DPOTrainer`、`RewardTrainer` 和 `PPOTrainer`。实际训练仍需要足够的 GPU 显存、正确的模型/数据字段，以及根据硬件调整 `training` 参数。

[configs/evaluations.yaml](configs/evaluations.yaml) 提供安全评测 profile；`alignment evaluate` 当前执行本地 `refusal_heuristic` 适配器。

## 7. 安全评估

安装评估依赖后，可以对训练产物运行安全 prompt 评估：

```bash
python -m pip install -e '.[eval]'
alignment evaluate \
  --config configs/experiments/rlhf.yaml \
  --profile safety_alignment
```

当前内置的是本地 `refusal_heuristic` 适配器：它加载 profile 中的 Hugging Face 或本地 prompt 数据，生成回答，保存逐样本 JSONL，并计算拒答率、攻击成功率和过度拒答率。它为评估流程提供可运行基线，不等同于 HarmBench 或 StrongREJECT 官方 judge；后续可以在同一个 evaluator 接口中接入官方分类器或 judge model。

评估模型由当前实验 YAML 的 `evaluation.model.stage` 自动定位，例如 RLHF 配置中的 `stage: ppo` 会加载 `models/rlhf/ppo`；`--model` 仍可作为临时本地 checkpoint 覆盖。

结果默认写入：

```text
models/rlhf/evaluation/safety_alignment/
├── harmbench.jsonl
├── strongreject.jsonl
├── jailbreakbench.jsonl
├── xstest.jsonl
└── summary.json
```

部分安全数据集可能需要 Hugging Face 登录或访问权限。`--max-samples` 可以先做小规模 smoke test：

```bash
alignment evaluate --config configs/experiments/rlhf.yaml \
  --max-samples 8
```

## 8. 测试与常见问题

```bash
python -m pytest -q
```

- 找不到 `alignment`：确认安装与运行使用同一 Python 环境，或使用 `python -m alignment_toolkit.cli`。
- 找不到数据或评估配置：新实验优先使用 YAML 内嵌的 `datasets` 和 `evaluation.suites`；旧配置中的 `datasets.registry` 和 `evaluation.registry` 会相对于对应 YAML 文件解析。
- `evaluate` 提示 backend 未实现：当前默认是本地 `refusal_heuristic`；配置为 `rewardbench` 等名称只代表预留适配点。
- `alignment train` 报缺少依赖：执行 `python -m pip install -e '.[train]'`，并确认当前 Python 与安装依赖的环境一致。
- PPO 训练失败：确认使用了固定的 TRL 版本、PPO 配置字段与当前 GPU/Transformers 版本兼容。
