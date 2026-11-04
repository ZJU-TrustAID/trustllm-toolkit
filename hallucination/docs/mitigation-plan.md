# 幻觉缓解工具包（hallucination.mitigation）完整实施计划 v4

> 版本：4.0  
> 日期：2026-10-25  
> 范围：**DoLa、CAA、ITI（可选）、CAD、CoVe**  
> 目标：把 `hallucination.mitigation` 做到 **0.1.0 可发布、可一键调用**的 Python 工具包标准  
> **最终验收：用户配置真实模型 API，或使用 Hugging Face / 本地模型完成实际生成与缓解；`demo --offline` 只做冒烟测试，不计入验收。**  
> 早期研究稿：`docs/archive/IMPLEMENTATION_PLAN_v1_research.md`（含 CAD/RARR 等候选调研；关键结论已合并到本版）

---

## 0. 一页结论

### 0.1 最终交付什么

一个与 `hallucination.detection` 并列、独立安装的 Python 包：

```text
hallucination/
  hallucination.detection/       # 检测：只报告风险
  hallucination.mitigation/      # 缓解：DoLa / CAA / ITI / CAD / CoVe
```

对外只暴露一个门面：

```python
from hallucination.mitigation import Mitigator

m = Mitigator(method="dola", model="Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
report = m.mitigate("请介绍蓝鲸的食性。")   # 真实模型，同时返回 baseline
print(report.to_markdown())
report.to_json("runs/dola.json")
```

以及一条 CLI：

```bash
hallucination-mitigation run --method dola \
  --model Qwen/Qwen2.5-1.5B-Instruct --prompt "请介绍蓝鲸的食性。"
```

### 0.2 五种方法的定位

| 方法 | 类型 | 是否需要拟合 | 需要白盒 | 是否支持 API 模型 | 默认难度 |
|---|---|---|---|---|---|
| **DoLa** | 解码：层间对比 | 否 | 是（hidden states + lm_head） | 否 | 中 |
| **CAA** | 激活：层向量 steering | **是**（一次对比数据拟合） | 是（residual stream） | 否 | 低 |
| **ITI** | 激活：逐头方向（可选） | **是**（探针 + 头选择） | 是（attention heads） | 否 | 高 |
| **CAD** | 解码：上下文对比 | 否 | 是（两次前向 + logits） | 否 | 中 |
| **CoVe** | 后处理：验证链 + 改写 | 否 | 否 | **是**（HF 或 OpenAI） | 低 |

设计原则：

1. **CAA 作为激活方向的默认实现**；ITI 保留为可选的高细粒度模式。
2. **DoLa、CAD、CoVe 零拟合、开箱即用**；CAA/ITI 是“先 `fit` 一次，之后一键”。
3. **CoVe 是唯一不依赖模型内部状态的方法**，负责覆盖闭源模型和纯黑盒场景。
4. 五种方法不混合分数、不自动挑“最好答案”；报告并列展示 baseline 与 mitigated，由用户/评测决定。
5. CAD 需要显式 `context`；没有上下文的 prompt 不属于 CAD 的适用场景。

### 0.3 “一键可用”与最终验收

**`demo --offline` 只做安装冒烟，不作为最终验收。** 最终验收必须跑真实模型：

| 级别 | 场景 | 命令示例 | 是否计入验收 |
|---|---|---|---|
| S0 | 安装冒烟 | `hallucination-mitigation demo --offline` | **否**，只证明装完能 import/CLI |
| R1 | Hugging Face Hub 真实模型 | `run --method dola --model Qwen/Qwen2.5-1.5B-Instruct --prompt ...` | **是，必须** |
| R2 | 本地模型路径（离线） | `run --method cad --model /models/Qwen2.5-1.5B-Instruct --local-files-only ...` | **是，必须** |
| R3 | 拟合后真实运行 | `fit-steering ...` + `run --method caa --steering ...` | **是（CAA 必须；ITI 可选）** |
| R4 | 真实模型 API | `run --method cove --backend openai --model gpt-4o-mini ...` | **是，必须** |
| R5 | 真实数据评测 | `benchmark --dataset truthfulqa_mc2 --methods dola,caa` | **是，必须** |

约束：

- **API 模型只支持 CoVe**。DoLa/CAA/ITI/CAD 需要本地白盒模型（hidden states / attention
  heads / 每步 logits），给 API 模型会直接报“不支持”，不会偷偷退化成 CoVe。
- R1/R2 至少覆盖 DoLa、CAD、CoVe；CAA 需完成一次 `fit-steering` 后运行；ITI 可选。
- R4 支持 OpenAI 官方 API 与 OpenAI-compatible 本地服务（vLLM/Ollama/LM Studio/TGI）。
- 验收报告必须是真实模型输出，包含 baseline、mitigated、provenance、warnings、cost；
  demo 的合成输出不能替代。
- 首次运行模型下载可以联网；本地模型路径验收必须能在断网环境运行。

### 0.4 时间与人力

单人全职约 **8 周**；兼职约 11–13 周。M2（DoLa）、M3（CAD）、M4（CoVe）可部分并行。

---

## 1. 产品定义

### 1.1 目标用户

- 教材读者：想复现书稿中的 DoLa / ITI / CAD 思路（激活默认用更易用的 CAA）；
- 研究者：需要 baseline vs mitigated 的配对实验与可审计轨迹；
- 应用开发者：想在本地模型或 API 模型上快速试一种缓解方法；
- 教学者：需要可读、可检查、可复现的最小实现，而不是论文级 monorepo。

### 1.2 核心用例

1. 从 Hugging Face Hub 指定模型 id，自动下载后跑 DoLa/CAD/CAA/CoVe；
2. 从本地模型目录加载，断网运行；
3. 配置 OpenAI 或 OpenAI-compatible API，跑 CoVe；
4. 本地 Llama/Qwen 模型上跑 DoLa，比较 baseline 与 mitigated 的事实性；
5. 用一批正/负样本拟合 CAA 向量，之后一行代码启用；
6. 用 TruthfulQA 拟合 ITI 探针，在特定模型上复现逐头干预；
7. 给定上下文，跑 CAD，让模型更忠实地使用上下文；
8. 对任意 HF/API 模型跑 CoVe，得到验证问题、独立回答和修订稿；
9. 在同一模型上连续比较多种方法，输出统一报告；
10. 在自定义数据上做配对评测，报告准确率、长度、重复率、延迟。

### 1.3 非目标

- 不做训练/微调（ITI/CAA 只拟合小工件，不更新语言模型参数）；
- 不做 RARR、Self-Refine、TruthX、FAVA 的 v0.1 实现（进入 backlog）；
- 不做多卡张量并行、batch>1、流式输出（v0.2 之后再评估）；
- 不服务闭源 API 模型的 DoLa/ITI/CAA/CAD（拿不到 hidden states/attention heads/logits）；
- 不打包模型权重、TruthfulQA/NQ-Swap/CNN-DM 数据或第三方向量；
- 不承诺“消除幻觉”；报告必须保留“可能过校正/仍可能错误”的警告。

### 1.4 与检测包的关系

- 两个包独立安装、独立版本；`hallucination.mitigation` 不依赖
  `hallucination.detection` 的核心代码。
- 提供可选 extra `[detect]`：安装后可在报告中附加检测包的 risk 变化；
  检测分数不是事实真值，报告中必须注明。
- 共享 tiny-model 测试夹具、文档风格和 CLI 约定，但不共享会阻碍独立安装的实现。

---

## 2. 方法矩阵与选择理由

### 2.1 为什么是这五个

| 路线 | 方法 | 覆盖能力 | 为什么选它 |
|---|---|---|---|
| 解码干预 | DoLa | 不需要拟合，直接改 logits | 经典、可解释、与书稿一致；单次前向即可，成本低 |
| 解码干预 | CAD | 上下文对比 | 书稿原方法；补足“有上下文但模型不听”的忠实性场景；零拟合 |
| 激活干预（默认） | CAA | 层向量 steering | 比 ITI 简单、稳定、易移植；MIT；自带 hallucination 行为数据 |
| 激活干预（可选） | ITI | 逐头、细粒度 | 书稿原方法；保留给需要 head-wise 控制的用户 |
| 后处理修正 | CoVe | 黑盒、模型无关 | 补足白盒方法无法覆盖的 API/闭源场景；实现成本低 |

五种方法覆盖三条路线：DoLa/CAD 改“怎么选词”，CAA/ITI 改“内部表征”，CoVe 改“生成后的答案”。

### 2.2 CAD 为什么重新纳入

CAD 需要同一模型两次前向、两份 KV cache，成本约 2×；但它是书稿三方法之一，且与 DoLa
共享“logits 对比”基础设施，增量工程量可控。v0.1 采用**单模型、单进程、logits 组合**的
实现，不复制官方多进程 group decode 框架，因此仍然满足“一键调用”。

### 2.3 模型支持矩阵

| 模型族 | DoLa | CAA | ITI | CAD | CoVe |
|---|---|---|---|---|---|
| Llama-2 / Llama-3 / TinyLlama | ✅ | ✅ | ✅（需拟合） | ✅ | ✅ |
| Qwen2 / Qwen2.5 | ✅ | ✅ | ⚠️ 实验性（结构适配） | ✅ | ✅ |
| Mistral | ✅ | ✅ | ⚠️ 实验性 | ✅ | ✅ |
| GPT-2 类 | ⚠️ 需 adapter | ⚠️ 需 adapter | ❌ | ⚠️ 需 adapter | ✅（API/HF） |
| OpenAI-compatible API | ❌ | ❌ | ❌ | ❌ | ✅ |

v0.1 承诺：**Llama 系 + TinyLlama + Qwen2.5 的 DoLa/CAA/CAD/CoVe**；ITI 只承诺 Llama 系。
其他架构标“实验性”，不进入默认 preset。

---

## 3. 官方实现审计与复用边界

### 3.1 审计结果

| 方法 | 官方/参考仓库 | Commit | 许可证 | 形态 |
|---|---|---|---|---|
| DoLa | https://github.com/voidism/DoLa | `805230e` | README 标 MIT，根目录无 LICENSE | `dola.py` + fork 的 transformers 4.28.1 |
| CAA | https://github.com/nrimsky/CAA | `5dabbbd` | **MIT** | Llama-2 wrapper + 向量生成/steering 脚本；含 hallucination 行为数据 |
| ITI | https://github.com/likenneth/honest_llama | `2c6b217` | **MIT** | 激活采集、探针训练、pyvene 干预、baked-in 权重 |
| CAD | https://github.com/xhan77/context-aware-decoding | `1281d7f` | **无 LICENSE** | `accelerate` 多进程 group decode + JSONL 输入 + 评测脚本 |
| CoVe | **无 Meta 官方仓库**；社区实现 https://github.com/ritun16/chain-of-verification | `23dde5d` | 无 LICENSE | LangChain + OpenAI + DuckDuckGo 的四阶段实现 |

### 3.2 关键事实

**DoLa**

- 官方 `DoLa.generate()` 依赖 fork 的 `transformers-4.28.1` 中的 `dola_decoding` /
  `early_exit_layers`；不能直接用于现代模型。
- 官方 README 声明主要支持 LLaMA-v1。
- 官方生成路径与 `lm_score` 的归一化顺序不完全一致（raw mature − logprob premature
  vs 两者都 log_softmax）；本包以论文公式为准，另设 `official_compat` 做数值对照。
- **复用边界**：只实现算法；不引入 fork。

**CAA**

- `generate_vectors.py` 对每个 layer 收集正/负样本最后 token 的 residual stream，
  计算 `mean(pos - neg)` 得到层向量；`prompting_with_steering.py` 在 residual stream
  加 `multiplier * vector`。
- 仓库自带 `hallucination` 行为数据（1000 generate / 50 test），适合做首个 demo。
- **复用边界**：MIT，可参考/改编并保留版权声明；实现上仍独立，避免绑定 Llama-2 wrapper。

**ITI**

- 每个 (layer, head) 训练 LogisticRegression；按验证准确率选 top-K 头；方向为
  `normalize(coef)`，尺度为投影标准差 `σ`；干预加在 `self_attn.o_proj.input`。
- 默认 `K=48, α=15` 是 Llama-2/3 7B/8B 的经验值，不是通用最优。
- **复用边界**：MIT；保留为可选 extra，避免默认安装 sklearn/pyvene。

**CAD**

- 官方实现不是单模型 logits processor，而是 `accelerate` 多进程 group decode：每个
  JSONL 实例给出 `context_string`、`assigned_model`、`assigned_weight`，各进程采样
  one-hot token，`all_reduce` 加权后再 softmax/argmax。
- 典型权重：CNN-DM 用 `1.5 / -0.5`（α=0.5），NQ-Swap 用 `2 / -1`（α=1）。
- 依赖较旧：`transformers==4.26.0`、`torch==1.13.1`、`accelerate==0.16.0`；无许可证。
- **复用边界**：不复制代码；按论文公式实现单模型“两次前向 + logits 组合”。
  官方 one-hot 组合与论文 logits 组合不完全等价，写入 fidelity 文档。

**CoVe**

- 论文方法：baseline → 规划验证问题 → 独立回答 → 汇总修订。
- 没有 Meta 官方代码；社区实现无许可证，且依赖 LangChain/OpenAI/DuckDuckGo。
- **复用边界**：不复制社区代码；按论文重写 prompt 与流程；v0.1 不做检索，
  报告中标记 `evidence="self"`。

### 3.3 许可证策略

1. 本项目使用 **MIT**。
2. DoLa/CAD/CoVe 社区代码无许可证 → **不复制**，只按论文/公式重实现。
3. CAA/ITI 是 MIT → 如参考代码，保留版权声明并写入 `NOTICE`；仍优先独立实现。
4. 模型与数据许可证独立：Llama 需 Meta 许可，Qwen/TinyLlama 为 Apache-2.0；
   TruthfulQA 等数据集按需下载，不打包。
5. CI 增加许可证检查：禁止把无许可证仓库的代码/文本复制进 `src/`。

---

## 4. 包结构与依赖

### 4.1 目录

```text
hallucination.mitigation/
  pyproject.toml
  README.md
  CHANGELOG.md
  LICENSE
  NOTICE
  CITATION.cff
  IMPLEMENTATION_PLAN.md
  docs/
    index.md
    quickstart.md
    methods/
      dola.md
      caa.md
      iti.md
      cad.md
      cove.md
    fidelity.md
    presets.md
    cli.md
    backends.md
    benchmarks.md
    troubleshooting.md
    api.md
    archive/
      IMPLEMENTATION_PLAN_v1_research.md
  examples/
    00_quickstart.ipynb
    01_dola.ipynb
    02_caa_fit_and_apply.ipynb
    03_cad.ipynb
    04_cove.ipynb
    05_compare_methods.ipynb
    06_iti_optional.ipynb
    07_custom_benchmark.ipynb
    data/
      hallucination_pairs_demo.jsonl
      context_demo.jsonl
  configs/
    demo_tiny.yaml
    dola_tinyllama.yaml
    caa_tinyllama.yaml
    cad_tinyllama.yaml
    cove_tinyllama.yaml
    dola_llama2_7b.yaml
    caa_llama2_7b.yaml
    cad_llama2_7b.yaml
    iti_llama2_7b.yaml
    cove_openai_gpt4o_mini.yaml
    cove_vllm_qwen.yaml
  src/hallucination.mitigation/
    __init__.py
    __main__.py
    _version.py
    py.typed
    api/
      mitigate.py
      report.py
      presets.py
    core/
      contracts.py
      base.py
      config.py
      model_config.py
      errors.py
      logging.py
      prompt.py
      sampling.py
    engine/
      hf.py
      decode.py
      streams.py
      hooks.py
      logits.py
      tracing.py
      quantization.py
    backends/
      base.py
      hf.py
      openai.py
      callable.py
    methods/
      dola.py
      caa.py
      iti.py
      cad.py
      cove.py
    steering/
      extract.py
      apply.py
      artifacts.py
      tune.py
    posthoc/
      prompts.py
      cove.py
    evaluation/
      datasets.py
      metrics.py
      benchmark.py
      report.py
    presets/
      *.yaml
      artifacts/
        demo_caa.npz
        demo_iti.npz
    cli.py
    web/
      __init__.py
      server.py
  tests/
    conftest.py
    fixtures/
      tiny_model.py
      tiny_tokenizer.py
      scripted_backend.py
    unit/
    contract/
    integration/
    packaging/
```

### 4.2 Python 与依赖

- Python **>=3.10**；CI 覆盖 3.10 / 3.11 / 3.12。
- 构建后端：`hatchling`（与检测包一致）。
- 基础依赖保持轻量：`numpy`、`pyyaml`、`tqdm`。
- 所有 torch/transformers/OpenAI/sklearn 依赖都放 extras。

```toml
[project]
name = "hallucination-mitigation"
version = "0.1.0"
description = "One-click inference-time hallucination mitigation: DoLa, CAD, CAA, ITI and CoVe."
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = ["numpy>=1.24", "pyyaml>=6.0", "tqdm>=4.66"]

[project.optional-dependencies]
hf = ["torch>=2.1", "transformers>=4.45", "accelerate>=0.30", "safetensors>=0.4", "huggingface-hub>=0.24"]
pca = ["scikit-learn>=1.3"]      # 仅 CAA 的可选多维 PCA；CAA 核心只需 numpy
iti = ["scikit-learn>=1.3"]
openai = ["openai>=1.0"]
datasets = ["datasets>=2.18"]
quant = ["bitsandbytes>=0.43"]
detect = ["hallucination-detection>=0.3"]
plots = ["matplotlib>=3.8"]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.5", "mypy>=1.10", "build>=1.2", "twine>=5.0", "pre-commit>=3.7"]
docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.25"]
all = [
  "hallucination-mitigation[hf,pca,openai,datasets,plots,dev,docs]",
  "bitsandbytes>=0.43",
]
```

### 4.3 包数据

- `presets/*.yaml` 与 `presets/artifacts/*.npz` 随 wheel 发布；
- demo artifact 只用于 tiny 模型，体积 < 100 KB；
- 不打包真实模型/数据集/大向量；
- 所有 artifact 带格式版本、模型 fingerprint、校验和。

---

## 5. 公共 API

### 5.1 `Mitigator`

```python
from hallucination.mitigation import Mitigator

m = Mitigator(
    preset="caa_tinyllama",      # 或 method="caa" + model=...
    device="cuda",
    dtype="bfloat16",
    seed=0,
)

report = m.mitigate(
    "请介绍蓝鲸的食性。",
    max_new_tokens=128,
    baseline=True,               # 默认同时生成 baseline
)
```

真实使用的三种入口：

```python
# 1) Hugging Face Hub（首次下载）
Mitigator(method="dola", model="Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

# 2) 本地模型目录（断网）
Mitigator(method="cad", model="/models/Qwen2.5-1.5B-Instruct",
          local_files_only=True, device="cuda")

# 3) OpenAI / OpenAI-compatible API（仅 CoVe）
Mitigator(method="cove", backend="openai-compatible",
          model="Qwen/Qwen2.5-7B-Instruct",
          api_base="http://localhost:8000/v1",
          api_key_env="HM_API_KEY")
```

构造参数：

| 参数 | 说明 |
|---|---|
| `preset` | 内置 YAML preset；与 `method` 二选一 |
| `method` | `dola` / `caa` / `iti` / `cad` / `cove` |
| `model` | HF id、本地路径，或 API 模型名 |
| `backend` | `hf` / `openai` / `openai-compatible` / callable |
| `api_base` | API base URL；OpenAI 兼容服务（vLLM/Ollama/LM Studio/TGI） |
| `api_key_env` | 存放 API key 的环境变量名，默认 `OPENAI_API_KEY` |
| `timeout` / `max_retries` | API 超时与重试；指数退避，不无限重试 |
| `device` / `dtype` | 本地模型的设备与精度 |
| `revision` | 固定模型版本，保证可复现 |
| `local_files_only` | 只读本地缓存/路径，断网运行 |
| `steering` | CAA 向量路径/对象 |
| `probe` | ITI artifact 路径/对象 |
| `layer` / `multiplier` | CAA 层与强度（可覆盖 preset） |
| `alpha` | ITI/CAD 强度（CAD 默认 0.5） |
| `context_template` | CAD 的 `plain` / `chat` 模板 |
| `seed` | 采样种子 |
| `load_in_4bit` | 实验性 |
| `trust_remote_code` | 默认 False；显式 opt-in |

**后端与方法约束**：`backend=openai/openai-compatible` 只允许 `method="cove"`；
DoLa/CAA/ITI/CAD 在 API 后端下直接抛 `UnsupportedModelError`，并提示改用本地白盒模型。

方法：

- `mitigate(prompt, context=None, **kwargs) -> MitigationReport`（CAD 必须给 `context`）
- `compare(prompt, context=None, methods=["dola", "caa", "cad", "cove"], **kwargs) -> ComparisonReport`
- `available_methods() -> list[str]`（按已安装 extras 过滤）
- `close()` / context manager，释放模型与 hook。

### 5.2 `MitigationReport`

```python
@dataclass
class MitigationReport:
    method: str
    prompt: str
    baseline: str | None
    response: str
    trace: dict[str, Any]
    warnings: list[str]
    cost: dict[str, Any]
    provenance: dict[str, Any]

    def to_markdown(self) -> str: ...
    def to_json(self, path: str | None = None) -> str: ...
    def to_dataframe(self): ...          # 需要 pandas extra
    def save(self, path: str) -> None: ...
```

`to_json` 必须是稳定 schema（带 `schema_version`），可直接给下游脚本消费。
`ComparisonReport` 包含每个方法的报告与一张对比表。

### 5.3 异常

| 异常 | 场景 | CLI 行为 |
|---|---|---|
| `MissingDependencyError` | 缺 `[hf]`/`[iti]`/`[openai]` | 打印安装命令，exit 3 |
| `UnsupportedModelError` | 架构缺少 `lm_head`/`o_proj` | 打印支持矩阵，exit 3 |
| `ArtifactNotFoundError` | CAA/ITI 缺 artifact | 打印 `fit-steering`/`fit-iti` 命令，exit 4 |
| `ArtifactMismatchError` | 模型 fingerprint 不匹配 | 打印期望/实际字段，exit 4 |
| `BackendError` | API 失败、密钥缺失 | 打印可操作信息，exit 5 |
| `GenerationError` | 空输出/截断/NaN | 保留原始输出并警告，exit 5 |
| `ConfigError` | preset 参数非法 | 打印字段，exit 2 |

### 5.4 错误 UX 原则

- 不静默回退到 baseline 或其他方法；
- 不自动下载 gated 模型；先提示许可证/登录；
- 不在库代码里 `print`；用 `logging`，CLI 负责渲染；
- 所有错误信息包含“下一步命令”。

---

## 6. 统一运行时

### 6.1 为什么自写逐 token 引擎

五种方法中：

- DoLa 需要每步所有层的 hidden state；
- CAD 需要两条序列、两份 KV cache、每步两次前向；
- CAA/ITI 需要在特定层/头做 hook；
- CoVe 不需要引擎，但要统一的 LLM 后端；
- `model.generate` 的 `LogitsProcessor` 拿不到层间状态，hook 又难控制“只干预生成 token”。

因此写一个**小型、可检查的逐 token 引擎**，五种方法共享加载、采样、轨迹和报告。
不追求覆盖所有生成特性；v0.1 只保证方法语义正确。

### 6.2 `HFEngine`

```python
engine = HFEngine.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    dtype="bfloat16",
    device="cuda",
    revision="...",
)

state = engine.prefill(input_ids)
out = engine.step(state, token_id, output_hidden_states=True)
# out.logits, out.hidden_states, out.past_key_values
```

要求：

- 单序列、KV cache、`use_cache=True`；
- 可选 `output_hidden_states`；
- 采样：greedy / temperature / top-p / top-k / repetition penalty；
- stop strings + EOS + max tokens；记录停止原因；
- `torch.inference_mode()`；
- 每步 trace：耗时、forward 次数、当前 token、峰值显存；
- 生成结束后清理 hook，保证同一进程可连续跑多方法；
- **多流支持**：`MultiStreamState` 维护 CAD 的 context/no-context 两份 cache，
  每步分别 `step`，再合并 logits；共享采样结果和停止条件。

### 6.3 Hook 管理

- `HookManager` 上下文管理器：注册/移除 forward hook，异常时也清理；
- 每个 hook 记录 `layer`、`module`、`position policy`、`applied_steps`；
- 支持 `position="generated"`（默认）与 `"all"`；
- hook 内不改 dtype/device，只做 in-place 加法或返回新 tensor；
- 对 tuple 输出、GQA、量化模型做兼容测试。

### 6.4 采样与随机性

- `seed` 固定 `torch`/`numpy`/`random`；
- greedy 时忽略 temperature/top-p；
- baseline 与 mitigated 使用同一 seed，保证配对；
- CAD 两条流共享同一个采样 token，保证序列同步；
- CoVe 的每个阶段独立 seed 或统一 seed（可配置），报告中记录。

### 6.5 `LLMBackend`（CoVe 用）

```python
class LLMBackend(Protocol):
    def generate(self, messages: list[dict], **kwargs) -> str: ...
    def generate_many(self, batch: list[list[dict]], **kwargs) -> list[str]: ...
```

实现：

- `HFBackend`：本地模型或 HF Hub，套 chat template；支持 `local_files_only`；
- `OpenAIBackend`：Chat Completions；支持 `api_base`、`api_key_env`、`timeout`、
  `max_retries`；兼容 OpenAI 官方、vLLM、Ollama、LM Studio、TGI 等 `/v1` 服务；
- `CallableBackend`：测试/自定义函数。

CoVe 不依赖 `HFEngine`；它只需要 `generate`。这样 API 模型和本地模型共用一套流程。

API 错误映射：

- 401/403 → `BackendError("API key 无效或无权限；检查 <env>")`；
- 404 → `BackendError("模型或 base_url 不存在；检查 --model/--api-base")`；
- 429/5xx → 指数退避重试，超过 `max_retries` 后报错；
- 超时 → 报错并提示增大 `--timeout`；
- 不把 API key、完整请求头写入日志。

### 6.6 Prompt / Chat 模板

- `PromptBuilder` 统一 plain / chat 两种模式；
- 模板版本化（`COVE_PROMPTS_VERSION`），报告记录版本；
- 不复制无许可证的社区 prompt；按论文重写；
- 对 base 模型与 chat 模型分别测试，避免重复 BOS/系统提示；
- CAD 使用 `context_template="plain"|"chat"` 构造两条 view：有 context 与无 context；
  两条 view 只差上下文块，chat 模式把 context 放进 system/user 前缀，并做 token 快照测试。

### 6.7 Artifact

统一 artifact 抽象：

```python
@dataclass
class Artifact:
    format: str                  # "hm/steering/v1" | "hm/iti/v1"
    method: str
    model: dict                  # id/revision/layers/heads/hidden_size
    payload: dict
    checksum: str
    created_at: str
    provenance: dict
```

保存格式：

- CAA：`.npz`（每层向量）+ 同名 `.json` 元数据；
- ITI：`.npz`（方向/尺度/头索引）+ `.json`；
- 读取时校验 shape、NaN、模型 fingerprint、格式版本；
- `np.load(allow_pickle=False)`；不使用 `torch.load` 加载不可信文件。

### 6.8 Provenance

每个报告记录：

- 包版本、git commit（如可获取）；
- `provider`（`hf` / `openai` / `openai-compatible` / `callable`）；
- 模型 id/revision、tokenizer revision、dtype、device；
- `api_base`（不含 key）、`api_key_env` 名称（不含值）、`local_files_only`；
- torch/transformers/openai 版本；
- seed、generation 参数、方法参数；
- preset 名称与文件 hash；
- 数据集名称/hash（benchmark）；
- 时间戳、命令（CLI）。

---

## 7. 方法实现

### 7.1 DoLa

**算法**

```text
q_j = lm_head(hidden_states[j])          # 候选 premature 层
q_N = lm_head(hidden_states[N])          # mature 层
M = argmax_j JSD(q_N || q_j)
mask = q_N < max(q_N) + log(relative_top)
p = softmax(log_softmax(q_N) - log_softmax(q_M))
```

**层索引**

HF `hidden_states[0]` 是 embedding，`hidden_states[k]` 是第 `k-1` 层 decoder 输出。
官方 Llama-7B 的 `early_exit_layers=32` 对应 `hidden_states[32]`，即最后一层 decoder
输出（未过 final norm）。本包保持同一映射：

- `mature_layer = num_hidden_layers`（默认）；
- `candidate_premature_layers < mature_layer`；
- early-exit hidden state 不额外做 final norm；
- 用 `model.get_output_embeddings()` 取 `lm_head`。

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `mature_layer` | `num_hidden_layers` | 最后一层 decoder 输出 |
| `candidate_premature_layers` | preset | TinyLlama 10–20 偶数；Llama-2/3 7B/8B 16–30 偶数 |
| `relative_top` | 0.1 | APC/relative-top；0 关闭 |
| `post_softmax` | True | 与官方 `lm_score` 对齐 |
| `official_compat` | False | 复现官方生成路径，仅对照 |
| `max_new_tokens` | 128 | 生成预算 |

**输出 trace**

每步记录：选中的 premature layer、JSD、mature entropy、top-1 token/prob、是否被
APC 过滤、耗时。

**失败模式**

- 模型没有 `lm_head` 或 hidden states：报 `UnsupportedModelError`；
- 层索引越界：加载时校验；
- 4-bit 模型：hidden states 可用但速度/数值需验证，标实验性。

**测试**

- 固定 logits 的公式测试（JSD、APC、log_softmax 差值）；
- tiny 模型端到端，事件数等于生成 token 数；
- `official_compat` 合成张量对照；
- 层映射测试（`hidden_states[k]` 与 `lm_head`）。

### 7.2 CAA

**数据格式**

```jsonl
{"prompt": "请介绍蓝鲸的食性。", "positive": "蓝鲸主要以磷虾为食。", "negative": "蓝鲸主要以海藻为食。"}
```

也支持 `question` + `answer_matching_behavior` + `answer_not_matching_behavior`。
每对正/负样本应只差关键事实，避免长度/风格混淆。

> **CAA 不训练分类器/判别头**：方向直接来自 `mean(h_pos - h_neg)`，没有 LogisticRegression、
> 没有梯度更新；`fit-steering` 只是一次统计平均。sklearn 只在可选的 PCA 多维方向时使用。

**拟合**

1. 对每条 `prompt + positive` 和 `prompt + negative` 做一次前向；
2. 取指定层输出（residual stream）最后一个非 padding token 的 hidden state；
3. `v_l = mean(h_pos - h_neg)`；
4. 归一化 `v_l / ||v_l||`，记录原始 norm；
5. 可选：对差值矩阵做 PCA，取 top-k 正交方向（v0.2）；
6. 保存 `SteeringArtifact`（layer, vector, norm, dataset hash, model fingerprint）。

**应用**

```text
h_l <- h_l + multiplier * v_l / ||v_l||
```

- 默认只作用于生成 token（`position="generated"`）；
- 通过 decoder layer 的 forward hook 修改 `output[0]`；
- `multiplier` 默认 1.0，必须用 validation 扫描；
- 不做“bake 到权重”：residual stream 的加法不是简单的 bias，运行时 hook 是正确实现。

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `layer` | preset | 7B 常用 10–16；TinyLlama 用 8–12；必须验证 |
| `multiplier` | 1.0 | 0 等于 baseline；过大导致退化 |
| `position` | `"generated"` | `"all"` 复现部分官方行为 |
| `pooling` | `"last"` | 拟合激活池化；`"mean"` 可选 |
| `normalize` | True | 单位向量，便于跨层比较 |

**层/乘数选择**

- `tune-steering` 在 validation split 上扫 `layer × multiplier`；
- 选择标准：truthfulness 提升、长度/重复率不崩、KL 不过大；
- test split 只做最终报告；
- 报告中必须写明“layer/multiplier 来自验证集”。

**失败模式**

- 正负样本太少（< 20 对）→ 警告，不建议用于生产；
- 向量范数异常/NaN → 报错；
- 层不存在/模型结构不匹配 → `ArtifactMismatchError`；
- 过校正：答案变短、重复、拒绝回答 → 质量护栏告警。

**测试**

- 合成正负激活，验证 `mean(pos-neg)` 方向与符号；
- 归一化后乘数可复现；
- hook 只作用于生成 token；
- `multiplier=0` 输出与 baseline 完全一致；
- artifact roundtrip / fingerprint mismatch。

### 7.3 CAD

**输入**

```python
Mitigator(method="cad", model="...").mitigate(
    "阿根廷赢过多少次世界杯？",
    context="阿根廷在 1978、1986 和 2022 年赢得了世界杯。",
)
```

`context` 必填；缺失时抛 `ConfigError`，不自动退化成 DoLa/普通解码。CAD 的 baseline 是
“带 context 的普通解码”，不是无 context 的 prompt。

**两条 view**

```text
view_c = PromptBuilder(context, question, with_context=True)
view_u = PromptBuilder(context, question, with_context=False)
```

- `plain`：`context + "\n\n" + question` / `question`；
- `chat`：context 放进 system message 或 user 前缀，再套
  `tokenizer.apply_chat_template(..., add_generation_prompt=True)`；
- 两条 view 只差 context 块，其余 system/user 内容完全一致；
- 对 plain/chat 分别做 token 快照测试。

**逐 token 流程**

1. 分别 `prefill(view_c)`、`prefill(view_u)`，得到两份 KV cache；
2. 每步分别 `step`，取 `q_c`、`q_u`；
3. 可选对每个 view 做 top-p 过滤（官方 `filter_p`）；
4. `logits = (1 + α) * log_softmax(q_c) - α * log_softmax(q_u)`；
5. 采样一个 token，**同时追加到两条序列和两个 cache**；
6. 以 context view 的 EOS 为准停止；记录两条 view 的 KL/一致率。

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `alpha` | 0.5 | CNN-DM 官方 `1.5/-0.5`；NQ-Swap 用 1.0 |
| `filter_top_p` | 1.0 | 关闭；官方可选 |
| `context_template` | `"plain"` | `plain` / `chat` |
| `combine` | `"logits"` | v0.1 只实现论文的 logits 组合 |
| `max_new_tokens` | 128 | 生成预算 |

**与官方差异**

- 官方 group decode 是“每个 view 先采样 one-hot，再按权重求和”，支持不同模型/不同
  context 的多进程组合；本包是“单模型、两次前向、logits 加权”，符合论文公式，也是
  一键调用最自然的形式。
- 官方 `transformers==4.26`；本包用现代 `transformers`，chat template 行为可能不同，
  用快照测试固定。
- 官方支持多模型/多 context；v0.1 只支持单模型双 view。

**输出 trace**

每步记录：context view 的 top-1/top-p、no-context view 的 top-1、两分布 KL、
`alpha`、合并后 top-1、cache 长度、是否 EOS。

**失败模式**

- 缺 context → `ConfigError`；
- context 过长 → 按模型 max length 截断并警告，或报错（`context_overflow="error"|"truncate"`）；
- chat template 不一致 → 启动时校验 token 前缀，失败则报错；
- 两份 cache 显存不足 → 提示减小 context/换小模型/4-bit；
- 一侧 EOS 但另一侧未 EOS → 以 context view 为准继续，直到 context EOS 或 max tokens。

**成本**

- 每步 2 次前向，约 2× 延迟；
- 7B + 2k context 两份 KV cache 约多 1–2 GB（fp16）；
- 报告必须记录 `forward_passes=2×`。

**测试**

- `alpha=0` 时输出与“带 context 的 baseline”完全一致；
- 固定 logits 的组合公式数值测试；
- plain/chat 模板 token 快照；
- EOS 同步、cache 长度一致；
- 缺 context 报错；
- TinyLlama 端到端。

### 7.4 ITI（可选）

**依赖与定位**

- 只在 `[iti]` extra 下可用；默认安装不引入 sklearn。
- 面向 Llama 系；Qwen/Mistral 标实验性。
- **ITI 是五种方法里唯一需要训练“判别器”的**：每个 (layer, head) 一个线性探针
  （LogisticRegression），用验证准确率选 top-K 头；探针不是新的模型头，也不反传更新
  语言模型参数。

**拟合**

1. 数据：JSONL `prompt` / `response` / `label`，或 TruthfulQA-MC2 适配器；
2. 按**问题**划分 train/val，避免同一问题的不同选项泄漏；
3. 对每条样本前向，在最后一个 token 处 hook 每层 `self_attn.o_proj.input`；
4. reshape 为 `[num_heads, head_dim]`；
5. 每个 (layer, head) 训练 LogisticRegression，按 val accuracy 选 top-K；
6. 方向 = normalize(coef)；尺度 = 投影标准差；
7. 保存 `ITIArtifact`：每层每头的方向、尺度、准确率、模型 fingerprint。

**应用**

```text
o_proj.input[..., current_pos, :] += alpha * sigma * theta
```

- 默认只干预生成 token；
- `position="all"` 复现官方 pyvene 行为；
- `bake=True` 可把偏移写进 `o_proj.bias`，得到零开销模型（显式 opt-in）；
- 生成结束移除 hook。

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `top_k` | 48 | 小模型自动截断 |
| `alpha` | 15 | 7B/8B 经验值；需验证 |
| `position` | `"generated"` | 官方为 all；默认生成 token 更保守 |
| `use_center_of_mass` | False | 官方可选方向 |
| `bake` | False | 写入 bias，修改内存模型 |

**失败模式**

- 缺 probe → 报错并给 `fit-iti` 命令；
- 头数/维度不匹配 → `ArtifactMismatchError`；
- top head 准确率接近随机 → 警告“该模型/数据上 ITI 可能无效”；
- 量化模型 hook 可能失败 → 标实验性。

**测试**

- hook 数学：增量等于 `alpha*sigma*theta`；
- prompt 位置不被修改；
- 合成线性可分数据，top head 被选中；
- artifact roundtrip / fingerprint mismatch；
- `bake` 与 hook 版本输出一致（容差内）。

### 7.5 CoVe

**流程**

```text
r0 = LLM(q)                                   # baseline
Q = LLM_plan(q, r0)                           # 验证问题列表
A_i = LLM_verify(q_i)                         # 独立回答，不看 r0
r = LLM_revise(q, r0, {(q_i, A_i)})            # 修订稿
```

**关键设计**

- `mode="factored"`（默认）：每个验证问题独立回答，避免复制 baseline 的错误；
- `mode="joint"`：一次回答所有问题，成本低但独立性弱；
- `max_questions` 默认 6；解析失败时重试一次，再失败则只用 baseline；
- 不引入检索；`evidence="self"`；报告中明确“这是模型自检，不是事实核验”；
- 保留每阶段原文、耗时、token 用量；修订稿与 baseline 做 diff。

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_questions` | 6 | 控制成本 |
| `mode` | `"factored"` | 独立/联合回答 |
| `temperature` | 0.0 | 验证阶段尽量确定性 |
| `max_answer_tokens` | 128 | 单题回答预算 |
| `include_search` | False | v0.1 不支持检索 |
| `backend` | `hf`/`openai` | 统一 LLMBackend |

**输出 trace**

`baseline`、`questions`、`answers`、`revised`、`parse_warnings`、`stage_costs`。

**失败模式**

- 问题解析失败 / 问题数为 0 → 返回 baseline，警告；
- 验证答案为空 → 跳过该问题；
- 修订稿为空/过短 → 回退 baseline，警告；
- API 限流/超时 → `BackendError`，不重试无限次。

**测试**

- `CallableBackend` 脚本化输出，验证四阶段顺序与 prompt 内容；
- 问题解析（numbered list / JSON / 异常文本）；
- 空输出回退；
- 本地 HF 与 OpenAI 各一个 smoke（集成测试）；
- 成本/token 统计。

---

## 8. CLI 与一键体验

### 8.1 命令总览

```bash
hallucination-mitigation --version
hallucination-mitigation doctor
hallucination-mitigation list-methods
hallucination-mitigation list-presets

# S0：安装冒烟（不计入最终验收）
hallucination-mitigation demo --offline

# R1：Hugging Face Hub 真实模型（首次下载）
hallucination-mitigation run --method dola \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype bfloat16 \
  --prompt "请介绍蓝鲸的食性。"

# R2：本地模型目录（断网）
hallucination-mitigation run --method cad \
  --model /models/Qwen2.5-1.5B-Instruct --local-files-only \
  --context "阿根廷在 1978、1986 和 2022 年赢得了世界杯。" \
  --prompt "阿根廷赢过多少次世界杯？"

# R3：拟合 + 真实运行
hallucination-mitigation fit-steering --method caa \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data examples/data/hallucination_pairs_demo.jsonl --out runs/caa.npz
hallucination-mitigation run --method caa \
  --model Qwen/Qwen2.5-1.5B-Instruct --steering runs/caa.npz \
  --prompt "请介绍蓝鲸的食性。"

hallucination-mitigation fit-iti --model meta-llama/Llama-2-7b-chat-hf \
  --dataset truthfulqa_mc2 --out runs/iti.npz
hallucination-mitigation run --method iti --model meta-llama/Llama-2-7b-chat-hf \
  --probe runs/iti.npz --prompt "..."

# R4：真实模型 API（仅 CoVe）
export OPENAI_API_KEY="sk-..."
hallucination-mitigation run --method cove --backend openai \
  --model gpt-4o-mini --api-base https://api.openai.com/v1 \
  --prompt "请介绍蓝鲸的食性。"

# OpenAI-compatible：vLLM / Ollama / LM Studio / TGI
export HM_API_KEY="local-key"
hallucination-mitigation run --method cove --backend openai-compatible \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-base http://localhost:8000/v1 --api-key-env HM_API_KEY \
  --prompt "..."

# R5：真实数据对比与评测
hallucination-mitigation compare --model Qwen/Qwen2.5-1.5B-Instruct \
  --methods dola,cad,cove --context "..." --prompt "..."
hallucination-mitigation benchmark --dataset truthfulqa_mc2 \
  --model Qwen/Qwen2.5-1.5B-Instruct --methods dola,caa --limit 50 \
  --out runs/bench.md
hallucination-mitigation benchmark --dataset nqswap \
  --model Qwen/Qwen2.5-1.5B-Instruct --methods cad --limit 50 \
  --out runs/cad_bench.md
```

模型/后端配置参数：

```bash
--model <hf-id|local-path|api-model>
--backend hf|openai|openai-compatible|callable
--api-base <url>
--api-key-env <ENV_NAME>
--timeout <seconds>
--max-retries <n>
--device cuda|cpu|mps
--dtype bfloat16|float16|float32
--revision <hf-revision>
--local-files-only
```

### 8.2 `run` 的默认行为

- 默认同时生成 baseline 与 mitigated；
- `--no-baseline` 可省一次生成；
- `--format markdown|json`，默认 markdown；
- `--out` 写文件；不指定则 stdout；
- `--seed` 默认 0；`--max-new-tokens` 默认 128；
- `--context` / `--context-file` 仅供 CAD；缺 context 时 `run --method cad` 报 `ConfigError`；
- 缺 artifact 时直接报错，不自动拟合；
- gated HF 模型先检查登录/许可，给出 `huggingface-cli login` 提示；
- API key 只从环境变量读取，**不接受命令行明文 key**；
- API 后端只允许 CoVe；其他方法报 `UnsupportedModelError`。

### 8.3 `demo --offline`

- **定位：安装冒烟，不是最终验收**；
- 用内存构建的 2 层随机 tiny 模型 + 内置 demo CAA/ITI artifact；
- 依次跑 DoLa、CAD、CAA、CoVe（CoVe 用 `CallableBackend` 脚本输出）；
- CAD 使用内置合成 context，验证双流/EOS/组合公式；
- 输出一份报告，验证 API、CLI、序列化；
- 不下载、不联网、不需要 GPU；
- 文档明确“仅验证管线，不代表真实效果”。

### 8.4 `doctor`

检查并打印：

- Python / 包版本；
- torch / transformers / CUDA / GPU 显存；
- `[hf]`/`[iti]`/`[openai]` 是否安装；
- HF token 是否存在（不打印 token）；
- 模型缓存路径；
- 本地模型路径是否存在、是否可读；
- `--backend openai --api-base ...` 时做一次轻量连通性检查（`/models` 或最小 chat），
  不发送用户 prompt；
- 可用方法列表，以及“哪些方法需要本地白盒模型”。

### 8.5 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数/配置错误 |
| 3 | 缺依赖/模型不支持 |
| 4 | artifact 缺失/不匹配 |
| 5 | 后端/生成失败 |

---

## 9. Presets 与配置

### 9.1 YAML schema

```yaml
name: caa_tinyllama
method: caa
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
backend: hf                # hf | openai | openai-compatible
revision: null
dtype: bfloat16
device: cuda
local_files_only: false
api_base: null
api_key_env: OPENAI_API_KEY
requires: [hf]
requires_context: false   # CAD preset 为 true
generation:
  max_new_tokens: 128
  do_sample: false
  temperature: 1.0
  top_p: 1.0
method_args:
  layer: 10
  multiplier: 1.0
  position: generated
artifact: null          # CAA/ITI 需要用户拟合后填入
metadata:
  paper: "..."
  notes: "layer/multiplier 是起点，不是通用最优"
```

CAD preset 的差异：

```yaml
name: cad_qwen2_5_1_5b
method: cad
model: Qwen/Qwen2.5-1.5B-Instruct
requires_context: true
method_args:
  alpha: 0.5
  context_template: chat
  filter_top_p: 1.0
```

API preset 的差异：

```yaml
name: cove_openai_gpt4o_mini
method: cove
backend: openai
model: gpt-4o-mini
api_base: https://api.openai.com/v1
api_key_env: OPENAI_API_KEY
requires: [openai]
method_args:
  max_questions: 6
  mode: factored
```

```yaml
name: cove_vllm_qwen
method: cove
backend: openai-compatible
model: Qwen/Qwen2.5-7B-Instruct
api_base: http://localhost:8000/v1
api_key_env: HM_API_KEY
requires: [openai]
```

### 9.2 内置 preset

| preset | 方法 | 模型 | 是否需拟合 | 上下文 | 用途 |
|---|---|---|---|---|---|
| `demo_tiny` | 全部 | 内存 tiny 模型 | 否 | 内置合成 context | 离线自检 |
| `dola_tinyllama` | DoLa | TinyLlama-1.1B-Chat | 否 | 不需要 | 英文/轻量 demo |
| `cad_tinyllama` | CAD | TinyLlama-1.1B-Chat | 否 | **必需** | 上下文忠实性 demo |
| `cove_tinyllama` | CoVe | TinyLlama-1.1B-Chat | 否 | 不需要 | 后处理 demo |
| `caa_tinyllama` | CAA | TinyLlama-1.1B-Chat | **是** | 不需要 | 激活 demo |
| `dola_qwen2_5_1_5b` | DoLa | Qwen2.5-1.5B-Instruct | 否 | 不需要 | 中文 demo |
| `cad_qwen2_5_1_5b` | CAD | Qwen2.5-1.5B-Instruct | 否 | **必需** | 中文上下文 demo |
| `cove_qwen2_5_1_5b` | CoVe | Qwen2.5-1.5B-Instruct | 否 | 不需要 | 中文后处理 |
| `caa_qwen2_5_1_5b` | CAA | Qwen2.5-1.5B-Instruct | **是** | 不需要 | 中文激活 |
| `dola_llama2_7b` | DoLa | Llama-2-7b-chat-hf | 否 | 不需要 | 论文复现 |
| `cad_llama2_7b` | CAD | Llama-2-7b-chat-hf | 否 | **必需** | 论文复现 |
| `caa_llama2_7b` | CAA | Llama-2-7b-chat-hf | **是** | 不需要 | 论文复现 |
| `iti_llama2_7b` | ITI | Llama-2-7b-chat-hf | **是** | 不需要 | 可选复现 |
| `cove_openai_gpt4o_mini` | CoVe | gpt-4o-mini（API） | 否 | 不需要 | 真实 API 验收 |
| `cove_vllm_qwen` | CoVe | Qwen2.5-7B（vLLM） | 否 | 不需要 | 本地 API 验收 |

默认中文 demo 用 **Qwen2.5-1.5B-Instruct**（Apache-2.0，中文更好）；英文/轻量用
TinyLlama；论文复现用 Llama-2/3（需 Meta 许可）。不在 v0.1 捆绑任何模型权重。

### 9.3 配置优先级

```text
CLI 参数 > 环境变量 > preset YAML > 方法默认值
```

冲突时在报告中记录最终生效值。

### 9.4 模型配置：HF Hub / 本地 / API

这是最终验收的核心。三种入口必须都能实际生成，而不是只跑 demo：

| 入口 | 配置 | 支持方法 | 说明 |
|---|---|---|---|
| **HF Hub** | `--model Qwen/Qwen2.5-1.5B-Instruct` | DoLa、CAD、CAA、CoVe、ITI | 首次下载；gated 模型需登录/许可 |
| **本地路径** | `--model /models/... --local-files-only` | DoLa、CAD、CAA、CoVe、ITI | 断网可运行；需完整模型目录 |
| **OpenAI 官方** | `--backend openai --model gpt-4o-mini --api-base https://api.openai.com/v1` | **仅 CoVe** | key 从 `OPENAI_API_KEY` |
| **OpenAI-compatible** | `--backend openai-compatible --api-base http://localhost:8000/v1` | **仅 CoVe** | vLLM/Ollama/LM Studio/TGI |

示例：

```bash
# HF Hub
hallucination-mitigation run --method dola \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype bfloat16 \
  --prompt "请介绍蓝鲸的食性。"

# 本地模型，断网
hallucination-mitigation run --method cad \
  --model /models/Qwen2.5-1.5B-Instruct --local-files-only \
  --context "阿根廷在 1978、1986 和 2022 年赢得了世界杯。" \
  --prompt "阿根廷赢过多少次世界杯？"

# OpenAI API（仅 CoVe）
export OPENAI_API_KEY="sk-..."
hallucination-mitigation run --method cove --backend openai \
  --model gpt-4o-mini --prompt "请介绍蓝鲸的食性。"

# 本地 vLLM（仅 CoVe）
export HM_API_KEY="local-key"
hallucination-mitigation run --method cove --backend openai-compatible \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-base http://localhost:8000/v1 --api-key-env HM_API_KEY \
  --prompt "请介绍蓝鲸的食性。"
```

配置规则：

1. **API key 只从环境变量读取**，CLI 不接受明文 key；
2. API 后端只允许 CoVe；其他方法在发请求前报 `UnsupportedModelError`；
3. `--local-files-only` 不访问网络，缺失文件直接报错；
4. 本地路径必须校验 `config.json`、tokenizer、权重文件，缺一不可；
5. 报告记录 `provider`、`model`、`revision`、`api_base`（不含 key）、`local_files_only`；
6. 真实 API 验收使用用户自己的 key；CI 只用 mock server，不把 key 写进仓库。

---

## 10. 评测与基准

### 10.1 协议

同一模型、同一 prompt、同一 seed、同一 generation 参数下，配对比较 baseline 与 mitigated：

```text
for each example:
    baseline = generate(prompt)
    mitigated = method(prompt)
    score both
aggregate: metric delta, win/tie/loss, latency, cost
```

### 10.2 数据集

| 数据集 | 任务 | 指标 | 适用方法 |
|---|---|---|---|
| TruthfulQA-MC2 | 多选 | MC2 accuracy | DoLa、CAA、ITI |
| NQ-Swap | 上下文问答 | Exact Match | CAD |
| CNN-DM | 摘要 | ROUGE-1/2/L | CAD |
| 自定义 JSONL | QA/事实性 | EM / 参考匹配 / 可选 judge | 全部 |
| TruthfulQA-Gen（可选） | 长回答 | LLM judge / 人工 | CoVe |
| 领域数据（用户提供） | 任意 | 用户定义 | 全部 |

NQ-Swap/CNN-DM 按需下载，不打包；CAD 评测必须使用带 context 的样本，不能把普通
TruthfulQA 直接喂给 CAD。

### 10.3 指标

- 主指标：accuracy / exact match / ROUGE / 用户提供的 score；
- 配对：win/tie/loss、McNemar（可选）、bootstrap 95% CI；
- 成本：延迟、tokens、forward 次数、峰值显存（CAD 固定 2× 前向）；
- 质量护栏：长度比、重复率、distinct-n、空输出/截断率；
- 过校正：baseline 正确 → mitigated 错误的比例；
- 可选：检测包 risk 变化（`[detect]`），仅作参考。

### 10.4 报告

```text
method   n   baseline   mitigated   delta   win/tie/loss   latency   tokens
dola    50     0.42        0.54      +0.12    30/12/8       1.08x     1.0x
caa     50     0.42        0.56      +0.14    32/10/8       1.01x     1.0x
cad     50     0.61        0.72      +0.11    28/14/8       2.02x     2.0x
cove    50     0.42        0.50      +0.08    25/15/10      3.6x      3.5x
```

同时输出退化样例和“possible over-correction”告警。超参数只能在 validation split 上选，
test split 只做最终报告。

---

## 11. 测试与质量

### 11.1 测试金字塔

| 层级 | 内容 | 默认 |
|---|---|---|
| 单元 | 公式、JSD/APC、均值差、hook、artifact、prompt 解析 | 是 |
| 契约 | 报告 schema、异常、CLI 退出码、preset 校验 | 是 |
| tiny 端到端 | 随机 2 层模型跑 DoLa/CAA/CAD/CoVe/ITI | 是 |
| 集成 | TinyLlama、Qwen2.5、Llama-2/3、真实 OpenAI-compatible API | 否，`-m integration` |
| 打包 | wheel 构建、干净 venv 安装、真实模型 smoke | 是（CI 用 tiny；真实模型手动） |
| 性能 | 延迟/tokens/显存冒烟 | 否，`-m slow` |

### 11.2 tiny 夹具

- `LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
  num_attention_heads=4, num_key_value_heads=4, vocab_size=128)`；
- tokenizer 用 `tokenizers` 的 WordLevel 在内存构建；
- 生成随机权重，固定 seed；
- 不下载任何模型/分词器。

### 11.3 关键测试

- DoLa：固定 logits 的公式测试；层映射；APC；`official_compat`；
- CAA：`mean(pos-neg)` 符号；`multiplier=0` 等于 baseline；hook 位置；
- CAD：`alpha=0` 等于带 context 的 baseline；双流 cache/EOS；plain/chat 快照；
- ITI：hook 增量；prompt 不干预；合成探针；artifact 校验；
- CoVe：四阶段顺序；问题解析；空输出回退；成本统计；
- Backend：`HFBackend` 本地路径/离线；`OpenAIBackend` 的 401/429/超时/重试；
  `api_base` 拼接、key 环境变量、key 不落日志；
- API 约束：`backend=openai` + `method=dola/caa/cad/iti` 必须提前报错；
- CLI：`demo --offline`、`run --model`、`run --backend openai`、`compare`、缺 context/artifact 错误；
- 报告：JSON schema、Markdown 快照、provenance（provider/model/api_base/key 来源）；
- 打包：wheel 中 preset/artifact 存在，`pip install` 后可 import。

### 11.4 覆盖率

- 核心 contracts/engine：≥85%；
- 方法实现：≥80%；
- CLI：≥70%；
- 集成测试不计入覆盖率门槛。

### 11.5 代码质量

- `ruff check` + `ruff format --check`；
- `mypy`（核心模块严格，第三方 stub 缺失处 `ignore_missing_imports`）；
- `pre-commit` 统一；
- 公共 API 有 docstring 和类型标注；
- `py.typed`。

---

## 12. CI/CD 与发布

### 12.1 GitHub Actions

`ci.yml`：

- matrix：Ubuntu / Windows / macOS × Python 3.10/3.11/3.12；
- CPU-only torch；
- `ruff`、`mypy`、`pytest -m "not integration"`；
- `python -m build`、`twine check`；
- wheel 安装到干净 venv，跑 tiny 模型 smoke（不要求下载真实模型）；
- 用本地 mock OpenAI-compatible server 测试 API 路径，不使用真实 key；
- 上传 coverage。

`docs.yml`：`mkdocs build --strict`，main 分支部署。

`integration.yml`（手动/夜间）：TinyLlama/Qwen2.5、Llama-2 7B（GPU）、真实
OpenAI-compatible API；key 从 GitHub Secrets 读取，未配置则跳过。

`release.yml`：tag 触发，构建 sdist/wheel，发布 TestPyPI → PyPI，附 changelog。

### 12.2 版本与发布

- 语义化版本；0.1.0 为首个可用版本；
- `CHANGELOG.md` 按 Keep a Changelog；
- 发布前 checklist：测试、文档、wheel、clean venv、许可证、示例；
- 公共 API 变更需 deprecation 周期（至少一个 minor）。

### 12.3 文档

- `README.md`：中文快速开始 + English API summary；
- `docs/quickstart.md`：安装 + **HF Hub/本地模型真实运行** + API 配置；`demo` 只作为冒烟；
- `docs/methods/*.md`：公式、参数、失败模式；
- `docs/fidelity.md`：与官方实现的差异；
- `docs/presets.md`：每个 preset 的模型/许可证/超参来源；
- `docs/backends.md`：HF Hub、本地路径、OpenAI-compatible、vLLM/Ollama/LM Studio、
  API key 环境变量、超时/重试；
- `docs/benchmarks.md`：可复现表格；
- `docs/troubleshooting.md`：OOM、gated model、缺依赖、artifact 不匹配、API 401/429；
- API 文档用 mkdocstrings 自动生成。

### 12.4 示例

- `00_quickstart.ipynb`：安装 + 真实 HF 模型 + 第一个报告；demo 只作冒烟；
- `01_dola.ipynb`：TinyLlama/Qwen，逐层选择可视化；
- `02_caa_fit_and_apply.ipynb`：拟合 → 层/乘数扫描 → 应用；
- `03_cad.ipynb`：上下文对比、双流轨迹、alpha 扫描；
- `04_cove.ipynb`：本地 HF + OpenAI-compatible 两种 backend；
- `05_compare_methods.ipynb`：同一模型比较 DoLa/CAA/CAD/CoVe；
- `06_iti_optional.ipynb`：拟合探针 + 应用；
- `07_custom_benchmark.ipynb`：自定义 JSONL 评测。

离线 notebook 保存输出；真实模型 notebook 不预填 key/路径。

---

## 13. 安全、许可与合规

### 13.1 模型加载

- `trust_remote_code=False` 默认；开启需显式参数并警告；
- 不在库代码中执行模型仓库的任意代码；
- 报告记录模型 revision，避免“同名不同权重”；
- 4-bit/量化标实验性，hook 行为需测试。

### 13.2 Artifact 安全

- `np.load(..., allow_pickle=False)`；
- 校验 shape、dtype、NaN/Inf、checksum；
- 拒绝与模型 fingerprint 不匹配的 artifact；
- 不自动下载不可信 artifact；`hf://` 需显式开启并校验 SHA256。

### 13.3 数据与隐私

- 不内置遥测；
- API key **只从环境变量读取**，不进入 CLI 参数、preset、日志或报告；
- 日志不打印 API key、完整 prompt（可选 `--log-prompts`）；
- API 请求默认校验证书；本地 `http://localhost` 允许，远程 HTTP 给出警告；
- OpenAI-compatible backend 只发送当前阶段所需文本；
- 文档说明数据会发送给第三方 API，以及如何配置代理/超时。

### 13.4 许可

- 包：MIT；
- DoLa/CAD/CoVe 社区代码：不复制；
- CAA/ITI：MIT，参考时保留 NOTICE；
- 模型/数据集：各自许可证；gated 模型需用户自行接受；
- CI 检查 `src/` 不含第三方版权头/无许可证代码。

---

## 14. 里程碑

| 阶段 | 时间 | 交付 | 退出标准 |
|---|---|---|---|
| **M0 基础** | 第 1 周（3–4 天） | 包骨架、pyproject、CI、`HFEngine`、`MultiStreamState`、`ModelConfig`、tiny 夹具、`doctor` | 干净 venv 安装 wheel；本地 tiny 与真实 HF 小模型均可加载；CI 绿 |
| **M1 CAA** | 第 1–2 周 | `fit-steering`、`SteeringArtifact`、hook、`run --method caa`、`tune-steering` | 合成测试 + 真实 TinyLlama/Qwen 端到端；`multiplier=0` 等于 baseline |
| **M2 DoLa** | 第 3 周 | `methods/dola.py`、层映射、APC/JSD、preset | 公式测试；真实 TinyLlama/Qwen 生成；层轨迹可查 |
| **M3 CAD** | 第 4 周 | `methods/cad.py`、双流引擎、context 模板、preset | `alpha=0` 等于带 context baseline；真实模型双流/EOS 测试 |
| **M4 CoVe** | 第 4–5 周 | `LLMBackend`、OpenAI-compatible、四阶段流程、trace、CLI | 真实 HF + 真实 API 各一条 smoke；CallableBackend 单测 |
| **M5 ITI 可选** | 第 5–6 周 | `fit-iti`、probe artifact、hook/bake、`[iti]` extra | 合成探针；Llama-2 真实集成；缺 extra 报错清晰 |
| **M6 评测/文档** | 第 7 周 | `benchmark`（TQA/NQ-Swap/CNN-DM）、presets、docs、notebooks | 真实模型 50 条可复现；docs build |
| **M7 发布** | 第 8 周 | 性能、清理、wheel、TestPyPI、0.1.0 | 真实 HF/本地/API 验收清单全部通过 |

可并行：M2 与 M3；M3 依赖 M0 的双流抽象；M4 可与 M3 并行；M5 依赖 M1 的 hook 抽象；
M6 依赖 M1–M5 的接口稳定。

---

## 15. 风险与开放问题

| 风险 | 影响 | 应对 |
|---|---|---|
| DoLa 官方 fork 与论文实现有差异 | 结果不可比 | 以论文公式为准；`official_compat` 对照；fidelity 文档 |
| CAA 层/乘数高度敏感 | 默认值不通用 | preset 标“起点”；validation 扫描；报告告警 |
| CAD 2× 前向/两份 cache | 延迟/显存 | 报告成本；默认小模型；4-bit 实验性 |
| CAD context/chat 模板不一致 | 两条 view 不对齐 | 启动校验 + token 快照；plain 优先 |
| ITI 探针模型/数据特定 | 换模型失效 | 可选 extra；fingerprint 校验；缺 probe 报错 |
| CoVe 自检仍会错 | 虚假安全感 | `evidence="self"`；报告警告；不宣称事实核验 |
| CoVe 成本 3–4 倍 | 延迟/费用 | 限制问题数；`mode="joint"`；成本报告 |
| API key 泄露 | 安全事故 | 只读环境变量；日志/报告脱敏；不写入 preset |
| API 兼容差异 | 调用失败 | 只承诺 `/v1/chat/completions`；文档列兼容矩阵；错误映射 |
| 真实 7B 显存不足 | 无法运行 | 默认 1.5B；7B 明确 GPU 要求；4-bit 实验性 |
| 本地模型目录不完整 | 加载失败 | 启动校验 config/tokenizer/weights；给出缺失文件 |
| gated 模型许可 | 一键失败 | `doctor` 检查 token；文档说明；默认用 Apache 模型 |
| Chat template 差异 | CoVe/CAD 输出错位 | 快照测试；plain/chat 两种模式 |
| 4-bit + hook/双流 | 数值/兼容问题 | v0.1 标实验性；默认不启用 |
| CPU 太慢 | 体验差 | 默认 1.5B/TinyLlama；7B 明确 GPU 要求 |
| 范围蔓延（RARR） | 延期 | 写入 backlog；0.1 不做 |

开放问题与建议：

1. 中文默认模型用 Qwen2.5-1.5B 还是 TinyLlama？**建议 Qwen2.5-1.5B 作为中文真实模型，TinyLlama 作为轻量英文模型。**
2. 是否随包发布 demo CAA/ITI artifact？**建议只发布 tiny 模型的 demo artifact（<100 KB），真实模型要求用户拟合。**
3. CoVe 是否 v0.1 支持 OpenAI/OpenAI-compatible？**必须支持，这是最终验收项；官方 OpenAI 与本地 `/v1` 都要测。**
4. CAD 默认 α 用 0.5 还是 1.0？**建议 0.5（CNN-DM 设置）作为通用起点，NQ-Swap preset 用 1.0，且报告标注来源。**
5. 是否提供 Web UI？**建议 v0.2；v0.1 的验收是 Python API/CLI 真实模型使用。**
6. 是否支持 batch>1？**建议 v0.2；v0.1 单序列，保证 hook/双流轨迹正确。**

---

## 16. 验收清单（0.1.0）

> **最终验收必须使用真实模型/API；`demo --offline` 只做安装冒烟，不能作为验收证据。**

### A. 安装与环境

- [ ] 干净 venv：`pip install dist/*.whl` 成功；
- [ ] `hallucination-mitigation --version`、`doctor` 正常；
- [ ] `doctor` 能识别 `[hf]`/`[iti]`/`[openai]`、CUDA、HF token、本地模型路径；
- [ ] `demo --offline` 通过（仅冒烟，不计入最终验收）。

### B. Hugging Face Hub 真实模型

使用 `Qwen/Qwen2.5-1.5B-Instruct` 或 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`（真实权重，非随机 tiny）：

- [ ] `run --method dola --model Qwen/Qwen2.5-1.5B-Instruct --prompt ...` 成功；
- [ ] `run --method cad --model ... --context ... --prompt ...` 成功；
- [ ] `run --method cove --model ... --prompt ...` 成功；
- [ ] `fit-steering --method caa --model ... --data ...` 成功；
- [ ] `run --method caa --model ... --steering ... --prompt ...` 成功；
- [ ] 报告包含真实 baseline / mitigated 文本、`provider=hf`、模型 id/revision、耗时；
- [ ] 输出非空、无 NaN/异常截断；CAD 的 `alpha=0` 与 baseline 一致。

### C. 本地模型路径（断网）

- [ ] 将模型完整目录放到 `/models/...`；
- [ ] `--local-files-only` 下 DoLa、CAD、CoVe 至少各跑通一次；
- [ ] 运行期间无网络访问；缺失文件时报错清晰；
- [ ] 报告记录 `local_files_only=true` 和本地路径（可脱敏）。

### D. 真实模型 API（仅 CoVe）

- [ ] OpenAI 官方 API：配置 `OPENAI_API_KEY`，跑通 CoVe；
- [ ] OpenAI-compatible：vLLM / Ollama / LM Studio / TGI 任一，配置 `--api-base` 跑通；
- [ ] 报告记录 `provider=openai/openai-compatible`、`model`、`api_base`（不含 key）；
- [ ] key 不出现在日志/报告/异常信息中；
- [ ] 401/404/429/超时各有一条可操作错误信息；
- [ ] `backend=openai` + `method=dola/caa/cad/iti` 在发请求前报 `UnsupportedModelError`。

### E. 真实数据评测

- [ ] `benchmark --dataset truthfulqa_mc2 --model ... --methods dola,caa --limit 50` 可复现；
- [ ] `benchmark --dataset nqswap --model ... --methods cad --limit 50` 可复现；
- [ ] 报告包含 baseline/mitigated 配对结果、win/tie/loss、延迟、tokens、退化告警；
- [ ] 超参数来自 validation split，test split 只做最终报告。

### F. 包与文档

- [ ] wheel 内含 preset/demo artifact，安装后可读；
- [ ] `docs/quickstart.md` 的“真实 HF 模型”步骤可照做；
- [ ] `docs/backends.md` 覆盖 HF Hub、本地路径、OpenAI、vLLM/Ollama；
- [ ] `docs/troubleshooting.md` 覆盖 OOM、gated model、API 401/429、artifact 不匹配；
- [ ] LICENSE/NOTICE 完整，无无许可证代码；
- [ ] `CHANGELOG.md` 与 0.1.0 一致；
- [ ] CI（Linux/Windows/macOS）全绿；
- [ ] TestPyPI 安装验证通过。

### G. 明确不作为验收的内容

- [ ] `demo --offline` 的合成 tiny 输出；
- [ ] 仅 mock backend 的单元测试；
- [ ] 只有 `--dry-run`/`--help` 通过；
- [ ] 只下载模型但未完成一次真实生成。

---

## 17. 参考

**论文**

- DoLa: *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models*, ICLR 2024. https://arxiv.org/abs/2309.03883
- CAA: *Steering Llama 2 via Contrastive Activation Addition*. https://arxiv.org/abs/2312.06681
- ITI: *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*, NeurIPS 2023. https://arxiv.org/abs/2306.03341
- CAD: *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding*. https://arxiv.org/abs/2305.14739
- CoVe: *Chain-of-Verification Reduces Hallucination in Large Language Models*. https://arxiv.org/abs/2309.11495

**代码**

- DoLa: https://github.com/voidism/DoLa (`805230e`)
- CAA: https://github.com/nrimsky/CAA (`5dabbbd`, MIT)
- ITI: https://github.com/likenneth/honest_llama (`2c6b217`, MIT)
- CAD: https://github.com/xhan77/context-aware-decoding (`1281d7f`，无 LICENSE)
- CoVe 社区实现（仅参考，不复制）: https://github.com/ritun16/chain-of-verification (`23dde5d`)

**相关实现**

- pyvene: https://github.com/stanfordnlp/pyvene
- RepE: https://github.com/andyzoujm/representation-engineering
- repeng: https://github.com/vgel/repeng
- IBM activation-steering: https://github.com/IBM/activation-steering

**书稿**

- `../book.md`：§7.2.1 层间对比解码、§7.2.2 推理激活干预、§7.3 忠实性增强解码。
