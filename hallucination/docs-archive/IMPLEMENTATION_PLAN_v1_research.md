# 幻觉缓解工具包（hallucination.mitigation）实施计划

> 目标：把 `book.md` 推理层面缓解部分的三种代表性方法——DoLa、ITI、CAD——做成与
> `hallucination.detection` 风格一致的 Python 工具包：一个 `Mitigator` 门面、一条 CLI、
> 一组可复现的 preset，以及可检查的中间轨迹。
>
> 本计划基于 2026-10-25 对三个官方仓库的检索与源码审阅（commit 见 §3）。

---

## 1. 范围与定位

### 1.1 本包做什么

| 能力 | v0.1 目标 |
|---|---|
| 方法 | DoLa（层间对比解码）、ITI（推理激活干预）、CAD（上下文感知解码） |
| 模型 | 本地 Hugging Face causal LM，Llama 系优先；单机单卡/CPU，batch=1 |
| 输入 | `prompt`；CAD 额外需要 `context` |
| 输出 | baseline 与 mitigated 两段文本、逐 token/逐层干预轨迹、耗时与显存、告警 |
| 调用 | `Mitigator(...).mitigate(...)` 与 `hallucination-mitigation` CLI |
| 评测 | baseline vs mitigated 对比；TruthfulQA-MC2、NQ-Swap、CNN-DM、自定义 JSONL |
| 可复现 | 模型 revision、dtype、device、随机种子、包版本、方法参数全部写入结果 |

### 1.2 非目标（v0.1 不做）

- 不做训练/微调；ITI 只做“少量标注数据拟合线性探针”，不是训练语言模型。
- 不服务闭源 API 模型。DoLa/ITI 需要隐藏状态或注意力头，CAD 需要每步两次前向与
  logits，API 场景既不可行也不经济。
- 不做 RAG、事实核查、CoVe 事后修正；这些属于检测包或后续版本。
- 不做多卡张量并行、batch>1、流式输出；这些在 v0.2 之后按需求增加。
- 不打包模型权重或 TruthfulQA 数据；首次运行按需下载，并尊重各自许可证。

### 1.3 与检测包的关系

`hallucination.detection` 与 `hallucination.mitigation` 是同一仓库下的两个独立
distribution：

```text
hallucination/
  book.md
  hallucination.detection/       # hallucination-detection，检测与风险报告
  hallucination.mitigation/      # hallucination-mitigation，解码/激活干预
```

- 不把 mitigation 逻辑塞进 detection 包；检测包保持“只检测、不改写”的边界。
- mitigation 包提供可选 extra `[detect]`，安装后可在报告里附加检测分数：
  baseline 与 mitigated 各跑一次 `Detector`，展示风险变化。检测分数不是事实真值，
  报告中必须保留这句警告。
- 两个包可以共享 tiny-model 测试夹具和文档风格，但不共享会阻碍独立安装的核心代码。

---

## 2. 书稿中的三种方法

| 书稿位置 | 方法 | 核心公式 | 需要的模型访问 |
|---|---|---|---|
| §7.2.1 层间对比解码 | DoLa | `p_t = softmax(log q_N - log q_M)`，`M = argmax_j JSD(q_N \|\| q_j)` | 每层 hidden state、`lm_head`、逐 token logits |
| §7.2.2 推理激活干预 | ITI | `x^{l+1} = x^l + Σ_h Q_l^h(Att_l^h(P_l^h x^l) + α σ_l^h θ_l^h)` | 注意力头输出、可插入前向 hook、少量标注数据 |
| §7.3 忠实性增强解码 | CAD | `p_t = softmax((1+α) log q(y_t\|c,x,y_<t) - α log q(y_t\|x,y_<t))` | 同一模型两次前向（有/无上下文）、逐 token logits |

三种方法的共同点：**不修改模型参数、只在推理时干预**；不同点：DoLa 改 logits，
ITI 改激活，CAD 改条件分布。工具包应把“共同部分”（模型加载、逐 token 解码、采样、
轨迹记录）沉到统一引擎，把“不同部分”留在三个方法文件里。

---

## 3. 官方代码库审计

### 3.1 检索结果

| 方法 | 官方仓库 | 审阅 commit | 许可证 | 代码形态 |
|---|---|---|---|---|
| DoLa | https://github.com/voidism/DoLa | `805230e`（2025-01-17） | README 标注 MIT，但仓库根目录**没有 LICENSE 文件**；GitHub API 识别为无许可证 | 单文件 `dola.py` + **完整 fork 的 transformers 4.28.1** + 各数据集评测脚本 |
| ITI | https://github.com/likenneth/honest_llama | `2c6b217`（2025-01-27，master） | **MIT** | 研究脚本：`get_activations/`、`validation/`、`utils.py`、`interveners.py`，依赖 pyvene/baukit |
| CAD | https://github.com/xhan77/context-aware-decoding | `1281d7f`（2024-11-17） | **无 LICENSE 文件**，GitHub API 识别为无许可证 | `accelerate` 多进程 `group_decode_fileio.py` + JSONL 输入 + 评测脚本 |

### 3.2 关键事实

**DoLa**

- `dola.py` 的 `DoLa.generate()` 调用的是 `model.generate(..., dola_decoding=True,
  mature_layer=..., candidate_premature_layers=...)`；这些参数只存在于仓库内
  `transformers-4.28.1/` 的补丁版本中。
- 补丁的核心在 `generation/utils.py` 的 `dola_greedy_decode` / `dola_sample`，以及
  `models/llama/modeling_llama.py` 的 `early_exit_layers`：对指定层
  `self.lm_head(outputs.hidden_states[layer])` 得到该层 logits。
- README 明确写“currently we only support LLaMA-v1”；官方脚本用 `huggyllama/llama-7b`
  等旧模型。
- 生成路径与 `lm_score` 的归一化顺序不完全一致：`lm_score` 对 mature/premature 都做
  `log_softmax` 再相减；`dola_greedy_decode` 是 mature 原始 logits（经 relative-top
  mask）减去 premature 的 log-softmax。工具包以论文公式为准，另设 `official_compat`
  用于数值对照。
- 结论：**不能直接 `pip install`，也不能把 4.28.1 的 fork 带进现代模型环境**。

**ITI**

- `get_activations.py` 在 TruthfulQA-MC2 的每个答案最后一个 token 上，收集每一层
  `self_attn.o_proj.input`（即各头输出拼接）的激活；`utils.train_probes()` 对每个
  (layer, head) 训练一个 LogisticRegression，按验证准确率选 top-K 个头。
- `validate_2fold.py` 用 pyvene 在 `model.layers[l].self_attn.o_proj.input` 上注入
  `ITI_Intervener`：`b[0, -1] += direction * alpha`，其中 direction 已按投影标准差
  缩放；默认 `K=48`、`alpha=15`。
- `edit_weight.py` 把同样的偏移量写进 `o_proj.bias`，得到“baked-in”模型，推理时零额外
  开销；README 也提供了 Hugging Face 上已烘焙的 Llama-2/Llama-3 模型。
- 结论：**ITI 不是纯推理方法，探针拟合是方法的一部分**；没有“通用真实性方向”。
  一键调用必须要么先 `fit-iti` 一次，要么加载预拟合 artifact/baked-in 模型。

**CAD**

- 官方实现不是单模型 logits processor，而是一个多进程“group decode”框架：每个
  JSONL 实例给出 `context_string`、`assigned_model`、`assigned_weight`，各进程独立
  采样一个 one-hot token，`all_reduce` 加权求和后再 softmax/argmax。
- 典型权重：CNN-DM 用 `1.5 / -0.5`（α=0.5），NQ-Swap 用 `2 / -1`（α=1）。
- 依赖较旧：`transformers==4.26.0`、`torch==1.13.1`、`accelerate==0.16.0`。
- 结论：**官方代码是评测框架，不是可复用库**；单模型 CAD 需要按论文公式重新实现为
  “两次前向 + logits 组合”的解码器。官方 one-hot 组合与论文的 logits 组合并不完全
  等价，需在 fidelity 文档中说明。

### 3.3 复用边界

| 内容 | 是否复用 | 说明 |
|---|---|---|
| 论文公式、参数范围、提示格式 | 是 | 以论文和书稿为准 |
| DoLa 的 transformers fork | **否** | 旧版本、与 Llama-3/Qwen2 不兼容；只借鉴算法 |
| CAD 多进程代码 | **否** | 无许可证，且与“单模型一键调用”目标不符；独立实现 |
| ITI 的探针/干预代码 | 可参考 | MIT；仍以独立实现为主，保留版权声明 |
| 官方评测脚本/数据 | 否 | 数据按需下载，脚本不复制 |

> 许可提醒：DoLa 根目录无 LICENSE、CAD 无任何许可证。工具包不复制其代码，只实现
> 论文算法并引用仓库；若未来要 vendoring，必须先获得作者书面许可。

---

## 4. 包结构

```text
hallucination.mitigation/
  pyproject.toml
  README.md
  IMPLEMENTATION_PLAN.md
  docs/
    fidelity.md                 # 与三个官方实现的逐项差异
    methods.md                  # 公式、参数、失败模式
  configs/
    demo_tiny.yaml
    dola_llama2_7b.yaml
    dola_llama3_8b.yaml
    cad_llama3_8b.yaml
    iti_llama2_7b.yaml
  src/hallucination.mitigation/
    __init__.py                 # 延迟导入，未装 torch 也能 import
    api/
      mitigate.py               # Mitigator 门面
      report.py                 # MitigationReport
    core/
      contracts.py              # Request / Result / Event / Provenance
      base.py                   # BaseMitigationMethod
      config.py                 # YAML preset 解析
      prompt.py                 # CAD 的 context/chat 模板
      sampling.py               # top-p/top-k/temperature/stop
    engine/
      hf.py                     # 模型/分词器加载、dtype/device
      decode.py                 # 逐 token 引擎、KV cache、trace
      hooks.py                  # hidden-state 捕获、attention-head 干预
      logits.py                 # log_softmax/JSD/relative-top/组合
    methods/
      dola.py
      cad.py
      iti.py
    probes/
      fit.py                    # ITI 探针拟合
      artifact.py               # 保存/加载/校验
    evaluation/
      benchmark.py              # baseline vs mitigated
      datasets.py               # truthfulqa_mc2 / nqswap / cnndm / jsonl
      metrics.py                # accuracy/EM/ROUGE/repetition/KL
    cli.py
    __main__.py
  tests/
  examples/
```

依赖分层：

```toml
[project]
dependencies = ["numpy>=1.24", "pyyaml>=6.0"]

[project.optional-dependencies]
hf = ["torch>=2.0", "transformers>=4.45", "accelerate>=0.30"]
iti = ["scikit-learn>=1.4"]        # 探针拟合
quant = ["bitsandbytes>=0.43"]     # 4-bit，可选
detect = ["hallucination-detection>=0.3"]
dev = ["pytest>=8.0"]
all = [...]
```

基础安装保持轻量；`import hallucination.mitigation` 不导入 torch。缺少 `[hf]` 时，
`doctor` 和 `list-methods` 仍可用，`run` 给出可操作的安装提示。

---

## 5. 统一运行时

### 5.1 为什么不复用 `model.generate`

三种方法都需要 `generate` 不暴露的东西：

- DoLa 需要每一步每一层的 hidden state，并对多层做 `lm_head` 投影；
- ITI 需要在每个解码步对注意力头输出做原位加法；
- CAD 需要两个独立 KV cache、每步两次前向，再组合 logits。

`LogitsProcessor` 只能看到最终 logits，拿不到层间状态；hook + `generate` 又很难保证
“只干预生成 token、不干预 prompt”。因此写一个**小型逐 token 引擎**，三种方法共用。
它不追求覆盖所有生成特性，只保证方法语义正确、轨迹可检查。

### 5.2 `HFEngine`

```python
engine = HFEngine.from_pretrained(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="bfloat16",
    device="cuda",
    revision="...",
)

state = engine.prefill(input_ids)          # 返回 past_key_values
out = engine.step(state, token_id, output_hidden_states=True)
# out.logits, out.hidden_states, out.past_key_values
```

要求：

- 单序列、KV cache、`use_cache=True`；
- 可选 `output_hidden_states`，返回当前步所有层的 hidden state；
- 采样函数支持 greedy / temperature / top-p / top-k / repetition penalty；
- stop strings 与 EOS 都支持；记录停止原因；
- `torch.inference_mode()`；可选 `torch.compile`（v0.2）；
- 统一 trace：每步耗时、forward 次数、当前 token、峰值显存。

### 5.3 方法接口

```python
class BaseMitigationMethod(ABC):
    name: str
    requires: set[str]

    @abstractmethod
    def run(self, engine: HFEngine, request: MitigationRequest,
            config: MethodConfig) -> MitigationResult:
        ...
```

不在基类里硬塞 `transform_logits` 之类的钩子。CAD 需要双流、ITI 需要 hook、DoLa 需要
多层投影，强行统一会写出难读的“万能钩子”。共用的是引擎、采样、轨迹和报告，不是控制流。

### 5.4 结果契约

```python
@dataclass
class MitigationRequest:
    prompt: str
    context: str | None = None          # CAD
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = 0
    stop: list[str] = field(default_factory=list)

@dataclass
class InterventionEvent:
    step: int
    kind: str                           # "layer_contrast" | "head_bias" | "context_combine"
    detail: dict[str, Any]

@dataclass
class MitigationResult:
    method: str
    response: str
    baseline_response: str | None
    events: list[InterventionEvent]
    trace: dict[str, Any]               # 每步概率/层选择/KL 等
    provenance: dict[str, Any]
    warnings: list[str]
    cost: dict[str, Any]                # latency/tokens/forward_passes
```

`MitigationReport` 负责组合多方法结果、Markdown/JSON 导出、baseline 对比表。
不引入 `score`/`risk` 字段；缓解结果的好坏由评测指标决定，不能用一个内禀分数冒充事实性。

### 5.5 失败与降级

- 缺依赖：`ImportError` 带安装命令；
- 模型不支持：加载前检查 `config`/模块结构，报“支持矩阵”而不是运行到一半失败；
- ITI 缺 probe：直接报错并给出 `fit-iti` 命令；
- 生成退化：默认只告警，不自动回退到 baseline；自动回退会掩盖方法效果，且让实验不可比。
  可选 `on_degenerate="baseline"` 仅供演示。

---

## 6. 一键调用体验

### 6.1 Python API

```python
from hallucination.mitigation import Mitigator

# 1) DoLa：预设 + 一次调用；默认同时返回 baseline
m = Mitigator(preset="dola_llama3_8b")
report = m.mitigate("阿根廷赢过多少次世界杯？")
print(report.to_markdown())
report.to_json("runs/dola.json")

# 2) CAD：context 是必填
m = Mitigator(method="cad", model="meta-llama/Meta-Llama-3-8B-Instruct", alpha=0.5)
report = m.mitigate(
    "阿根廷赢过多少次世界杯？",
    context="阿根廷在 1978、1986 和 2022 年赢得了世界杯。",
)

# 3) ITI：先拟合一次，之后一键
m = Mitigator(method="iti", model="meta-llama/Llama-2-7b-chat-hf",
              probe="probes/llama2_7b_truthfulqa.npz", alpha=15)
report = m.mitigate("...")

# 4) 同一模型连续比较三种方法（模型只加载一次）
report = Mitigator(model="...").compare(
    "阿根廷赢过多少次世界杯？",
    methods=["dola", "cad", "iti"],
)
```

`Mitigator` 的构造参数：

```python
Mitigator(
    preset=None,              # 与 method/model 二选一
    method="dola",            # dola | cad | iti
    model=None,               # HF id 或本地路径
    device="cuda",
    dtype="bfloat16",
    revision=None,            # 固定模型版本，保证可复现
    load_in_4bit=False,       # v0.1 仅实验性支持
    alpha=None,               # 方法默认值可覆盖
    **method_kwargs,
)
```

### 6.2 CLI

```bash
hallucination-mitigation doctor
hallucination-mitigation list-methods

# 一条命令：加载模型 → baseline → DoLa → 报告
hallucination-mitigation run --preset dola_llama3_8b \
  --prompt "阿根廷赢过多少次世界杯？" --out runs/dola.md

# CAD
hallucination-mitigation run --method cad --model meta-llama/Meta-Llama-3-8B-Instruct \
  --context-file context.txt --prompt "阿根廷赢过多少次世界杯？" --alpha 0.5

# 同一模型比较
hallucination-mitigation compare --model ... --methods dola,cad \
  --prompt "..." --out runs/compare.md

# ITI 拟合 + 运行
hallucination-mitigation fit-iti --model meta-llama/Llama-2-7b-chat-hf \
  --dataset truthfulqa_mc2 --top-k 48 --out probes/llama2_7b_truthfulqa.npz
hallucination-mitigation run --method iti --model ... --probe probes/...npz --prompt ...
```

统一约定：

- `run` 默认 `--baseline`，报告同时给 baseline 与 mitigated；`--no-baseline` 可省一次生成。
- `--dry-run` 用内置随机 tiny 模型跑通管线，不下载任何东西，供 CI 和新用户验证安装。
- `--format markdown|json`；`--out` 写文件。
- 所有错误在 CLI 层转成一行 `error: ...`，不打印 traceback（与检测包一致）。
- `doctor` 打印 torch/transformers/CUDA/显存/模型缓存，不加载模型。

### 6.3 Preset

```yaml
name: dola_llama3_8b
method: dola
model: meta-llama/Meta-Llama-3-8B-Instruct
revision: null
dtype: bfloat16
device: cuda
generation:
  max_new_tokens: 128
  do_sample: false
method_args:
  mature_layer: 32
  candidate_premature_layers: [16, 18, 20, 22, 24, 26, 28, 30]
  relative_top: 0.1
```

内置 preset：

| preset | 用途 | 是否需要下载 |
|---|---|---|
| `demo_tiny` | 随机 2 层模型，离线跑通 API/CLI/测试 | 否 |
| `tinyllama_chat` | TinyLlama-1.1B-Chat，Apache-2.0，适合单卡/CPU 演示 | 是，约 2 GB |
| `dola_llama2_7b` | DoLa 论文设置（TruthfulQA 高层桶） | 是，需 Meta 许可 |
| `dola_llama3_8b` | Llama-3 8B 对应层桶 | 是，需 Meta 许可 |
| `cad_llama3_8b` | CAD，α=0.5 | 是 |
| `iti_llama2_7b` | ITI，需配套 probe artifact | 是 + probe |

默认 real preset 用 **TinyLlama**，避免一上来就要求 7B/8B 和 Meta 登录；Llama preset
明确标注“需接受模型许可证”。

---

## 7. 三种方法的实现方案

### 7.1 DoLa

**逐 token 流程**

1. `prefill(prompt)`，得到第一层到最后一层的 hidden states；
2. 对候选 premature 层集合 `C` 和 mature 层 `N`：
   - `q_j = lm_head(hidden_states[j])`，`q_N = lm_head(hidden_states[N])`；
   - 仅取当前步最后一个位置；
3. 对每个 `j ∈ C` 计算 `JSD(q_N || q_j)`，选最大者为 `M`；
4. APC/relative-top：把 `q_N` 中低于 `max(q_N) + log(relative_top)` 的 token 置 `-inf`；
5. `logits = log_softmax(q_N) - log_softmax(q_M)`；可选 `post_softmax`；
6. 采样下一个 token，追加到序列，进入下一步。

**层索引映射（必须与官方一致）**

HF `hidden_states` 的第 0 项是 embedding，第 `k` 项是第 `k-1` 层 decoder 的输出。
官方 DoLa 对 Llama-7B 用 `early_exit_layers=32`，对应 `hidden_states[32]`，即最后一层
decoder 输出（未过 final norm）。因此：

- `mature_layer = num_hidden_layers`（默认）；
- `candidate_premature_layers` 取 `< num_hidden_layers`；
- 不额外对 early-exit hidden state 做 final norm（官方也没有）；
- 用 `model.get_output_embeddings()` 取 `lm_head`，兼容 tied weights。

**默认参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `mature_layer` | `num_hidden_layers` | 最后一层 decoder 输出 |
| `candidate_premature_layers` | preset 给出 | 7B/8B TruthfulQA 用 16–30 偶数；长文 FACTOR 用 0–14 偶数 |
| `relative_top` | `0.1` | 对应书稿 APC 的 α=0.1；设 0 关闭 |
| `post_softmax` | `True` | 与官方 `lm_score` 对齐；生成路径可关 |
| `official_compat` | `False` | True 时复现官方“raw mature − logprob premature”顺序，仅用于对照 |

**性能**

- 每步 1 次前向；额外 `K` 次 `lm_head` 投影 + `K` 次 JSD（vocab 维），相对 7B 前向很小；
- 显存增加约 `K × hidden_size` 的当前步 hidden state（不是整段序列）；
- 预期延迟 1.0–1.2× baseline。

**测试**

- 固定 logits 的公式测试：JSD 选层、APC mask、`log_softmax` 差值、`α=0` 退化；
- 层映射测试：tiny 模型上 `hidden_states[k]` 与 `lm_head` 形状/索引；
- tiny 模型端到端：能生成、事件数等于生成 token 数；
- `official_compat` 数值对照（用合成张量复现官方顺序）；
- 真实模型集成测试标记 `integration`，默认不跑。

**Fidelity 记录**

- 官方 fork 的生成路径与 `lm_score` 归一化顺序不一致，本包以论文公式为准；
- 官方只验证 LLaMA-v1；本包对 Llama-2/3、Mistral、Qwen2 做泛化，但需在文档中标注
  “层桶是论文/社区经验值，不是普适最优”；
- 官方 relative-top 与书稿 APC 在数学上同为“相对最大概率的阈值”，实现采用同一参数名。

### 7.2 CAD

**两条序列**

```text
view_c = PromptBuilder(context=context, question=prompt, with_context=True)
view_u = PromptBuilder(context=context, question=prompt, with_context=False)
```

`PromptBuilder` 支持两种模式：

- `plain`：`context + "\n\n" + question` / `question`；
- `chat`：把 context 放进 system message（或 user message 前缀），再套
  `tokenizer.apply_chat_template(..., add_generation_prompt=True)`；两条 view 使用同一
  模板，只差 context 块。

**逐 token 流程**

1. 分别 `prefill(view_c)`、`prefill(view_u)`，得到两个 KV cache；
2. 每步分别 `step`，取 `q_c`、`q_u`（当前步最后一个位置）；
3. 可选对每个 view 做 top-p 过滤（官方 `filter_p`）；
4. `logits = (1 + α) * log_softmax(q_c) - α * log_softmax(q_u)`；
5. 采样一个 token，**同时追加到两条序列和两个 cache**；
6. 以 context view 的 EOS 为准停止；记录两条序列的 KL/一致率。

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `alpha` | `0.5` | CNN-DM 官方 `1.5/-0.5`；NQ-Swap 用 `1.0` |
| `filter_top_p` | `1.0` | 关闭；官方可选 |
| `context_template` | `plain` | `plain` / `chat` |
| `combine` | `logits` | v0.1 只实现论文的 logits 组合 |

**与官方差异**

- 官方 group decode 是“每个 view 先采样 one-hot，再按权重求和”，支持不同模型/不同
  context 的多进程组合；本包是“单模型、两次前向、logits 加权”，符合论文公式，也是
  一键调用最自然的形式。
- 官方 `transformers==4.26`；本包用现代 `transformers`，chat template 行为可能不同，
  用快照测试固定。

**性能**

- 每步 2 次前向，约 2× 延迟；7B + 2k context 两份 KV cache 约多 1–2 GB（fp16）；
- 可选 `filter_top_p` 降低候选，不影响前向次数。

**测试**

- `α=0` 时输出必须与 baseline 完全一致（同模型、同采样、同种子）；
- 组合公式对固定 logits 的数值测试；
- plain/chat 模板的 token 快照测试；
- EOS 同步、cache 长度一致；
- 真实模型 smoke（TinyLlama）。

### 7.3 ITI

ITI 分两阶段：**拟合探针**与**推理干预**。

#### A. `fit-iti`

1. 读取标注数据：
   - 内置 `truthfulqa_mc2` 适配器（需 `.[hf,datasets]`），字段映射为
     `prompt = "Q: {question} A: {choice}"`，`label = mc2_targets.labels`；
   - 自定义 JSONL：`{"prompt": "...", "response": "...", "label": 1}` 或
     `{"text": "...", "label": 0}`；
2. 按**问题**而非答案划分 train/val，避免同一问题的不同选项泄漏；
3. 对每条样本做一次前向，在最后一个 token 处 hook 每一层
   `self_attn.o_proj.input`，reshape 成 `[num_heads, head_dim]`；
4. 每个 (layer, head) 训练 `LogisticRegression`，在 val 上算准确率；
5. 按准确率选 top-K 个头（默认 48；小模型自动截断到 `layers*heads`）；
6. 方向 = 归一化后的 `probe.coef_`；尺度 = 训练/调参集上投影值的标准差；
7. 保存 artifact（见下），并记录数据来源、模型 fingerprint、随机种子、准确率分布；
8. 若 top-1 头准确率接近随机（例如 < 0.6），给出警告：该模型/数据上 ITI 可能无效。

#### B. `run`

1. 加载 artifact，校验模型 fingerprint：
   `num_hidden_layers`、`num_attention_heads`、`hidden_size`、`head_dim`、
   `model_id/revision`；
2. 对每个被选中的层安装 forward pre-hook：
   `o_proj.input[..., current_pos, :] += α * σ * θ`；
3. 默认只干预**生成 token**（`position="generated"`），不干预 prompt；官方 pyvene 版本
   干预所有位置，`position="all"` 可复现；
4. 生成结束后移除 hook，保证同一进程内可连续跑 baseline/其他方法；
5. `bake=True` 时把偏移写进 `o_proj.bias`，得到零开销模型（与官方 `edit_weight.py`
   等价），但会修改内存中的模型，需显式确认。

#### Artifact 格式

```json
{
  "format": "hallucination-mitigation/iti-probe/v1",
  "model": {
    "id": "meta-llama/Llama-2-7b-chat-hf",
    "revision": "...",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "hidden_size": 4096,
    "head_dim": 128
  },
  "method": {"alpha": 15, "top_k": 48, "position": "generated"},
  "heads": [
    {"layer": 14, "head": 18, "direction": [0.01, -0.02, "..."], "scale": 0.42, "val_accuracy": 0.833}
  ],
  "training": {"dataset": "truthfulqa_mc2", "n_train": 800, "n_val": 200, "seed": 42}
}
```

方向向量很小（48 × 128 floats），单文件 JSON 即可；更大的模型可存 `.npz` + JSON sidecar。

#### 预拟合 artifact

- v0.1 支持本地路径与 `hf://` 引用（`hf_hub_download` + SHA256），不内置权重；
- 可选 preset 指向社区已烘焙的 `likenneth/honest_llama2_chat_7B` 这类模型：此时不是
  本包做干预，而是直接加载已编辑模型，报告中必须注明 `backend="baked_model"`；
- 不承诺“通用真实性方向”；不同模型、不同数据集、不同语言都需要重新拟合。

#### 测试

- hook 数学：给定输入与方向，输出增量等于 `α*σ*θ`；
- 位置掩码：prompt 位置不被修改，生成位置被修改；
- 探针拟合：合成两类线性可分数据，top head 应被选中；
- artifact roundtrip / fingerprint mismatch / 缺文件错误；
- `bake` 后输出与 hook 版本一致（容差内）。

---

## 8. 评测与基准

缓解方法的评测不能只看“更像真话”，还要看是否损伤流畅性/信息量。`benchmark` 的目标
是**同一模型、同一数据、同一解码参数下 baseline vs mitigated 的配对比较**。

### 8.1 数据集

| 数据集 | 任务 | 指标 | 用途 |
|---|---|---|---|
| TruthfulQA-MC2 | 多选 | MC2 accuracy | DoLa/ITI 的主要论文指标 |
| NQ-Swap | 上下文问答 | Exact Match | CAD 的主要论文指标 |
| CNN-DM | 摘要 | ROUGE-1/2/L | CAD 的忠实性/信息量 |
| 自定义 JSONL | 任意 | 用户提供 label/参考 | 领域评测 |

数据按需下载，不打包；每个数据集在 `docs/` 记录来源、许可证、字段映射、split。

### 8.2 配对报告

```text
method   n   baseline_metric   mitigated_metric   delta   win/tie/loss   latency   forward_passes
dola    200      0.41              0.52            +0.11    120/50/30      1.08x     1.0x
cad     200      0.44              0.55            +0.11    110/60/30      2.01x     2.0x
iti     200      0.41              0.58            +0.17    130/40/30      1.01x     1.0x
```

同时输出：

- 长度比、空输出率、重复率、截断率；
- 与 baseline 的 token 级 KL（在有 logits 的步骤上）；
- 退化样例（前 N 条）；
- 可选检测包风险分数变化（`[detect]` extra）。

### 8.3 过校正与质量护栏

书稿明确提到对比解码可能“过校正”、损害发散思维。报告必须包含：

- `length_ratio`、`repetition_rate`、`distinct_n`；
- `empty_or_truncated` 计数；
- `baseline_vs_mitigated` 的 win/tie/loss，而不是只报平均值；
- 若退化率超过阈值，给出 `WARNING: possible over-correction`。

不在 v0.1 自动调参；超参数（DoLa 层桶、CAD α、ITI K/α）只能在 validation split 上选，
test split 只用于最终报告。这条与检测包的校准原则一致。

---

## 9. 测试与 CI

默认测试必须离线、快速、确定性：

| 层级 | 内容 | 是否默认 |
|---|---|---|
| 单元 | 公式、JSD、APC、logits 组合、采样、stop、artifact | 是 |
| 契约 | `MitigationResult` 序列化、baseline 对比、错误信息 | 是 |
| 引擎 | 内置随机 tiny 模型（2 层、4 头、hidden 32、vocab 128）端到端 | 是 |
| CLI | `--dry-run`、`list-methods`、缺依赖/缺 probe 的错误 | 是 |
| 快照 | CAD chat/plain 模板 token ids；DoLa 层索引 | 是 |
| 集成 | TinyLlama 真实生成、7B/8B GPU、官方数值对照 | 否，`-m integration` |
| 性能 | 每方法 latency/forward 计数冒烟 | 否，`-m slow` |

关键原则：

- 测试不下载模型；tiny 模型用 `transformers` 配置在内存中构建；
- tokenizer 用内置 `tokenizers` 构造 WordLevel，或复用 tiny 模型的本地夹具；
- 任何真实模型/remote 数据测试都显式标记，不让 CI 意外联网；
- `pytest -m "not integration"` 必须在无 GPU、无网络环境通过。

---

## 10. 交付里程碑

| 阶段 | 内容 | 预计 | 退出标准 |
|---|---|---|---|
| M0 | 包骨架、contracts、`HFEngine`、tiny 模型夹具 | 2–3 天 | `doctor`/`--dry-run` 可跑；空方法报错清晰 |
| M1 | CAD + CLI `run`/`compare` + 报告 | 1 周 | α=0 与 baseline 完全一致；TinyLlama 端到端 |
| M2 | DoLa + 层映射 + fidelity 对照 | 1 周 | 公式测试通过；TinyLlama 生成；层选择轨迹可查 |
| M3 | ITI `fit-iti` + hook 干预 + artifact | 1.5 周 | 合成数据探针可用；Llama-2 7B 集成通过 |
| M4 | benchmark（TQA/NQ-Swap/CNN-DM）+ 文档/示例 | 1 周 | 配对报告可复现；退化告警生效 |
| M5 | 真机 parity、性能、打包、0.1.0 | 1 周 | 三方法在同一模型上可连续比较；wheel 安装通过 |

总计约 5–6 周（单人）。M1/M2 可并行做接口与测试，M3 依赖 M0 的 hook 抽象。

### v0.1 发布门槛

1. `pip install -e ".[hf]"` 后，`hallucination-mitigation doctor` 正常；
2. `run --preset demo_tiny` 与 `run --preset tinyllama_chat` 都能出 baseline + mitigated；
3. 三种方法至少各有一个真实模型 smoke；
4. ITI 无 probe 时给出明确 `fit-iti` 指引，不静默使用随机方向；
5. 所有公开 preset 的参数都有来源（论文/官方 README/验证集搜索）；
6. README 明确“缓解不保证事实正确，可能过校正”。

---

## 11. 风险与开放决策

| 风险 | 影响 | 应对 |
|---|---|---|
| DoLa fork 依赖旧 transformers | 无法与现代模型共存 | 独立逐 token 引擎；不引入 fork |
| CAD 无许可证 | 不能复制代码 | 只实现论文公式；引用仓库；不 vendoring |
| DoLa 根目录无 LICENSE | 法律状态不清晰 | 不复制；需要时联系作者 |
| ITI 探针依赖数据/模型 | “一键”不是零配置 | `fit-iti` + artifact；预拟合模型标注来源 |
| CAD 2× 前向 | 延迟/显存翻倍 | 报告明确成本；v0.2 研究 cache 复用/量化 |
| chat template 差异 | 两视图 token 不对齐 | `PromptBuilder` + 快照测试；优先 plain 模式 |
| 7B 显存 | 消费级卡跑不动 | TinyLlama 默认；4-bit 实验性；70B 不做 |
| 过校正/质量下降 | 指标提升但输出退化 | 质量护栏 + win/tie/loss + 告警 |
| 数据许可证 | 不能打包 | 按需下载；每数据集记录许可证 |

待确认的产品决策：

1. v0.1 是否必须支持 4-bit？建议先不做，只在文档标注“实验性”。
2. ITI 预拟合 artifact 是否随包发布？建议不发布权重，只发布格式与下载指引。
3. 是否把 mitigation 接入现有 web demo？建议 v0.2，先保证 API/CLI。
4. 是否支持非 Llama 架构？建议 v0.1 只承诺 Llama 系 + TinyLlama，其他标“实验性”。

---

## 12. 追加候选：推理阶段经典方法（v0.2+）

### 12.1 筛选标准

在 DoLa、ITI、CAD 之外，再找“推理阶段、真正改变输出、可独立封装”的方法。筛选条件：

1. 不改模型参数，或只需一次性拟合小工件；
2. 有论文、有公开代码（作者仓库或官方组织仓库）；
3. 能抽象成 `Mitigator` 的一个方法，而不是整套训练框架；
4. 许可证允许独立重实现或可选依赖；不接受把 GPL/无许可证代码直接复制进核心包；
5. 与现有三条路线互补：解码干预、激活干预、**后处理修正**、采样选择。

### 12.2 候选表

| 方法 | 类型 | 代码 | Commit | 许可证 | 现状与依赖 | 一键难度 | 结论 |
|---|---|---|---|---|---|---|---|
| **RARR** | 后处理：检索 + 编辑 | https://github.com/anthonywchen/RARR | `51a1a10` | 无 LICENSE | 作者公开实现；OpenAI `text-davinci-003` + Bing Search API + spaCy；Python 3.8 | 高 | v0.3 可选，价值高但需替换检索/生成后端 |
| **CoVe** | 后处理：验证问题 + 改写 | **无官方仓库**；社区实现 https://github.com/ritun16/chain-of-verification | `23dde5d` | 无 LICENSE | LangChain + OpenAI + DuckDuckGo；论文方法为 4 阶段 prompting | 低 | **v0.2 首选**，按论文自研，不依赖社区代码 |
| **Self-Refine** | 后处理：自反馈迭代 | https://github.com/madaan/self-refine | `9a206d4` | Apache-2.0 | 作者官方；`prompt-lib` + OpenAI `text-davinci-003`；面向通用任务，不是幻觉专用 | 低 | 可选基线，不作为核心方法 |
| **TruthX** | 白盒：真值空间表征编辑 | https://github.com/ictnlp/TruthX | `a41093a` | **GPL-3.0** | ACL 2024 官方；需 TruthX 自编码器 + 自定义 transformers/FastChat；有 baked-in Llama-2 模型 | 高 | 高级适配器/案例，不进宽松许可核心 |
| **Contrastive Decoding** | 解码：专家/业余模型对比 | https://github.com/XiangLi1999/ContrastiveDecoding | `170e914` | 无 LICENSE | 作者官方；fork transformers；目标是开放式生成质量，不是幻觉专用 | 中 | 仅作为解码基线 |
| **DExperts** | 解码：expert/anti-expert | https://github.com/alisawuffles/DExperts | `4ef198f` | 无 LICENSE | 作者官方；需要微调 expert/anti-expert；面向毒性/情感控制 | 中 | 不推荐，非幻觉专用 |
| **FAVA** | 后处理：训练式编辑模型 | https://github.com/abhika-m/FAVA | `7ec08c2` | 无 LICENSE | 需训练编辑模型 + Contriever 检索；提供 HF 权重 | 高 | 不推荐进入 v0.1，属于“训练 + 检索”系统 |

补充说明：

- **RARR** 的官方脚本依赖的 Bing Search API 已退役、`text-davinci-003` 已下线，原样跑不通；
  但它的“问题生成 → 检索 → 一致性门控 → 编辑 → 归因报告”流程很适合与检测包的
  `Evidence`/`LocalCorpusRetriever` 结合，值得做，但要替换外部服务。
- **CoVe** 严格来说没有 Meta 官方代码库（`facebookresearch`、`meta-llama` 下均未找到），
  GitHub 上的实现是社区版且无许可证。好处是方法本身只是 prompting，按论文重写成本很低，
  不需要复制代码。
- **Self-Refine** 是通用的“生成 → 反馈 → 修订”框架，不是为幻觉设计；它在事实性任务上的
  收益不稳定，适合作为 baseline，不适合作为卖点。
- **TruthX** 的效果和“推理时编辑表征”定位都很好，但 GPL-3.0 + 自定义 transformers/FastChat
  分叉，意味着只能做可选适配器；不能把代码并入 MIT/Apache 的核心包。
- **CD / DExperts** 是经典解码方法，但优化目标是生成质量/风格/毒性，不是事实性；
  可以放在“对比基线”，不应作为幻觉缓解的主打方法。
- **FAVA** 需要一个训练好的编辑模型和检索库，属于“训练 + 推理”系统，不符合“一键调用”的
  v0.1 范围。

### 12.3 推荐顺序

**v0.2：CoVe（自研）+ Best-of-N（自研）**

- CoVe 补齐“后处理修正”路线，且不依赖白盒访问；任何本地 HF 模型或 OpenAI 兼容 API 都能跑。
- 流程：baseline → 生成验证问题 → 独立回答 → 汇总修订；3–4 次 LLM 调用。
- 风险：没有外部证据时只是“模型自检”，仍可能自信地错；报告中必须标注
  `evidence="self"`，不能宣称事实核验。
- 可与 CAD 组合：CAD 负责“听上下文”，CoVe 负责“事后核对”。

**v0.3：RARR（可选 extra）**

- 复用检测包的检索/证据抽象，新增可插拔 retriever（本地语料、HTTP 搜索、向量库）。
- 需要外部检索时显式开启，默认离线；没有检索 key 时用本地 corpus 或直接报错，不伪造证据。
- 输出包含 `queries`、`evidence`、`agreement_gate`、`edits`、`citations`，可审计。

**v0.4：TruthX 适配器（可选）**

- 只做“加载 baked-in TruthX 模型 + 统一报告”的适配器，或按论文重实现编辑层；
  不复制 GPL 代码进核心。
- 适合作为“白盒表征编辑”的第二个案例，与 ITI 对比。

**可选基线：Self-Refine**

- 用统一的 `LLMBackend` 实现 2–3 轮反馈/修订，作为“无检索、无白盒”的朴素对照；
  不承诺稳定增益。

### 12.4 没有单一官方仓库但值得做：Best-of-N / 自一致性选择

这类方法没有“官方代码库”（通常是训练/评测框架的一部分），但实现成本最低，
且能直接复用检测包的分数：

```text
prompt -> N 个采样 -> 用检测分数/奖励模型/LLM judge 排序 -> 选最高分
```

- 优点：模型无关、无需训练、与 `Detector` 天然结合；
- 缺点：N 倍成本，且“检测分数高”不等于事实正确；
- 适合作为 v0.2 的第二个方法，和 CoVe 一起构成“采样选择 + 后处理修正”。

### 12.5 激活方向专题：ITI 之外的实现

如果把“激活方向”理解为**如何估计干预向量、加在哪里、加多强**，ITI 不是唯一选择，
而且在工程上不一定是最优解。

| 方法 | 方向来源 | 干预位置 | 代码/许可 | 评价 |
|---|---|---|---|---|
| **ITI** | 每个注意力头一个 Logistic 探针，`θ` 归一化后乘投影标准差 `σ` | `self_attn.o_proj.input`，逐头 | MIT，官方研究脚本 | 细粒度，但依赖头结构、探针数据和 K/α 调参；换模型易失效 |
| **CAA** | 每层 `mean(h_pos - h_neg)`，一对对比样本一个层向量 | residual stream，按层加 | MIT，官方代码；自带 hallucination 行为数据 | **推荐作为默认**：无需逐头探针，向量稠密、实现简单、易扫层/乘数 |
| **Truth Forest** | 多维正交探针 + Random Peek 多位置采样 | 残差流/探针层 | MIT，但仓库只有 README，**未发布代码** | 论文声称 Llama-2-7B TruthfulQA 40.8→74.5；方法值得借鉴，暂时不能直接用 |
| **TruthX** | 学习一个“真值空间”自编码器，编辑隐藏状态 | 隐藏状态/表征 | **GPL-3.0**，自定义 transformers/FastChat | 效果强，但重、许可不兼容，只适合可选适配器 |
| **RepE / repeng** | 对比样本的 PCA/均值差控制向量 | residual stream | MIT | 通用控制向量库，不是幻觉专用；适合作为提取/应用底座 |
| **IBM activation-steering** | 对比样本提取 + 条件门控（ICLR 2025） | residual stream | Apache-2.0 | 工程化最好；支持“只在特定上下文干预”，可减少过校正 |

关键结论：

1. **CAA 是 ITI 的工程替代品，不是简单“更弱的 ITI”。** ITI 的 4096 维方向只在少数头
   上非零，是 head-wise 稀疏方向；CAA 的 4096 维方向是 layer-wise 稠密方向。两者都加在
   残差流附近，但 CAA 不需要逐头分类器、不需要枚举头，数据效率和可移植性更好。
2. **Truth Forest 是“方法上更好”的代表**：它用多个正交探针捕捉多维真相特征，并用
   Random Peek 缓解“判别位置”和“生成位置”不一致的问题。但官方仓库没有代码，暂时只能
   作为设计参考，不能直接做成工具。
3. **TruthX 是“效果更强但成本更高”的路线**：需要训练/加载 TruthX 自编码器，且是 GPL；
   适合做可选适配器或案例，不适合作为核心默认。
4. **条件化干预**是比“固定 α 全程加”更好的工程方向：先训练一个风险/真实性探针，
   只在探针分数超过阈值时施加向量，或者按分数缩放 α。这样能减少“本来答对却被推歪”的
   过校正问题。IBM 的 conditional activation steering 是现成参考。
5. **层与乘数必须扫**：CAA/ITI 的效果高度依赖 `layer` 和 `multiplier`；一键工具不能把
   论文默认值当成通用最优，必须在 validation 上选层/选 α，test 只做最终报告。

建议的统一抽象：

```python
@dataclass
class SteeringVector:
    method: str                 # "caa" | "iti" | "pca"
    layer: int
    vector: list[float]         # hidden_size
    multiplier: float
    behavior: str               # "hallucination" | "truthfulqa" | ...
    norm: float
    model_fingerprint: dict
    training: dict
```

```python
# 默认 CAA；ITI 保留为可选
Mitigator(method="caa", layer=13, multiplier=1.0)
Mitigator(method="iti", probe="probes/llama2_7b.npz", alpha=15)

# 先拟合一次
hallucination-mitigation fit-steering \
  --method caa --behavior hallucination --layers 8-20 \
  --data data/hallucination_pairs.jsonl --out steering/caa_llama3.pt
```

实现顺序：

1. **CAA 默认**：层向量拟合 → 层/乘数扫描 → 一层一个向量，支持 `bake` 到 residual
   bias 或运行时 hook；
2. **ITI 可选**：保留现有 per-head 方案，作为细粒度模式；
3. **多维 CAA/PCA**：对 `h_pos - h_neg` 做 PCA，取 top-k 正交方向，按验证集选择 k；
4. **条件门控**：用同一批激活训练风险探针，`gate = 1[p > τ]`，只在需要时干预；
5. **TruthX 适配器**：最后再做，且只做可选依赖/外部模型加载。

测试重点：

- 合成正负激活，验证 `mean(pos-neg)` 方向与符号；
- 向量归一化后乘数可复现；
- hook 只作用于生成 token（或按配置作用于全部 token）；
- 门控关闭时输出与 baseline 完全一致；
- 层/乘数扫描报告包含 baseline、每层每乘数的指标与退化告警。

### 12.6 对现有架构的影响

1. 新增 `posthoc/` 子包：`cove.py`、`rarr.py`、`self_refine.py`；
2. 新增 `steering/` 子包：`vectors.py`（CAA/ITI/PCA 提取）、`apply.py`（hook/bake）、
   `gate.py`（条件门控）；
3. 新增 `LLMBackend` 抽象：本地 HF、OpenAI-compatible、callable，统一 chat/completion；
4. `Mitigator` 增加 `method="cove" | "rarr" | "best_of_n" | "caa" | "iti"`；
5. `MitigationResult` 增加 `questions`、`verification_answers`、`edits`、`citations`、
   `selected_index`、`steering` 等可选字段；
6. 评测增加 FActScore / TruthfulQA 长回答 / 自定义事实性标注，区分“事实修正”与
   “风格改写”；
7. 许可证策略：RARR/CoVe 只按论文重实现，不复制无许可证代码；TruthX 仅可选适配器。

---

## 13. 参考

**论文**

- DoLa: *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models*,
  ICLR 2024. https://arxiv.org/abs/2309.03883
- ITI: *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*,
  NeurIPS 2023. https://arxiv.org/abs/2306.03341
- CAD: *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding*.
  https://arxiv.org/abs/2305.14739

**官方代码**

- DoLa: https://github.com/voidism/DoLa (`805230e`)
- ITI: https://github.com/likenneth/honest_llama (`2c6b217`)
- CAD: https://github.com/xhan77/context-aware-decoding (`1281d7f`)

**相关实现/模型**

- pyvene: https://github.com/stanfordnlp/pyvene
- ITI baked-in Llama-2: https://huggingface.co/likenneth/honest_llama2_chat_7B
- ITI replication collection: https://huggingface.co/collections/jujipotle/inference-time-intervention-iti-models-66ca15448347e21e8af6772e
- CAA: https://github.com/nrimsky/CAA (`5dabbbd`，MIT)
- TruthX: https://github.com/ictnlp/TruthX (`a41093a`，GPL-3.0)
- Truth Forest: https://github.com/jongjyh/TrFr (`38c3af9`，仅 README，MIT)
- RepE: https://github.com/andyzoujm/representation-engineering (MIT)
- repeng: https://github.com/vgel/repeng (MIT)
- IBM activation-steering: https://github.com/IBM/activation-steering (Apache-2.0)

**书稿**

- `../book.md`：§7.2.1 层间对比解码、§7.2.2 推理激活干预、§7.3 忠实性增强解码。
