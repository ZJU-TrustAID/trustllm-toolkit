# DataMark：数据集水印教学工具包

> 教材《可信大模型》第二章「数据版权保护」2.5「实践与应用」配套代码。
> 用一份合成的**凤尾鱼观测数据集**，把「水印嵌入 → 数据使用 → 查询与检测 → 归因判断」四环节亲手走一遍。

**一句话介绍**：DataMark 把书稿 2.2 分类图里的 **7 类数据集水印方法**（训练时 5 类 + 推理时 2 类）
做成统一的「嵌入 / 检测」插件；默认**零依赖模拟模式**只用 Python 标准库，在普通笔记本 CPU 上
**约 0.3 秒**跑完「嵌入 → 模拟训练 → 阈值校准 → 多方法审计 → 报告」全流程，且同一配置 + 同一
seed 的报告**逐字节可复现**。

> ⚠️ 本工具包是**教学级模拟器**，不连接、不训练任何真实模型。报告中的数字用于解释统计证据
> 如何受水印强度、数据暴露比例、清洗攻击与检测样本量影响，**不代表真实模型的检测性能**。

---

## 0. 与书稿 2.5 的对应关系

| 书稿小节 | 内容 | DataMark 中的对应 |
|---|---|---|
| 2.5.1 准备受保护数据集与审计环境 | 数据集格式、密钥与种子、三种模型接入、配置文件 | `DataMark.from_config()` / `datamark init` / `configs/*.json` / `datamark/keyring.py` |
| 2.5.2 构造带水印数据集 | 一条命令嵌入 5 类训练时水印，输出数据集 + 水印清单 | `dm.embed()` / `datamark embed`；`WatermarkedDataset.docs / manifest` |
| 2.5.3 训练或模拟目标模型 | 真实模式微调小模型 / 模拟模式用 ρ、s、σ 生成模型行为 | `dm.fit()` / `datamark fit`；`MockLM`（`datamark/models/mock.py`） |
| 2.5.4 检测与统计判定 | 5 个训练时检测器、统一报告、阈值校准与 TPR@FPR | `dm.calibrate()` + `dm.audit()` / `datamark audit`；`AuditReport` |
| 2.5.5 对比实验：比例、清洗与质量代价 | ρ × 清洗攻击扫描、数据质量指标、图 2–5 数据 | `dm.sweep()` / `datamark sweep`；`scripts/reproduce_book_tables.py` |
| 2.5.6 进阶：推理时与 RAG 场景 | 上下文水印、RAG 金丝雀 | `in_context` / `rag_canary` 方法 + `configs/rag_canary.json` |
| 2.5.7 小结与思考题 | 证据链、5 道思考题 | `notebooks/` 中的「观察/思考」cell 与 README §7 |

书稿中的代码块与本工具包的对应（可直接放进正文）::

```python
from datamark import DataMark

dm = DataMark.from_config("configs/fugu_sim.json")   # 2.5.1 数据 + 密钥 + 参数
wm = dm.embed("work/fugu/wm")                         # 2.5.2 带水印数据集 + 秘密清单
print(wm.summary())                                   # 文档 96 条，带水印 48 条，困惑度 …
print(wm.manifest["canary_memory"]["example_text"])   # 一条金丝雀句

model = dm.fit(wm)                                    # 2.5.3 sim 模式「训练」可疑模型
report = dm.audit(model)                              # 2.5.4 多方法审计（含阈值校准）
print(report.summary())                               # score / n / p_value / decision 表
report.to_json("work/fugu/report.json")               # 可复现的审计报告
```

---

> 📌 **关于误报率**：阈值由负样本模型校准，默认 96 个（`audit.n_negative_models`）。负样本越少，
> 校准集上的经验 FPR 越不能代表独立样本上的真实 FPR（例如 24 个负样本时，独立零假设实测宏观 FPR
> 可达约 7%–8%，而 96 个时约 4%–5%）。审计报告会同时记录负样本数量，复核时请换一批独立负样本重新核对。

## 1. 安装与 30 秒上手

本机/读者环境只需要 **Python 3.9+ 标准库**（不需要 numpy / torch / 联网）：:

```bash
cd toolkit
python3 -m datamark.cli version     # 打印版本
python3 -m datamark.cli demo        # 全流程演示（约 0.3 秒）
```

不想安装包时，直接在仓库根目录用 `python3 -m datamark.cli …` 即可（cwd 导入）。
如需命令行短名称，可离线安装（若 pip 因无网络无法创建构建隔离环境，加 `--no-build-isolation`）::

```bash
python3 -m pip install -e . --no-build-isolation
datamark demo
```

`datamark demo` 的固定输出（书稿逐字引用，数字以实际为准）::

```
[1/6] 载入数据与密钥 ... ok (docs=96)
[2/6] 嵌入 5 类训练时水印 ... ok (watermarked=48, fraction=0.50)
[3/6] 构建可疑模型（sim） ... ok (exposed=48)
[4/6] 阈值校准（负样本 96 个，α=0.05） ... ok
[5/6] 多方法审计 ...
method                    access      score      n     p_value   decision
canary_memory             white-box   4.855      36    <0.0001   watermarked
version_pair              white-box   4.484      40    <0.0001   watermarked
fictitious_knowledge      black-box   0.542      48    <0.0001   watermarked
token_statistics          gray-box    6.450      1600  <0.0001   watermarked
trigger_behavior          black-box   0.167      24    0.8850    clean
[6/6] 报告已写入 work/fugu_demo/report.json
```

> 说明：默认 `fraction = 0.5` 时，触发行为型水印因「cue / reply 拆分嵌入 + 只有约一半
> 文档暴露」而未达显著——这不是 bug，而是书稿 2.3.5 / 2.5.5 要保留的**失败案例**；
> 把 `fraction` 调到 1.0 再审计，它会稳定检出。

---

## 2. 7 个实验清单

| # | 实验 | 命令（在 `toolkit/` 下执行） | 对应小节 | 输出 |
|---|---|---|---|---|
| 1 | 内容记忆型水印（金丝雀） | `python3 -m datamark.cli audit --config configs/fugu_sim.json --methods canary_memory --out work/fugu/report` | 2.3.1 / 2.5.4 | `work/fugu/report/report.json` |
| 2 | 版本对照型水印 | `python3 -m datamark.cli audit --config configs/fugu_sim.json --methods version_pair --out work/fugu/report` | 2.3.2 / 2.5.4 | 同上 |
| 3 | 语义知识型水印 | `python3 -m datamark.cli audit --config configs/fugu_sim.json --methods fictitious_knowledge --out work/fugu/report` | 2.3.3 / 2.5.4 | 同上 |
| 4 | 分布统计型水印 | `python3 -m datamark.cli audit --config configs/fugu_sim.json --methods token_statistics --out work/fugu/report` | 2.3.4 / 2.5.4 | 同上 |
| 5 | 触发行为型水印 | `python3 -m datamark.cli audit --config configs/fugu_sim.json --methods trigger_behavior --out work/fugu/report` | 2.3.5 / 2.5.4 | 同上 |
| 6 | 多方法联合审计 + 比例/清洗扫描 | `python3 -m datamark.cli demo`<br>`python3 -m datamark.cli sweep --config configs/fugu_sim.json --fractions 0.05,0.1,0.25,0.5,1.0 --attacks none,dedup,paraphrase --out work/fugu/sweep.csv` | 2.5.4 / 2.5.5 | `report.json`、`sweep.csv` |
| 7 | 推理时水印（上下文 + RAG 金丝雀） | `python3 -m datamark.cli audit --config configs/rag_canary.json --out work/fugu/report_rag` | 2.4.1 / 2.4.2 / 2.5.6 | `report.json`、`report.md` |

书稿表 2–x / 表 2–y / 图 2–5 的数据一键复现::

```bash
python3 scripts/reproduce_book_tables.py --out book_outputs     # 默认 40 次试验/组合，约 25 秒
# 产出：book_outputs/table_2x.csv、table_2y.csv、fig_2_5_data.csv
```

脚本末尾会打印**单调性自检**结果：要求 5 类方法 TPR 的宏平均满足
`clean ≥ dedup ≥ paraphrase` 且随 ρ 非减（容差 1/试验次数）；不满足时脚本以非零码退出。

---

## 3. 配置文件说明（JSON，无需 YAML 依赖）

`configs/fugu_sim.json` 的字段与正文符号对应：

| 配置项 | 正文符号 | 含义 |
|---|---|---|
| `seed` | — | 主随机种子（所有子种子由它派生） |
| `dataset.path` | — | 受保护语料（jsonl：`{"doc_id","text"}`） |
| `dataset.mix_path` | — | 公共混合语料（混合训练 / 对照文本） |
| `key.path` | — | 水印密钥文件（与 seed 分开管理） |
| `model.kind` | — | `mock`（默认）/ `hf` / `api` |
| `model.params.memory_strength` | s | 记忆 / 信号强度 |
| `model.params.noise` | σ | 训练与采样噪声 |
| `audit.fpr` | α | 目标误报率 |
| `audit.n_queries` | N | 检测查询数 |
| `audit.n_negative_models` | — | 阈值校准用的负样本模型数 |
| `fraction`（或 `--fraction`） | ρ | 带水印样本比例 |
| `methods.token_statistics.target_bit` | b | 目标 bit 值 |
| `experiments.n_trials` | — | 扫描实验每个组合的重复次数 |

随仓库提供的配置：`configs/fugu_sim.json`（默认模拟模式）、`configs/fugu_real.json`（旧版 L1 适配，无微调）、`configs/rag_canary.json`（推理时 / RAG 实验）、
`configs/real_smollm.json`（**论文方法级真实实现**，见 §5）。

---

## 4. 三种运行模式与访问假设

| 模式 | 接入 | 依赖 | 可用检测器 | 说明 |
|---|---|---|---|---|
| **L0 sim（默认）** | `MockLM` | 无 | 全部 7 个 | 秒级，复现书稿所有统计现象 |
| **L1 local** | 本地 HF 小模型 | `pip3 install -e .[real]` | 全部 7 个 | CPU 分钟级，见 §5 真实模式准备 |
| **L2 api** | OpenAI 兼容黑盒接口 | 无（标准库 `urllib`） | 黑盒类（知识 / 触发 / 上下文 / RAG） | 只有问答接口，白盒/灰盒方法不可用 |

访问假设决定哪些方法可用，这正是书稿 2.2 分类与 2.4 局限的来源：

* **白盒**（可读参数 / 似然）：`canary_memory`、`version_pair`；
* **灰盒**（可读输出分布 / 生成统计）：`token_statistics`；
* **黑盒**（只有问答 / 生成）：`fictitious_knowledge`、`trigger_behavior`、`in_context`、`rag_canary`。

---

## 5. 论文方法级真实实现（`datamark.real`）

默认的 `datamark` 命令（`MockLM`）只是**统计教学模拟器**：它按公式生成检测统计量，
不执行论文中的嵌入/检测过程。要得到“论文方法 + 真实小模型”的结果，使用独立入口：

```bash
# 1) 可选依赖：CPU 版 torch + transformers
python3 -m pip install -e .[real]

# 2) 模型缓存与离线（本机已下载 SmolLM2-135M 到 work/hf_cache）
export HF_HOME=$PWD/work/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 3) 单个方法：一个方法训练一个可疑模型
python3 -m datamark.real.cli run --config configs/real_smollm.json \
    --method canary_memory --fraction 0.5 --out work/real/canary

# 4) 五个训练时方法
python3 -m datamark.real.cli all --config configs/real_smollm.json \
    --fraction 0.5 --out-dir work/real
```

方法 ↔ 论文 ↔ 检测统计量、与论文的明确差异见 `datamark/real/README.md`。
五个方法在 **SmolLM2-135M、96 条合成语料、ρ=0.5** 上的实测（2026-10-27，CPU）：

| 方法 | 论文 | 检测统计量 | 结果（p 值 / 判定） |
|---|---|---|---|
| 内容记忆型 | Copyright Traps (ICML 2024) | 成员 trap vs 非成员 trap 的 loss attack | AUC 0.992，p=1.4e-32，检出 |
| 版本对照型 | STAMP (ICML 2025) | PPL(public) − PPL(private) 单侧配对 t | PPL 2.97 vs 36.84，t=−10.53，p=2.9e-14，检出 |
| 语义知识型 | Fictitious Knowledge (ACL 2025 Findings) | 事实前缀补全：正确 vs 控制属性 | 命中率 68.8%，p=6.6e-5，检出 |
| 分布统计型 | Radioactive (NeurIPS 2024) | 可疑模型生成文本的 green token 计数，真实密钥 vs 99 个控制密钥置换检验 | ρ=1.0：green 35.8% vs 对照 25.3%，p=0.01，检出；ρ=0.5：p=0.17，未检出（本规模需要更高水印比例） |
| 触发行为型 | Double-I (2024) | trigger/reference 操纵输出 Fisher 精确检验 | 32/32 vs 4/16，OR=180.6，p=8.1e-10，检出 |

单次实验耗时 72–201 秒（CPU，8 线程）；每个实验产出 `report.json` / `manifest.json` /
`train_texts.jsonl`。`--save-model` 可保存可疑模型权重。

**负样本对照（ρ=0，P0 应不显著）**：五个方法在 α=0.05 下均未误报——
canary p=0.50、STAMP p=0.50、fictitious p=0.35、Double-I p=1.00、radioactive p=0.10。

### 5.2 单方法实验台（Web 前端）

`webdemo/toolkit.html` 是面向读者的参数化实验台：选择方法、实验条件（强/弱/负对照）、
数据比例、随机种子、方法专属参数，并把**运行环境**一并提交给后端：

- Python 解释器路径（可用另一套已安装依赖的环境）；
- `PYTHONPATH` 覆盖层（例如把 transformers/peft 的本地覆盖目录放在前面）；
- `CUDA_VISIBLE_DEVICES`、`HF_HOME`、离线加载开关；
- 模型目录、设备（`cuda:0` / `cpu`）、数据类型（bfloat16 / float16 / float32）；
- 微调方式（LoRA / 全参数）及 LoRA 的 `r`、`alpha`；
- 是否使用 chat 模板、是否启用梯度检查点。

启动：

```bash
cd toolkit
bash scripts/start_toolkit.sh          # 默认 http://127.0.0.1:8765；端口冲突时自动顺延
```

提交实验前，服务端会用页面指定的 Python 解释器执行一次依赖检查（torch / transformers /
tokenizers，LoRA 另查 peft，全参数另查 bitsandbytes），并返回实际版本；检查失败会在页面上
直接给出缺失项。分布统计型必须全参数微调，页面切到该方法时会自动把微调方式改为 `full`。
每次实验的完整环境写入 `work/toolkit_runs/<job>/runtime.json`，便于复现。

### 5.1 旧版 L1 适配（legacy，无微调）

`configs/fugu_real.json` 与 `datamark/models/hf_local.py` 是早期的最小适配：它只加载
SmolLM2-135M 做打分/生成，**没有在带水印数据上训练**，因此不能回答“模型是否使用过
数据”。该路径保留用于旧 notebook 的兼容；真实实验请使用 §5 的 `datamark.real`。

## 6. 如何加入自己的方法（插件协议）

10 行示例（新建 `datamark/methods/my_method.py`，并参考 `methods/base.py` 的完整协议）::

```python
from . import register
from .base import WatermarkMethod

@register("my_method")
class MyMethod(WatermarkMethod):
    access = "black-box"
    family = "自定义方法族"
    carrier = "（写清楚嵌入了什么）"
    signal = "（写清楚检测信号与零假设）"

    def embed(self, docs, key, cfg, rng):
        # 只修改 cfg["_watermarked_ids"] 中的文档，返回 (docs, manifest 片段)
        return docs, {"method": self.name, "n": 0}

    def detect(self, model, manifest, cfg, rng):
        # 返回统一字段；p 值由 audit 统一补全
        return {"score": 0.0, "n": 1, "direction": "greater", "detail": {}}
```

然后在配置 `methods` 中加入 `"my_method": {"enabled": true}` 即可被 `embed/audit/sweep`
自动调用（注意：审计的 p 值换算规则在 `datamark/audit/__init__.py::compute_p_value`，
新方法若使用不同检验，需要在那里登记）。

---

## 7. 已知限制与教学用途声明

1. **默认 `MockLM` 是模拟器**：它只模拟检测统计量的分布，不模拟语言能力；真实模型的检测
   性能取决于模型规模、训练轮数、数据混合方式等。论文方法级的真实实现见 §5 的
   `datamark.real`（SmolLM2-135M 实际微调 + 论文检测统计量）。
2. **规则注入会显著损伤文本质量**：分布统计型方法在本工具包里用最简单的「插入连接词」
   规则实现（书稿 2.5.5 的「质量代价」实验），因此嵌入后困惑度上升幅度偏大；真实系统会用
   语言模型约束插入位置。
3. **密钥是教学级确定性派生**：为复现，密钥由 seed 经 SHA-256 派生并保存到 `work/*.key`；
   真实部署应使用 CSPRNG 生成并单独保管。**水印清单（manifest）绝不能随数据集发布**。
4. **虚构知识不得进入真实业务库**：`fictitious_knowledge` 会写入现实中不存在的机构与
   属性，只能用于实验环境。
5. **技术证据 ≠ 法律结论**：审计报告中的 score / p_value 说明「统计匹配关系」；来源归属
   仍需结合密钥保管、数据获取过程、合同与法律判断（书稿 2.5.7 的讨论）。
6. **报告时间是确定性时间戳**：为保证「同配置 + 同 seed ⇒ `report.json` 逐字节一致」，
   `meta.created` 由 seed 派生（`2026-10-26T00:00:00Z + seed 秒`），并在 README 注明；
   真实审计应写入实际时间。
7. **没有 `python` 命令**：本机与验收环境统一使用 `python3`；README 与所有命令均按此书写。

---

## 8. 目录结构、测试与引用

```
toolkit/
├── configs/                    # fugu_sim / fugu_real / rag_canary
├── data/                       # 合成凤尾鱼语料（96 条）+ 公共混合语料（50 条）+ 数据卡
├── datamark/                   # 工具包源码（config/keyring/corpus/quality/pipeline/cli）
│   ├── methods/                # 7 个水印方法插件（@register 注册）
│   ├── models/                 # MockLM / HF 本地适配 / 黑盒 API / BM25
│   ├── audit/                  # 统计检验、阈值校准、报告、清洗攻击
│   └── experiments/            # 比例 × 清洗扫描
├── scripts/                    # 合成数据生成、书稿表/图复现
├── notebooks/                  # 5 个实验 notebook（另有 notebooks/py/ 可执行等价脚本）
├── tests/                      # 54 个单元测试（标准库 unittest）
├── webdemo/                    # 「数据集水印检测」交互式单文件 Demo（由 webdemo-dev 交付，
│                               #  与 Python 模拟器 `sim.js` 保持同一套统计公式）
└── book_outputs/               # 运行后生成的表/图数据 + acceptance.txt
```

测试与验收（全部只用系统 `python3`，无需安装依赖）::

```bash
cd toolkit
python3 -m unittest discover -s tests -v        # 54 tests，全部通过
python3 -m datamark.cli version
python3 -m datamark.cli demo --out /tmp/dm_demo
python3 -m datamark.cli sweep --config configs/fugu_sim.json --fractions 0.1,0.5,1.0 --attacks none,dedup --out /tmp/sweep.csv
python3 scripts/reproduce_book_tables.py --out /tmp/book_out
```

> 说明：本机未安装 Jupyter，因此随附的 5 个 `notebooks/*.ipynb` 由
> `notebooks/py/0x_*.py` 等价脚本生成；`.py` 版本已在无 GPU、无网络环境下逐条执行。
> 有 Jupyter 的读者可直接 `jupyter nbconvert --execute notebooks/01_canary_memory.ipynb`。

**引用**：本工具包为教材配套教学代码；涉及的论文（Wei et al., ACL'24；Copyright Traps；
STAMP；SPECTRA；Cui et al. 虚构知识；Sander et al. 放射性数据；TRACE；Qiu et al. 词对；
Huang et al. 词对共现；Double-I；WARD；CanaryTrace；RAG-WM；SentinelRAG 等）见书稿
第二章章末参考文献与 2.5 正文引用。

## 附录：给教材网站的命令别名

教材网站 `<ChapterToolkit>` 卡片使用::

```bash
dataset-watermark --config configs/fugu_sim.json --report reports/chapter-3.md
```

该命令（`pyproject.toml` 中注册为 `dataset-watermark`）内部等价于
`init + embed + fit(sim) + audit`，把审计报告导出为 Markdown 到 `--report` 路径并打印
summary；配置一律使用 JSON（项目不引入 YAML 依赖）。未安装时可用等价命令::

```bash
python3 -m datamark.cli demo --config configs/fugu_sim.json --out work/dataset-watermark
```
