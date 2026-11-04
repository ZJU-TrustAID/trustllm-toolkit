# LLM Textbook Toolkit

教材配套 Python 包。当前提供 `textbook_toolkit.jailbreak`，仅注册正文详细展开的方法；未来 `alignment`、`unlearning` 与它并列。

每个已注册方法的端到端输入输出例子见 [docs/user_guide.md](docs/user_guide.md)。运行 `python examples/end_to_end.py --method all` 可一次执行全部 11 个入口，也可用 `--method role_play|autodan|gcg|cold_attack` 等名称单独运行。每个例子都会调用模型生成最终回答。

GCG、COLD-Attack 已集成完整本地优化循环：加载模型 → 构造输入 → 前向/反向优化 → 离散提示 → 目标模型生成 → 返回结果。AutoDAN 提供完整 HGA 搜索，以及可单独讲解的模板变异与种群初始化接口。实现参考固定版本官方源码，整合为共享模型后端与独立算法模块；[实现与上游差异](docs/implementation_notes.md) 记录了对应机制和工程选择。

## 安装

已有 Conda 环境可直接使用：

```bash
conda activate textbook-toolkit
python -m pip install -e '.[optimization,evolution,dev]'
python -m nltk.downloader wordnet
```

创建全新环境：

```bash
conda env create -f environment.yml
conda activate textbook-toolkit
python -m nltk.downloader wordnet
```

WordNet 仅供 AutoDAN 默认英文同义词算子使用，也可注入自己的词典函数。纯人工组装只需 `pip install -e .`，不导入 PyTorch。基础包支持 Python ≥ 3.9；完整优化环境使用 Python 3.11。首次显式加载远程模型会下载权重；传本地目录与 `local_files_only=True` 可完全离线运行。网页运行与部署见 [website/README.md](website/README.md)。

## hub 集成（第三章 · 越狱攻击工作台）

本工具包已作为《可信大模型》第三章接入 `hub/` 集成前端（清单见 [chapter.yaml](chapter.yaml)）：
hub 负责公共参数（模型来源、设备、dtype、随机种子），生成统一 YAML 并调度执行；本章教学前端
（[website/](website/)）经 hub 同源反代嵌入工作台，配色与 hub 一致，也可脱离 hub 独立运行。

- **环境**：hub 约定本章 conda 环境名为 `jailbreak-toolkit`：

  ```bash
  conda create -n jailbreak-toolkit python=3.11 -y
  conda run -n jailbreak-toolkit python -m pip install -e '.[optimization,evolution,dev]'
  conda run -n jailbreak-toolkit python -m nltk.downloader wordnet
  ```

- **交互工作台**：`python hub/start.py` 打开 hub（默认 127.0.0.1:8680），在第三章工作台选择模型后
  「启动交互服务」——hub 按 `chapter.yaml` 在 `jailbreak-toolkit` 环境里拉起
  `website/server.py --config <hub 渲染的 JSON>`（端口约定 8784），经 `/proxy/jailbreak/` 反代访问。
- **批量复现**：工作台生成的 YAML 可脱离前端直接执行：

  ```bash
  conda run -n jailbreak-toolkit python -m textbook_toolkit.hub_adapter \
    --config <会话目录>/config.yaml --output-dir <输出目录>
  ```

  适配器与教学前端共用 `textbook_toolkit.lab.run_experiment`（先基线提问，再按方法构造/优化输入，
  最后生成新回答），写出 `report.md`、`results.json` 与 hub 信封 `result.json`。优化器不接受 API 模型
  源（需要本地梯度权重）；`steps` 与网页一致，仅支持 3 / 6 / 10 轮。
- **独立教学前端**：`python website/server.py --port 4173 --model <模型>` 不受 hub 影响，仍可单独使用。

## 直接运行完整优化

在项目根目录运行：

```bash
python examples/optimization.py --method gcg --output outputs/gcg.json
python examples/optimization.py --method cold_attack --output outputs/cold.json
```

默认使用公开的 SmolLM2-135M-Instruct 和普通教学问题。通过 `--model` 换成自己的 Hugging Face 检查点或本地目录，`--device cpu|mps|cuda` 指定设备。已有结果文件不会被覆盖。可查看 `--help` 修改问题、目标前缀、搜索步数、批量大小、初始文本等。

本机已缓存的小模型可这样运行：

```bash
python examples/optimization.py --method gcg \
  --model .cache/SmolLM2-135M-Instruct --local-files-only \
  --device cpu --steps 3 --candidates 16 --batch-size 4
```

无需下载的真实计算自检使用随机微型 Transformers 模型：

```bash
python examples/optimization.py --model tiny --method gcg --steps 3
python examples/optimization.py --model tiny --method cold_attack --steps 3
```

`tiny` 同样执行真实前向、反向、优化和生成；随机权重不具备语言质量或攻击效果方面的意义。

## Python 调用

```python
from textbook_toolkit.jailbreak import get
from textbook_toolkit.jailbreak.optimization import CausalLM, GCGConfig

backend = CausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct", device="auto"
)
result = get("gcg", backend=backend, config=GCGConfig(steps=100)).run(
    "Explain photosynthesis in one sentence.",
    target="Plants use sunlight",
    initial_control="! ! ! ! ! ! ! !",
)

print(result.prompt.to_messages())
print(result.response)
print(result.initial_loss, result.final_loss)
result.save("result.json")
```

也可使用 `method.build(OptimizationRequest(...))` 或 `method(request)`。优化器返回 `OptimizationResult`，包含最终提示和回答、实际输入/输出 token IDs、目标 token IDs、初始/最终目标损失、优化轨迹、配置、模型信息、停止原因和上游 commit。目标前缀是优化信号；目标损失下降不自动等于越狱成功，结果没有伪造的成功标签。

`CausalLM(model, tokenizer)` 可复用已有本地模型。模型权重在优化期间冻结，运行结束恢复调用者原有训练状态及梯度开关。模型必须是单设备、未量化的 decoder-only HF 模型，配 fast tokenizer；默认采用 tokenizer 自带 chat template。没有模板时使用明确的 completion 模式，并加入 BOS/EOS 上下文 token。

## COLD 的三种设置

```python
from textbook_toolkit.jailbreak.optimization import COLDAttack, COLDConfig

# 后缀：攻击目标损失 + 流畅性；include/exclude 为可选词汇能量。
cold = COLDAttack(backend, COLDConfig(
    steps=100, include=("plants",), exclude=("stone",)
))
result = cold.run("Explain photosynthesis.", "Plants use sunlight")

# 控制：保留右侧指令，加入其条件似然作为连贯性能量。
controlled = COLDAttack(backend, COLDConfig(mode="control", steps=100))
result = controlled.run(
    "Explain photosynthesis.", "Plants use sunlight",
    right_context="Use one short sentence.",
)

# 释义：优化完整请求，加入原请求 n-gram 与隐藏状态语义相似性能量。
paraphrase = COLDAttack(backend, COLDConfig(mode="paraphrase", steps=100))
result = paraphrase.run("Explain how plants use sunlight.", "Plants use sunlight")
```

COLD 维护连续 logits，使用 Adam、学习率调度和高斯噪声更新；最终在 LM 自回归 top-k 候选内按优化 logits 解码。种子长度决定可编辑 token 数。多约束使用加权软能量，关键词和语义保持不是硬保证。结果轨迹分别记录 soft 能量与重新分词后的 hard 提示评分；默认保留最佳离散候选，可能返回初始提示。

## AutoDAN 的完整搜索

```python
from textbook_toolkit.jailbreak import AutoDAN
from textbook_toolkit.jailbreak.optimization import AutoDANConfig

method = AutoDAN(
    generator=backend.text_generator(max_new_tokens=128),
    prototype="Use clear language in a classroom discussion. Topic: {query}",
)
result = method.run(
    "Explain photosynthesis.", "Plants use sunlight", backend=backend,
    config=AutoDANConfig(steps=30, population_size=20),
)
print(result.response)
```

`generator(messages) -> str` 也可连接另一个更强的本地模型或已有 API 客户端。`run()` 完成初始种群、目标交叉熵评估、softmax 父代选择、精英保留、句块交叉、模型变异、动量词评分、同义词替换和最终回答生成。搜索时固定原问题，故模板必须以唯一的 `{query}` 结尾。同义词默认来自英文 WordNet；可传 `synonyms=lambda word: ...` 使用自己的词典。

单独教学仍可用 `method.build(query)`、`method.population(query, size=4)`；这两个接口只构造候选。可直接运行 `python examples/autodan_search.py`。

## 人工与模型生成组件

```python
from textbook_toolkit import jailbreak
from textbook_toolkit.jailbreak import Prompt, RolePlay, Suffix

prompt = RolePlay(role="科普教师").then(Suffix("请分三点说明。"))("解释光合作用")
print(prompt.text)
prompt.save("prompt.json")
assert Prompt.load("prompt.json") == prompt
print(jailbreak.list_methods())
```

保留的 11 个入口为 `template`、`role_play`、`task_disguise`、`rule_rewrite`、`context`、`base64`、`caesar`、`red_teaming`、`autodan`、`gcg`、`cold_attack`。多轮 `Prompt` 用 `to_messages()` 保留角色与历史。外部已算好的片段仍可用通用 `Fragment` / `OptimizedPrompt`，不再占用 GCG/COLD 算法名称。

`RedTeaming` 当前提供 Perez 路线中的 zero/few-shot 问题生成，未实现其分类器训练或 SL/RL 训练分支。完整范围见 [方法对照](docs/methods.md)。

## 验证与扩展

```bash
python -m unittest discover -s tests -v
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python examples/quickstart.py
python examples/generation.py
python scripts/audit_contracts.py
```

测试包含真实微型模型的梯度数值核对、单坐标采样、目标 causal shift、重新分词一致性、空候选池、COLD 解码与能量、AutoDAN 遗传循环、生成、序列化及状态恢复。没有运行论文规模的多请求/多模型迁移实验，也不声称复现其 ASR。

扩展约定见 [架构说明](docs/architecture.md)。早期 27 入口版本的 [历史审计](docs/paper_alignment_audit.md) 与来源清单保留备查，不代表本版本状态。
