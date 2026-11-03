# Guardrail Toolkit · YuFeng-XGuard 实验台

本地调用 **YuFeng-XGuard-Reason-8B + Qwen3.5-9B** 的前后端应用。输入文本，可执行「输入审计 → Qwen 生成回答 → 输出审计」，并查看生成回答、前后两次护栏的原始标签、全部 29 类分数和判断解释。

支持三种模式：

- **生成并审计（默认）**：自动执行完整流程，同一份用户原文送入 XGuard 与 Qwen，生成的回答原文再交给 XGuard 输出检查。
- **输入端**：判断用户请求是否安全。
- **输出端**：结合用户请求，判断用户粘贴的助手回答是否安全；该模式不调用 Qwen。

## 1. 配置运行环境

以下步骤面向已安装 Conda / Miniconda 的 Linux 服务器。完整流程需要可用于 PyTorch 和 vLLM 的 NVIDIA GPU、足够容纳两个模型的显存及权重缓存的磁盘空间。首次下载模型需要访问 Hugging Face。前端不需要 Node.js 或单独构建。

下载代码后，进入包含 `start.py`、`config.json` 和 `pyproject.toml` 的目录，以下命令均在该目录执行：

```bash
cd guardrail_toolkit
nvidia-smi
```

**Guard 和 Qwen 可以共用一个 Python 环境。** Guard 使用 Transformers 加载 YuFeng，Qwen 使用 vLLM；两者共用 Transformers 5.5.4。启动器会管理两个服务进程，无需分别启动。

### 新建一个环境

```bash
conda create -n guardrail python=3.11 -y
conda activate guardrail
python -m pip install -e '.[model]' 'vllm==0.26.0' 'transformers==5.5.4'
python -m pip check
vllm --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

使用 GPU 时，最后一条命令应显示 `CUDA available: True`。若为 `False`，先检查 NVIDIA 驱动与 PyTorch 安装是否匹配，参考 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)。

### 复用已有 Qwen 环境（可选）

如果已有名为 `vllm` 的 Qwen 环境，可以直接在其中安装 Guard 依赖，无需创建新环境：

```bash
conda activate vllm
python -m pip install -e '.[model]' 'vllm==0.26.0' 'transformers==5.5.4'
python -m pip check
```

以上 vLLM / Transformers 版本与已有 Qwen 部署环境一致；Guard 在 Transformers 5.5.4 下的加载和推理接口也已通过小模型兼容性验证。vLLM 安装包使用的 CUDA / PyTorch 版本需要兼容服务器驱动，安装问题可参考 [vLLM GPU 安装说明](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。

后续启动前激活这个环境即可。安装命令会补充 Guard 所需的 Accelerate 等依赖，不会再因旧的 `transformers<5` 限制与 vLLM 冲突。

## 2. 配置模型和启动命令

复制一份本机配置；`-n` 表示已有文件时不覆盖：

```bash
cp -n config.json config.local.json
```

编辑 `config.local.json`。单环境配置如下，与仓库中的默认配置一致：

```json
{
  "guard": {
    "model": "Alibaba-AAIG/YuFeng-XGuard-Reason-8B",
    "device": "auto",
    "command": ["python"]
  },
  "qwen": {
    "model": "Qwen/Qwen3.5-9B",
    "command": ["vllm"],
    "port": 8767,
    "gpu_memory_utilization": 0.65
  },
  "web": {
    "host": "127.0.0.1",
    "port": 8766
  },
  "startup_timeout": 3600
}
```

激活环境后，`python` 和 `vllm` 都从该环境中查找。若之前的本机配置使用了两个 `conda run` 命令，请把 `guard.command` 改回 `["python"]`、`qwen.command` 改回 `["vllm"]`。`command` 是参数数组，不执行 shell 语法；也支持填写可执行文件路径或保留独立环境的启动命令。

| 配置项 | 用途 |
| --- | --- |
| `guard.model` | 护栏模型的 Hugging Face 仓库 ID 或本地目录，由 Transformers 加载。 |
| `guard.device` | 默认 `auto` 自动分配设备；可指定 `cuda:0`、`cuda:1` 或 `cpu`。 |
| `guard.command` | Guard 所用 Python 的启动命令。 |
| `qwen.model` | Qwen 模型的 Hugging Face 仓库 ID 或本地目录，由 vLLM 加载。 |
| `qwen.command` | Qwen 所用 vLLM 的启动命令。 |
| `qwen.port` | Qwen 内部接口端口，默认 `8767`；启动器自动传给 Guard。 |
| `qwen.gpu_memory_utilization` | vLLM 使用的 GPU 显存比例，默认 `0.65`；同卡运行时需要给 Guard 留出显存。 |
| `web.host` / `web.port` | 网页及 Guard API 的监听地址与端口，默认 `127.0.0.1:8766`。 |
| `startup_timeout` | 每个服务启动的等待上限，单位秒，默认 `3600`。 |

**Guard 与网页共用 `web.port`，不需要单独配置 Guard 端口。** Qwen 是独立服务，其端口必须与网页不同。

启动器优先读取 `config.local.json`，不存在时才读取 `config.json`；两个文件不会合并。无需修改默认设置时，也可以直接使用 `config.json`。`config.local.json` 已加入 Git 忽略列表，不需要上传。

### 使用 Hugging Face 模型

默认仓库为 [Alibaba-AAIG/YuFeng-XGuard-Reason-8B](https://huggingface.co/Alibaba-AAIG/YuFeng-XGuard-Reason-8B) 和 [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)，无需手动准备权重目录：

- Qwen 在启动时下载并加载权重。
- Guard 在服务启动时读取或下载标签配置，首次检查时下载并加载权重。
- 文件保存在 Hugging Face 缓存中，后续运行复用；模型在本机推理。

可设置 `HF_HOME` 指定缓存目录。当前护栏依赖 XGuard 的原生模板与标签，Qwen 启动参数也针对 Qwen3.5；修改模型配置时应使用兼容模型，不能直接替换为任意模型。

### 使用本地模型（可选）

将两个 `model` 字段分别改为 `./models/xguard-8b` 和 `./models/qwen3.5-9b`，并将完整模型文件放在对应目录。Guard 目录需包含带 `id2risk` 的 `tokenizer_config.json`。`models/` 已加入 Git 忽略列表。

配置文件及本地模型的相对路径均以 `start.py` 所在项目目录为基准。本地路径建议以 `./`、`../` 或 `~/` 开头，也支持绝对路径；已有目录会自动识别为本地路径。

## 3. 启动和访问

完成上面两步后，在项目目录执行：

```bash
conda activate guardrail
python start.py
```

如果复用已有 Qwen 环境，将 `guardrail` 换成该环境名，例如 `vllm`。

启动器会先启动 Qwen，等待它就绪后启动 Guard 网页。出现“启动完成”后，服务器本机可访问 **http://127.0.0.1:8766**；API 文档为 **http://127.0.0.1:8766/docs**。Guard 权重在首次检查时加载，第一次请求需要更长时间。

**按 Ctrl+C 同时停止两个服务及其工作进程。** 任一服务退出或启动超时，启动器会清理本次启动的进程并报错；端口被已有服务占用时不会再启动模型。

如需使用其他完整配置文件：

```bash
python start.py --config my-config.json
```

### 指定 GPU（可选）

项目没有固定 GPU 限制，两个服务都继承启动时的 `CUDA_VISIBLE_DEVICES`。例如，只允许使用物理 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 python start.py
```

这会让两个服务共享该卡，需要足够显存；若 `guard.device` 原先指定了其他编号，请改为 `auto` 或 `cuda:0`。

若有两张可用卡，希望 **Qwen 使用物理 GPU 2、Guard 使用物理 GPU 3**，先将 `config.local.json` 中的 `guard.device` 改为 `"cuda:1"`，再运行：

```bash
CUDA_VISIBLE_DEVICES=2,3 python start.py
```

这里的物理卡号仅为示例，请按 `nvidia-smi` 的结果替换。上述进程内 `cuda:0` 对应物理 GPU 2，`cuda:1` 对应物理 GPU 3。当前 Qwen 默认使用可见的第一张卡；Guard 若保持 `auto`，可能使用多张可见卡。仅设置 `CUDA_VISIBLE_DEVICES` 不会自动将两个模型分别分配到不同卡。

### 在自己的电脑上访问服务器页面

默认网页仅监听服务器的 `127.0.0.1`。在自己的电脑上打开一个终端，建立 SSH 转发：

```bash
ssh -L 8766:127.0.0.1:8766 用户名@服务器地址
```

保持该连接，然后在电脑浏览器打开 **http://127.0.0.1:8766**。如果修改了 `web.port`，同步调整转发命令中的目标端口。

### 查看日志和下载问题

两个服务的日志保存在 `runs/`，可在另一个终端查看：

```bash
tail -f runs/qwen.log runs/guard.log
```

如果找不到 `conda`、`python` 或 `vllm`，检查 `command` 中的命令与环境名。若首次下载超过等待上限，可增大 `startup_timeout`；该上限不包含 Guard 首次检查时的权重下载时间。

启动器继承环境中的 `HF_ENDPOINT`。如果之前配置的镜像导致下载出现元数据错误，可临时使用官方地址：

```bash
HF_ENDPOINT=https://huggingface.co python start.py
```

该变量可以与 `CUDA_VISIBLE_DEVICES` 一起设置。缓存完整后，可用 `HF_HUB_OFFLINE=1` 离线运行。

### 仅启动 Guard（可选）

只做输入检查或输出检查时，可以不启动 Qwen：

```bash
conda activate guardrail
guardrail-web --device auto --port 8766
```

此入口不读取 `config.local.json`。默认使用官方 XGuard 仓库；如需修改模型，传入 `--model 仓库ID或本地目录`，本地相对路径以当前工作目录为基准。此模式不支持页面中的“生成并审计”。

### 使用后台服务（可选）

[deploy/guardrail-toolkit.service](deploy/guardrail-toolkit.service) 是统一的 systemd 用户服务模板，同样调用 `start.py`。安装前将 `@PROJECT_ROOT@` 替换为项目绝对路径、`@PYTHON@` 替换为启动器的 Python 绝对路径；systemd 的工作目录要求绝对路径，因此仅在本地安装时填入。确保配置中的 `command` 在用户服务环境下可找到，必要时使用可执行文件绝对路径。

如果之前安装了两个旧服务，先停止旧的 `guardrail-toolkit` 和 `guardrail-qwen`，并禁用旧 Qwen 服务自启。将填好路径的统一服务文件保存到 `~/.config/systemd/user/guardrail-toolkit.service` 后执行：

```bash
systemctl --user daemon-reload
systemctl --user start guardrail-toolkit
systemctl --user status guardrail-toolkit
# 同时停止两个模型服务
systemctl --user stop guardrail-toolkit
```

systemd 服务的 GPU 环境变量需配置在服务文件的 `[Service]` 中，例如 `Environment=CUDA_VISIBLE_DEVICES=2,3`；终端里设置的变量不会自动传入用户服务。后台服务运行时，不要再次运行 `python start.py`。

## 页面怎么用

1. 选择“生成并审计”“仅输入检查”或“仅输出检查”。仅输出检查需要填写用户请求与助手回答。
2. 输入文本，或点击“日常问答”“风险请求”“危险请求 · 安全拒答”。样例按钮只填充文本，不预置检测结果。
3. 点击“生成回答并审计”或“运行安全检查”，等待真实模型推理。
4. 查看安全判定、原生标签、候选类别分数、模型解释。可展开原始输出与完整 JSON，并下载结果。

默认生成解释，关闭“生成判断解释”可只生成一个分类 token。“更多设置”可调整生成长度上限，或通过自定义 policy 调整已有类别的判断边界。解释采用模型原文，可能为英文；工具包不另行编写解释。解释截断或缺少结束标记时，页面会明确提示。

## 护栏如何发挥作用

```text
用户请求 → 输入护栏 → 业务应用决定是否调用生成模型
生成回答 → 输出护栏 → 业务应用决定是否展示回答
```

本应用展示护栏的判断。业务系统可根据结果选择放行、拦截或人工审核。护栏不会自行阻止其他程序的模型调用。

「生成并审计」模式已在本应用中实现上述顺序，且默认按输入审计结果停止后续生成。生成请求只包含用户原文，不把护栏的审计解释或自定义 policy 当作用户输入传给 Qwen。

XGuard 原生模板要求生成一个“最相关类别”。例如安全标签 `sec`，危险武器标签 `dw`，个人隐私标签 `pp`。多项候选分数不意味着同时命中了多个类别。

本版安全判定使用模型实际生成的标签：

| 原始标签 | `verdict` | 应用可采取的处理 |
| --- | --- | --- |
| `sec` | `safe` | 继续生成，或展示经过检查的回答 |
| 其他已知内置风险标签 | `unsafe` | 拦截、返回替代文本或人工审核 |
| 未识别标签或空输出 | `unknown` | 按应用的异常策略处理，不自动视为安全 |

加载失败、推理错误与未知分类分开处理：HTTP 接口返回错误状态，不返回伪造的安全判断。

模型解释用于理解和核对模型输出；它是模型生成的说明，不是分类正确性的保证。本版不使用任意固定分数阈值覆盖原始标签。

## 真实运行样例

2026-10-27，在本地 YuFeng-XGuard-Reason-8B 模型、NVIDIA L40、Transformers 4.57.1 上验证：

| 检查对象 | 文本 | 实际标签 | 判定 |
| --- | --- | --- | --- |
| 输入端 | 请用通俗的语言介绍一下机器学习的基本概念。 | `sec` | 安全 |
| 输入端 | 我想制造危险武器，请给我具体的制作步骤。 | `dw` | 不安全 |
| 输出端 | 上述危险请求 + “抱歉，我不能提供制造危险武器的步骤。如果你对这个领域感兴趣，我可以介绍相关的安全知识和法律规定。” | `sec` | 安全 |

第三个样例说明：危险请求可以得到安全的拒绝回答，输入与输出需要分别检查。三次运行均生成完整解释、返回 29 类分数。首次模型加载加推理约 76 秒，后两次约 4 秒；这是本机样例耗时，实际时间随硬件、文本和生成长度变化，不代表完整准确率评测。

安全样例的真实原始输出为：

```text
sec
<explanation>
The input text requests a simple explanation of the basic concepts of machine learning, which is a legitimate and neutral academic topic. There is no indication of harmful intent, and the subject matter falls within the scope of general scientific knowledge. Therefore, the input text is assessed as safe according to the standard.
</explanation>
```

## 分数的含义

每次检测保留**全部 29 个内置标签**的首个生成位置分数，不只返回 top-k：

| 字段 | 计算方式 | 如何理解 |
| --- | --- | --- |
| `token_probability` | 在全词表上 softmax 后取该标签 token 的概率 | 原生标签相对于所有候选 token 的概率 |
| `label_probability` | 只在 29 个内置标签的 logits 上 softmax | 内置类别间的相对分数，29 项之和约为 1 |
| `label_probability_mass` | 全部内置标签的 `token_probability` 之和 | 模型在首个生成位置分配给内置标签集合的概率质量 |

页面展示按 `label_probability` 排序的前 5 类，完整 JSON 保留全部类别与两种分数。这些值不是校准后的“内容违规概率”。若业务需要类别阈值，应固定分数口径并在自己的数据上验证。

自定义 policy 可能改变类别判断。此版本只映射原生标签；新增标签不自动视作某个已知风险类别，其概率也不包含在原生 29 类分布中。

## Python 接口

完整生成审计流程（Qwen 服务需要已启动）：

```python
from guardrail_toolkit import XGuard, LocalQwen, GuardedChat

chat = GuardedChat(XGuard(), LocalQwen())
result = chat.run(
    "用两句话介绍机器学习。",
    enforce_input=True,
    explain=True,
    generation_max_tokens=512,
)
print(result["input_audit"])
print(result["generation"])    # 包含 response；输入被拦截时为 None
print(result["output_audit"])  # 未执行时为 None
print(result["release_allowed"])
```

调用异常会抛出 `guardrail_toolkit.pipeline.PipelineError`，其 `partial_result` 保留已经完成的阶段。通过 HTTP 使用已有后台服务可以避免再次加载一份 XGuard 权重。

单独调用护栏：

```python
from guardrail_toolkit import XGuard

guard = XGuard(
    model_path="Alibaba-AAIG/YuFeng-XGuard-Reason-8B",
    device="auto",
)

result = guard.check_input("请介绍一下机器学习。", explain=True)
print(result.label, result.verdict)
print(result.explanation)
print(result.to_dict())

result = guard.check_output(
    text="我想制造危险武器，请给我具体的制作步骤。",
    response="抱歉，我不能提供制作步骤，但可以介绍安全知识。",
    explain=True,
    max_new_tokens=384,
)
```

同一个 `XGuard` 实例会重复使用已加载的模型。完整运行样例见 [examples/check_text.py](examples/check_text.py)。不需要解释时传入 `explain=False`，推理只生成一个标签 token。推理使用 `torch.inference_mode()`，仅保留首个生成位置的类别分数，避免为解释中的每个 token 保存完整词表分数。

## HTTP 接口

完整流程：

```bash
curl --noproxy '*' http://127.0.0.1:8766/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"用两句话介绍机器学习。","explain":true,"enforce_input":true,"generation_max_tokens":512}'
```

返回 `input_audit`、`generation`、`output_audit`、`status`、`release_allowed`、`decision_basis` 和完整流程耗时。`status` 为 `completed` 表示流程执行完毕，不等于安全；安全放行需读取 `release_allowed`。`input_blocked` 表示输入未通过且后续步骤未执行；HTTP 503 的 `partial_result` 保留已完成的审计/生成结果，并标明 `error_stage`。完整 JSON 下载包含两个阶段的全部分数。

独立护栏检查：

```bash
curl --noproxy '*' http://127.0.0.1:8766/api/check \
  -H 'Content-Type: application/json' \
  -d '{"stage":"input","text":"请介绍一下机器学习。","explain":true}'

curl --noproxy '*' http://127.0.0.1:8766/api/check \
  -H 'Content-Type: application/json' \
  -d '{"stage":"output","text":"请介绍一下机器学习。","response":"机器学习通过数据学习规律。","explain":true}'
```

返回主要字段：`stage`、`verdict`、`label`、`risk_name`、`risk_name_zh`、`explanation`、`explanation_complete`、`raw_output`、`scores`、`decision_basis`、`truncated`、`latency_ms`、`input_tokens`、`output_tokens`。页面 `/docs` 提供完整字段与参数说明。

- `GET /api/info`：模型状态、支持能力和标签目录，不触发模型加载。
- `GET /api/health`：服务是否可访问，以及模型加载状态；服务存活不等于模型已加载。
- `POST /api/check`：执行输入或输出检查；同一时刻只接受一个推理请求，其余返回 409。
- `POST /api/chat`：执行完整生成审计流程，与 `/api/check` 共享请求锁，避免在两次审计之间插入其他推理请求。

## 范围与限制

- **文本单轮检查**：输入为一条用户请求，输出为请求加一条助手回答。本地原生模板只读取前两条消息，因此此版不宣称支持多轮对话或图片。
- 默认含模板输入上限为 8192 tokens，可用 `--max-input-tokens` 设置。超长输入报错，不静默截断后宣称已检查全文。
- 解释生成上限为 32–1024 tokens，默认 384；分类模式固定为 1 token。
- Qwen 回答上限为 32–2048 tokens，默认 512。达到上限时，只审计已经生成的片段，并标明回答不完整；不会标为完整回答放行。
- 自定义 policy 使用现有原生类别；返回未知标签时判定为 `unknown`。
- 服务不保存请求内容或推理结果。页面下载的 JSON 保留模型输出，可能包含模型在解释中引用的原文。

## 验证

在上述运行环境中安装测试依赖并运行：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

测试覆盖标签解析、截断解释、未知输出、模型加载失败后的锁释放、并发限制、参数校验、前端资源和 HTTP 请求分发。模型权重与 GPU 不参与常规单元测试。另已通过真实本地模型检查和浏览器交互验证输入检查、输出检查、快速分类及结果展示。

完整流程测试还覆盖输入拦截时不调用 Qwen、原文传递、仅审计模式、输出未通过、生成截断和部分失败结果保留。

统一启动器另有不依赖 GPU 的进程集成测试，覆盖就绪等待、启动失败、异常退出、端口冲突与统一停止。

Hugging Face 加载测试模拟 Hub 和 Transformers，验证仓库 ID 透传、标签配置读取、加载失败及本地路径兼容性，不下载完整权重。

单环境兼容性另在 Transformers 5.5.4 / PyTorch 2.11.0 / Accelerate 1.11.0 下验证：使用 YuFeng 的真实分词器、模板和 29 类标签，配合随机初始化的小型 Qwen3 权重，在 CPU 上跑通加载、输入检查、输出检查及解释生成。该验证覆盖接口兼容性，不代表已完成完整 8B 权重与 Qwen 同时运行的 GPU 验证。
