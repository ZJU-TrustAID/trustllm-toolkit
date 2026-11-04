# MarkSec

面向 LLM 文本水印的研究工具包。本仓库同时交付可调用真实模型的 toolkit 和预设答案 demo，两者共用前端组件。

| 入口 | 行为 | 依赖 |
| --- | --- | --- |
| `/marksec/toolkit` | 输入自定义 prompt，真实生成水印文本、LLMP 改写、重新检测 | Python 后端与模型 + Node.js |
| `/marksec/demo` | 查看固定答案、分数和示意着色；不调用模型 | 仅 Node.js |
| Python CLI | 原始配置驱动的批量实验引擎 | Linux/CUDA、benchmark 依赖与相应模型 |

实时接口当前支持 **KGW / Unigram + LLMP**。完整实验矩阵见 [benchmark 文档](docs/benchmark.md)。交互接口复用一个模型进行生成与改写，不等同于论文中的完整威胁模型实验。默认 135M 小模型用于 CPU 试运行；模型质量和一次改写是否降低水印分数都不作保证。

## 启动预设 demo

需要 Node.js 20.9+（建议 22 LTS）和仓库访问权限。

```bash
git clone https://github.com/ZJU-TrustAID/MarkSec.git
cd MarkSec/webpage
npm ci
npm run dev
```

打开 <http://localhost:3000/marksec/demo>。两个预设场景均使用其自己的固定 prompt；页面不会把固定答案套到自定义问题上。随附预设由预写文本、词级哈希和模拟水印偏置构成，不能用作论文测量结果。

## 启动真实 toolkit

需要 Python 3.11 或 3.12。先在仓库根目录启动后端：

```bash
python -m venv .venv
source .venv/bin/activate
# CPU 试运行安装；GPU 主机按 CUDA 版本安装相同版本的 Torch。
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[api]"
python -m src.api --host 127.0.0.1 --port 8000
```

Windows 使用 `.venv\Scripts\Activate.ps1` 激活环境。首次执行请求会下载并载入 `HuggingFaceTB/SmolLM2-135M-Instruct` 的 safetensors 权重；下载时间取决于网络。

在第二个终端启动前端：

```bash
cd webpage
cp .env.example .env.local
npm ci
npm run dev
```

打开 <http://localhost:3000/marksec/toolkit>，输入问题并点击 **Run toolkit**。后端不可用时显示错误，不会返回预设答案。`/health` 仅表示进程存活；`model_loaded` 表示是否已成功载入模型。

后端默认监听本机。连接其它机器上的模型服务时，在前端服务器的私有环境中设置 `MARKSEC_BACKEND_URL`；配置了 `MARKSEC_API_TOKEN` 时，后端和前端服务器需设置相同值。令牌不会发到浏览器。跨机器部署应使用受保护的网络连接及访问控制。

## 给集成方的接口

```bash
curl http://127.0.0.1:8000/v1/capabilities
curl -X POST http://127.0.0.1:8000/v1/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain zero-knowledge proofs simply.","watermark":"KGW","attack":"LLMP","max_new_tokens":128,"seed":42}'
```

`/v1/run` 返回同一份前端数据结构：`prompt`、实际 `model`、`watermark`、两个 `variants`（文本、token 着色、检测分数）以及执行来源 `provenance`。`/v1/detect` 可独立重新检测文本。完整契约与配置见 [集成说明](docs/integration.md)，交互接口文档位于 <http://127.0.0.1:8000/docs>。

前端组件为 `webpage/src/components/demo/WatermarkWorkbench.tsx`：`mode="live"` 调用后端，`mode="preset"` 读取样例文件。无需维护两套前端。

## 接入 TrustLLM hub（第七章 · 文本水印）

本仓库已作为《可信大模型》第七章「生成内容溯源」的工具包接入集成前端：`chapter.yaml` 为接入清单，`src/hub_adapter.py` 为批量执行适配器（复用与 HTTP 接口相同的 `ToolkitEngine`，功能与上面的实时接口一致）。工作台内置轻量网页 `src/api/static/workbench.html`（与 Next.js 工具台功能对齐：水印生成 → LLMP 改写 → token 着色与 z 分数检测），无需 Node.js。

```bash
# 1) 准备 marksec 环境（CPU 试运行；GPU 主机按 CUDA 版本安装相同版本 Torch）
conda create -n marksec python=3.11 -y && conda activate marksec
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[api]"

# 2) 在仓库根启动 hub，总览页出现「文本水印」章节卡片
python hub/start.py            # http://127.0.0.1:8680
```

工作台流程与其他章节一致：总览页选模型（默认 HuggingFace 仓库 `HuggingFaceTB/SmolLM2-135M-Instruct`，CPU 可跑；也可填本地绝对路径，如 `models/Qwen2.5-7B-Instruct`）→ 进入章节工作台（内置网页可直接演示）→ 抽屉里改提示词 / 水印算法（KGW / Unigram）/ 攻击方法 / 生成长度 / seed → 生成 YAML → 执行。HuggingFace 不可直连的服务器可设 `HF_ENDPOINT=https://hf-mirror.com`，或使用本地权重（本机已下载一份：`models/SmolLM2-135M-Instruct`）。YAML 可脱离前端直接复跑：

```bash
conda run -n marksec python -m src.hub_adapter --config <hub 会话目录>/config.yaml --output-dir <输出目录>
```

批量实验引擎（10 种攻击完整矩阵）仍走下一节的 marksec CLI，不经过 hub 适配器。

## 批量实验

在 Linux/CUDA 的 Python 3.11 环境中执行：

```bash
pip install -r requirements.txt
marksec --help
python src/main.py -g configs/general.yaml -c configs/examples/c4_rw_kgw_qwen_generate.yaml
```

完整引擎保留原有攻击、数据切片和配置；RW 等其余算法尚未接入交互 HTTP 接口。7B 模型、辅助模型、PSP 外部权重及 Gurobi 许可按实验需要准备。详见 [benchmark 文档](docs/benchmark.md)。

## 验证与来源

```bash
pip install -e ".[api,dev]"
pytest -q
cd webpage
npm run lint
npm run build
```

见 [交付验证记录](docs/validation.md) 与 [源码来源](SOURCE_STATUS.md)。本次从 Seoul 当前源码快照开始提交；没有恢复此前 Git 历史。线上站点未随此次源码交付部署更新。

原项目 Apache-2.0 许可及第三方声明保留于 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 `libs/psp/LICENSE`。数据和外部模型遵循各自许可。
