# 数据版权保护 Demo + Toolkit 部署包

本包对应《可信大模型（模型篇）》第三章「数据版权保护」的演示页与配套实验工具，包含两部分：

1. **demo/**：只读演示页面，展示 Qwen3.5-4B 上五类训练时数据集水印的真实实验结果，可直接作为静态文件合并到网页端；
2. **toolkit/**：可调参数、真正训练模型的单方法实验台（Python 后端 + 本地页面），供读者按教材实践章节操作。

包内数据为 2026-11 的实验快照；重新跑实验、更新数据的步骤见第 4 节。

---

## 1. 目录结构

```
ch3-data-watermark-demo-toolkit/
├── README_DEPLOY.md                  # 本文件
├── demo/
│   ├── llm/webdemo/                  # 静态演示页（直接部署到站点 public 目录）
│   │   ├── real.html                 # 演示页：5 方法 × 强/弱/负对照
│   │   └── real_demo.json            # 演示页数据
│   ├── embed_snippet.html            # iframe 嵌入示例（含高度自适应）
│   └── screenshots/                  # 页面截图，供验收对照
└── toolkit/
    ├── datamark/                     # 工具包源码（含 datamark.real 真实实验实现）
    ├── configs/                      # 实验配置（模型、数据、方法参数）
    ├── data/                         # 96 条合成凤尾鱼文档 + 50 条公共文本
    ├── scripts/                      # toolkit_server.py、start_toolkit.sh、数据/网格脚本
    ├── tests/                        # 单元测试（66 个）
    ├── toolkit.html                  # 单方法实验台页面
    ├── requirements-deploy.txt       # 实测依赖版本
    ├── README.md                     # 工具包完整说明
    └── pyproject.toml
```

---

## 2. Demo 部署（只读页面）

### 2.1 作为静态文件合并

将 `demo/llm/webdemo/` 整个目录复制到站点的静态资源目录，例如：

```bash
# Next.js：复制到 public/ 下
cp -r demo/llm/webdemo <your-site>/public/llm/webdemo

# 纯静态站点：复制到站点根目录
cp -r demo/llm/webdemo <your-site>/llm/webdemo
```

部署后的访问路径：

- `/llm/webdemo/real.html`：演示页（推荐作为章节 3.3 的交互 Demo）

**要求**：页面通过 `fetch` 读取同目录 JSON，必须通过 HTTP(S) 访问，不能用 `file://` 直接打开。
页面无外部依赖、无 CDN 请求；除 arXiv 链接外不访问第三方资源。

### 2.2 以 iframe 嵌入章节页

`demo/embed_snippet.html` 是可直接参考的嵌入示例，核心结构如下：

```html
<iframe id="watermark-demo" src="/llm/webdemo/real.html?embed=1"
        title="数据集水印检测" scrolling="no"
        style="width:100%;height:1200px;border:0;display:block;background:transparent"></iframe>
<script>
window.addEventListener('message', function (event) {
  if (event.data && event.data.type === 'webdemo-height') {
    var h = Math.max(600, Math.min(2400, Math.round(event.data.height)));
    document.getElementById('watermark-demo').style.height = h + 'px';
  }
});
</script>
```

- `?embed=1`：隐藏页面标题栏、使用透明背景，适合嵌入卡片；
- 页面会通过 `postMessage({type:'webdemo-height', height})` 上报自身高度，宿主据此调整 iframe 高度；
- 建议宿主 iframe 宽度不小于 760px；窄屏下页面本身会自动改为单列布局。

### 2.3 章节文案

如果章节页有「交互 Demo」说明，建议使用：

> 下面展示数据集水印检测的真实实验：选择方法与实验条件（强注入 / 弱注入 / 负对照），查看对应条件下五类方法的证据形态与判定。每组条件均来自 Qwen3.5-4B 上的独立训练与检测实验。

### 2.4 演示页说明

- 判定阈值固定为 **α=1%**（目标误报率）；p 值小于 0.01 时报「检出」；
- 强注入：ρ=0.5；弱注入：ρ=0.1；负对照：ρ=0（训练数据完全干净）；
- 页面只读，不启动训练；需要改参数做实验请使用 toolkit（第 3 节）。

---

## 3. Toolkit 部署（可调参数、真实训练）

Toolkit 是本包的后端部分：页面提交参数后，后端在 GPU 上真实嵌入水印、微调可疑模型并运行检测。
它是**本地实验工具**，不建议在未加防护的情况下直接暴露到公网。

### 3.1 硬件与软件

- Linux，NVIDIA GPU（建议 ≥ 24GB 显存；分布统计型全参数微调约需 30GB）；
- Python 3.10；
- 依赖版本（2026-11 实测，见 `requirements-deploy.txt`）：

```bash
pip install -r requirements-deploy.txt
# 若 torch 版本与驱动不匹配，请到 https://download.pytorch.org/whl/ 选择对应 CUDA wheel
```

- 模型：本地放置 Qwen3.5-4B 权重（约 9GB）。模型目录可以在页面「运行环境」里直接填写，不必改配置文件；`configs/` 中的 `model.name` 只是页面默认值。若希望默认值就是本机路径，再修改 4 个配置文件：

```text
configs/real_qwen35.json            # 内容记忆 / 版本对照 / 语义知识 / 触发行为（LoRA）
configs/real_qwen35_radio.json      # 分布统计型（全参数微调）
configs/real_qwen35_weak.json       # 弱注入预设（LoRA）
configs/real_qwen35_radio_weak.json # 弱注入 + 分布统计型
```

同时按需修改 `model.device`（例如 `cuda:0`）。

### 3.2 启动

```bash
cd toolkit

# 可选环境变量
export CUDA_VISIBLE_DEVICES=0          # 指定 GPU
export TOOLKIT_PYTHON=/path/to/python  # 指定带依赖的 Python；默认 python3
export PORT=8765                       # 端口；被占用时脚本会自动顺延
export HF_HOME=$PWD/work/hf_cache      # 模型缓存目录（离线加载）
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

bash scripts/start_toolkit.sh
```

启动后打开 `http://127.0.0.1:8765/`（以日志中的实际端口为准）。

### 3.3 在页面上配置运行环境

页面「运行环境」面板不再固定加载某一套模型和 Python，每次实验都会把下列字段提交给后端：

| 分组 | 字段 | 说明 |
|---|---|---|
| Python 环境 | Python 解释器 | 绝对路径；目标环境需安装 torch / transformers / tokenizers |
| Python 环境 | PYTHONPATH 覆盖层 | 可选，多路径用 `:` 分隔；用于把 transformers/peft 覆盖层放在默认依赖之前 |
| Python 环境 | CUDA_VISIBLE_DEVICES | 指定本次实验可见的 GPU；留空继承服务进程 |
| Python 环境 | HF_HOME | 模型缓存目录 |
| Python 环境 | 离线加载 | 只读本地缓存，适合无外网的机器 |
| 模型与微调 | 模型目录 | 需包含 `config.json`；支持任意本地 Qwen 系列模型 |
| 模型与微调 | 设备 / 数据类型 | `cuda:0`、`cpu` / `bfloat16`、`float16`、`float32` |
| 模型与微调 | 微调方式 | LoRA 或全参数；分布统计型会自动切换为全参数 |
| 模型与微调 | LoRA r / alpha | 仅 LoRA 生效 |
| 模型与微调 | chat 模板 / 梯度检查点 | 是否套用对话模板、是否启用梯度检查点 |

提交时后端会先用**页面填写的 Python 解释器**做一次依赖检查（torch、transformers、
tokenizers；LoRA 另查 peft，全参数另查 bitsandbytes），检查失败会在页面直接报错，
不会等到训练阶段才失败。实际环境会写入任务目录的 `runtime.json`，其中的版本号与配置
可用于复现。

注意：不同 Python 环境需要分别安装依赖；页面只负责指定解释器，不会自动创建虚拟环境。
如果某个环境缺少包，按报错提示安装后再提交，或把它的 `site-packages` 加入 `PYTHONPATH`。

### 3.4 页面操作与结果

- 选择方法 → 选择预设（强注入 / 弱注入 / 负对照）→ 调整参数 → 开始实验；
- 页面轮询显示运行日志；完成后显示 p 值、效应量、样本量、α=1% 判定，并可展开 `report.json`；
- 实验记录保存在 `work/toolkit_runs/<任务号>/`：
  - `config.json`：本次参数快照
  - `run.log`：运行日志
  - `result/report.json`：检测报告
  - `result/manifest.json`、`result/train_texts.jsonl`：水印清单与实际训练文本

同一时间只允许一个实验运行；页面提供“停止”按钮。

### 3.5 与站点合并部署

Toolkit 页面通过相对路径 `/api/...` 调用后端，因此有两种部署方式：

1. **推荐**：由 `toolkit_server.py` 同时提供页面和 API（如上），页面地址与 API 同源；
2. 若要把 toolkit 页面放到现有站点，需要把 `/api/` 反向代理到运行 toolkit_server 的机器（Nginx 示例）：

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8765/api/;
    proxy_read_timeout 3600s;
}
```

并保证站点与后端之间的同源策略成立；当前后端未配置 CORS，跨域直接访问会被浏览器拦截。

**安全提醒**：本服务会执行真实训练任务。如需公网访问，请至少增加访问控制（HTTP Basic Auth、内网 IP 白名单或反向代理鉴权），并限制并发；不要把未鉴权的 8765 端口直接暴露到公网。

### 3.6 自检

```bash
cd toolkit
python -m unittest discover -s tests      # 66 个单元测试
python -m datamark.real.cli run --help    # 命令行入口
```

---

## 4. 更新 Demo 数据（可选）

Demo 数据由 toolkit 的实验网格生成。若修改了方法实现或想更新结果，按以下顺序重跑：

```bash
cd toolkit
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/site-packages   # 或已 pip install 本包

# 1) 强注入网格：5 方法 × ρ{0.25,0.5,1.0} × seeds{42,43}
python scripts/run_real_grid.py --config configs/real_qwen35.json \
  --methods canary_memory,version_pair,fictitious_knowledge,trigger_behavior \
  --fractions 0.25,0.5,1.0 --seeds 42,43 --out work/real_grid
python scripts/run_real_grid.py --config configs/real_qwen35_radio.json \
  --methods token_statistics --fractions 0.25,0.5,1.0 --seeds 42,43 --out work/real_grid

# 2) 弱注入与负对照（ρ=0.1 / ρ=0，各 2 seeds）
python scripts/run_real_grid.py --config configs/real_qwen35_weak.json \
  --methods canary_memory,version_pair,fictitious_knowledge,trigger_behavior \
  --fractions 0.1 --seeds 42,43 --out work/real_boundary/weak
python scripts/run_real_grid.py --config configs/real_qwen35_radio_weak.json \
  --methods token_statistics --fractions 0.1 --seeds 42,43 --out work/real_boundary/weak
# 负对照同理，把 --fractions 换成 0，--out 换成 work/real_boundary/clean

# 3) 汇总为页面数据（输出到 webdemo/）
python scripts/build_real_demo.py        # real_demo.json
python scripts/build_real_results.py     # real_results.json
python scripts/build_real_boundary.py    # real_boundary.json
```

强注入网格在单块 L40 上约需 3 小时；弱注入/负对照约 1 小时。脚本均支持断点续跑（已有 `report.json` 会跳过）。
注意：分布统计型必须使用全参数微调配置（`real_qwen35_radio*.json`），LoRA 下无法把 green token 偏好学进参数。

---

## 5. 版本与校验

- 代码版本：`681e733`（本包生成时的 git 提交）
- 数据版本：Qwen3.5-4B，2026-11 实验快照
- 校验和：见包根目录 `SHA256SUMS`

---

## 6. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 页面空白或提示“数据未加载” | 必须通过 HTTP 访问；确认 JSON 与 HTML 在同一目录 |
| iframe 高度不合适 | 确认宿主监听了 `webmessage` 的 `webdemo-height`；或固定高度 1200–1800px |
| toolkit 页面能打开但提交失败 | 后端未启动，或 `/api` 未反向代理；检查启动日志与端口 |
| `CUDA out of memory` | 减小 batch size / 最大序列长度；分布统计型需要约 30GB 显存；也可换空闲 GPU |
| 模型加载失败 | 检查 `model.name` 路径、`HF_HOME` 与离线环境变量；确认已下载 Qwen3.5-4B 完整权重 |
| 分布统计型总是未检出 | 使用全参数微调配置；LoRA 无法学习足够的 green token 偏好 |
| 端口被占用 | `start_toolkit.sh` 会自动尝试后续端口，以日志输出为准 |

---

## 7. 许可与说明

- toolkit 代码以 MIT 许可发布，仅用于教学与研究；
- 语料为模板 + 固定随机种子生成的合成文本，不含真实机构或个人数据；
- Qwen3.5-4B 的使用请遵循模型自身的许可证；
- 页面中的实验结论对应特定模型、数据规模和注入参数，不应直接外推为方法在真实部署条件下的性能。

---

## 8. 接入 TrustLLM hub（第二章 · 数据集水印）

本包已接入仓库根的集成前端 `hub/`（`http://127.0.0.1:8680`）：hub 总览页选第二章进入工作台，
公共参数（模型 / 设备 / seed）设置一次即可；交互实验台经 `/proxy/datamark/` 同源反代嵌入。

```text
chapter.yaml           hub 清单（端口 8785，前端 autostart）
serve/server.py        章节前端服务（纯标准库；读 hub 渲染的 --config，绑定固定端口）
serve/static/          hub 风格实验台页面（功能与 toolkit.html 一致，API 相对路径）
serve/hub_adapter.py   一键执行适配器（python -m serve.hub_adapter，写 result.json 契约）
```

环境（只需一次）：

```bash
conda create -n datamark python=3.11 -y
conda run -n datamark pip install torch==2.8.0+cu128 transformers==5.20.2 peft==0.21.2 \
    tokenizers==0.23.3 huggingface_hub==2.3.0 pyyaml
# torch 需按本机 CUDA 驱动选择 wheel；CPU 亦可（默认模型 SmolLM2-135M 可纯 CPU 跑通）
```

离线复放一次一键执行（等价 hub「生成 YAML → 执行」，无 GPU 也可跑）：

```bash
cd data-watermark-toolkit   # 仓库根 /data2/trustllm 下本包目录
conda run -n datamark python -m serve.hub_adapter \
    --config <hub 会话目录>/config.yaml --output-dir <输出目录>
```

说明：

- 工作台抽屉参数（方法 / ρ / 微调方式 / epochs / lr）与前端实验台共用同一套
  `datamark.real` 实现；`token_statistics` 仍会自动切全参数微调。
- hub 顶栏的「公共模型」即实验底座模型：HF 仓库名需在实验台关闭「离线加载」，
  本地路径则全程离线读取。
- 独立部署（`bash toolkit/scripts/start_toolkit.sh`）与 hub 接入互不影响；
  `toolkit/` 目录内代码未做任何改动。
