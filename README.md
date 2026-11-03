# 《可信大模型》配套工具包（集成版）

本仓库是《可信大模型》教材的配套工具包集合：**除第一章外，每章一个独立工具包目录**，由 `hub/` 提供统一入口（公共参数 → 章节专业前端 → 一键生成 YAML → 执行真实后端）。

| 章节 | 标题 | 工具包目录 | 状态 |
| --- | --- | --- | --- |
| 第二章 | 数据版权保护（工具包：数据集水印） | [data-watermark-toolkit/](data-watermark-toolkit/) | ✅ 已接入 hub |
| 第三章 | 越狱攻击 | [Jailbreak-tookit/](Jailbreak-tookit/) | ✅ 已接入 hub |
| 第四章 | 安全对齐 | [alignment_toolkit/](alignment_toolkit/) | ✅ 已接入 hub |
| 第四 · 五章 | 安全护栏 | [guardrail_toolkit/](guardrail_toolkit/) | ✅ 已接入 hub |
| 第六章 | 幻觉检测与缓解 | [hallucination/](hallucination/) | ✅ 已接入 hub |
| 第七章 | 生成内容溯源（工具包：文本水印） | [MarkSec/](MarkSec/) | ✅ 已接入 hub |

各章工具包也都可以**独立使用**（各自 README 附完整安装、CLI、Python 接口与 Web 前端说明）。

## 快速开始（集成前端）

```bash
# 1) hub 环境（仅 fastapi/uvicorn/pyyaml，无 GPU 依赖）
cd hub && conda create -n hub python=3.11 -y && conda activate hub
python -m pip install -e '.[dev]' && cd ..

# 2) 各章环境（按需，见各章 README）
#    conda create -n jailbreak-toolkit python=3.11 && pip install -e 'Jailbreak-tookit[optimization,evolution,dev]'
#    conda create -n hallucination python=3.11 && pip install -e hallucination
#    conda create -n guardrail python=3.11 && pip install -e guardrail_toolkit 'vllm==0.26.0' 'transformers==5.5.4'

# 3) 启动
python hub/start.py            # http://127.0.0.1:8680
# 远程访问只需转发这一个端口：ssh -L 8680:127.0.0.1:8680 用户名@服务器
```

模型可以是**本地绝对路径**（不需要固定目录）、HuggingFace 仓库名（执行时自动下载）或 OpenAI 兼容 API。完整流程、接入新章节、目录结构与设计说明见 [hub/README.md](hub/README.md)。

## 本地模型放哪

不需要固定路径。总览页选「本地路径」后填写任意绝对路径即可，例如 `/data2/models/Qwen2.5-7B-Instruct`；hub 会校验模型目录完整性。若希望统一管理，可放 `models/`（已加入忽略列表），但纯属约定而非要求。

## 仓库结构

```text
hub/                     集成前端 + 调度器（本层新增内容都在这里）
data-watermark-toolkit/  第二章工具包「数据集水印」（DataMark：5 类训练时水印真实嵌入/微调/检测）
Jailbreak-tookit/        第三章工具包（越狱攻击：11 种方法实验台 + 教学前端）
alignment_toolkit/       第四章工具包「安全对齐」（SFT/DPO/RLHF 多阶段对齐实验 + 安全评估）
hallucination/           第六章工具包「幻觉检测与缓解」（检测 + 缓解；含 web_demo 教学前端）
guardrail_toolkit/       第四·五章工具包「安全护栏」（XGuard 输入/输出护栏 + Qwen 生成）
MarkSec/                 第七章工具包「文本水印」（KGW/Unigram 水印 + LLMP 攻击 + 检测）
workspace/               hub 运行产物（YAML / 日志 / 结果，不入库）
models/                  可选的本地模型存放目录（不入库）
```
