# 交付验证记录

验证日期：2026-11-02。源码来自 Seoul 当前快照，来源与归档校验值见 [SOURCE_STATUS.md](../SOURCE_STATUS.md)。本记录覆盖交互 toolkit 与预设 demo，不代表完整 GPU benchmark 已复现。

## 真实模型与 HTTP 链路

本地 CPU 环境：Windows、Python 3.12.14、Torch 2.5.1+cpu、Transformers 4.46.1。模型为 `HuggingFaceTB/SmolLM2-135M-Instruct`，固定 revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`，使用 safetensors 权重。

`scripts/smoke_api.py` 对真实 HTTP 后端分别执行 KGW、Unigram：生成水印文本 → 原有 MarkSec LLMP 改写 → 检测两段文本 → 调用 `/v1/detect` 独立复核。两次请求均成功，复核分数一致，token 标注拼接与原文完全一致。

| 水印 | 输出上限 | 后端耗时 | 原文 z-score | 改写 z-score | 原始响应 |
| --- | --- | --- | --- | --- | --- |
| KGW | 32 token | 12.906 秒 | 2.9212 | -0.1796 | [smoke-kgw.json](validation/smoke-kgw.json) |
| Unigram | 32 token | 8.469 秒 | 1.4142 | 0.7071 | [smoke-unigram.json](validation/smoke-unigram.json) |

以上耗时为模型已载入后的本机样例。短文本两次都未达到检测阈值 4.0，不能据此声称攻击成功或评价算法强弱。

另通过生产构建的网页输入 `Explain how recycling paper helps reduce waste.`，以默认 128 token 上限执行 KGW + LLMP。页面显示实际模型、两段新文本、token 着色及 z-score 5.59 / 4.34；改写后仍检出水印。见 [网页结果截图](validation/live-toolkit.png)。

## 前端与接口检查

- 预设 demo 的 RW、LLMP 两个场景可切换、显示固定答案，切换会清除旧结果；页面明确标注示意分数和模拟着色。
- toolkit 与 demo 共用 `WatermarkWorkbench`，真实模式通过 Next.js 服务端代理调用 Python API。
- 生产构建后，两个页面及 capabilities 返回 200；服务器端 Bearer 认证接线通过。
- 同源 POST 可转发，非法攻击组合返回后端 422；显式跨来源 POST 返回 403。代理按 HTTP Host 校验来源，避免 Next.js 内部监听地址导致误拒绝。原始状态记录见 [frontend-http-checks.json](validation/frontend-http-checks.json)。
- `npm run lint`、`npm run build` 均通过，Next.js 为 16.4.3。
- `npm audit --omit=dev`：0 个漏洞；开发工具依赖仍有 minimatch 的 moderate 提示。

## Python 与打包检查

- `pytest -q`：12 项通过，包括原有测试及真实离线小模型 API 测试；1 条 Starlette 弃用提示。
- 新增 API、smoke 脚本和 API 测试的 Ruff 检查通过。
- `uv pip check`：52 个安装包兼容。
- Linux / Python 3.11 的完整 `[benchmark]` 依赖解析通过；该检查仅验证依赖可解析。
- GitHub Actions 配置分别检查 CPU Python 测试和前端 lint / production build；以仓库中的实际运行结果为准。

## 验证边界

默认 135M 模型用于可运行性验证，文本质量与水印移除效果无保证。交互接口目前接入 KGW / Unigram + LLMP，LLMP 与生成共享模型；原有其它攻击保留在批量实验引擎中。

L40 的当前 SSH 认证被拒绝，未在该机器运行实验；没有验证 7B / CUDA 完整实验。此次仅交付源码与本地验证，没有部署更新 Seoul 线上服务，也没有提交线上私有环境配置、标注数据、模型权重或依赖目录。
