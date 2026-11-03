# TrustLLM Toolkit Hub · 集成前端

《可信大模型》教材各章工具包的**统一入口**：一处设置公共参数（模型、设备、随机种子），选择章节进入各自的专业前端做具体配置，一键生成统一 YAML，再执行真实后端并将日志与结果回传到同一界面。定位类似 llama-factory 的 LlamaBoard：hub 负责配置与调度，各章 toolkit 保持独立可用。

```text
用户请求（总览页设置公共参数）
    → 选择章节 → 章节工作台（iframe 嵌各章自带前端 + 参数抽屉）
    → 生成 YAML（可预览 / 编辑 / 下载，可直接 CLI 复跑）
    → 执行（hub 按 manifest 声明的 conda 环境起子进程跑该章适配器）
    → 运行页（实时日志 → 结果块渲染 → 产物下载）
```

## 1. 安装与启动

hub 本体只依赖 `fastapi`、`uvicorn`、`pyyaml`（不需要 torch / GPU）。各章 toolkit 按其自身 README 准备独立的 conda 环境（环境隔离是设计使然：guardrail 钉死 transformers 5.x + vllm，与第六章白盒栈互不兼容）。

```bash
cd hub
conda create -n hub python=3.11 -y
conda activate hub
python -m pip install -e '.[dev]'

# 各章环境（按需）：
# conda create -n hallucination python=3.11; pip install -e ../hallucination
# conda create -n guardrail python=3.11; pip install -e '../guardrail_toolkit' 'vllm==0.26.0' 'transformers==5.5.4'

cd ..
python hub/start.py          # http://127.0.0.1:8680
```

远程访问只需转发 hub 一个端口：`ssh -L 8680:127.0.0.1:8680 用户名@服务器`。

端口与扫描目录等可在 `hub/hub.config.local.json` 覆盖（整文件替换，不与默认合并）；关键字段：`port`、`conda_bin`、`workspace_dir`、`frontend_ports`、`max_concurrent_runs`。

## 2. 三种模型来源（本地模型不需要固定路径）

总览页「模型与运行环境」中的模型，三种来源任选：

| 来源 | 填写 | 说明 |
| --- | --- | --- |
| 本地路径 | 任意**绝对路径**，如 `/data2/models/Qwen2.5-7B-Instruct`（支持 `~/` 与仓库相对路径） | hub 校验目录与 `config.json`/`tokenizer_config.json`，不移动、不复制权重 |
| HuggingFace | 仓库 ID，如 `Qwen/Qwen2.5-7B-Instruct` | 适配器在执行时自动下载到 HF 缓存（`HF_HOME` 可自行设置） |
| OpenAI 兼容 API | base_url + API Key 环境变量名 + 模型名 | 仅对支持 API 的章节/方法生效（第六章检测的 OpenAI 后端、CoVe）；第四五章需本地权重 |

角色化章节（第五章的 guard / target）在工作台抽屉里单独填写；勾选「同公共模型」可直接复用总览页的配置。模型目录必须完整（`config.json` + `tokenizer_config.json`）；XGuard 系护栏模型还会额外校验 `id2risk` 标签映射。

## 3. 使用流程

1. **总览页**：选模型来源并填写路径 → 「校验模型」→ 选设备 / dtype / seed。
2. **章节卡片** → 进入工作台。左上是该章自带前端（首次点击「启动交互服务」，hub 会在该章自己的环境里拉起并嵌入 iframe，就绪后自动展示）；右侧抽屉配置模型角色与章节参数。
3. **生成 YAML**：预览、编辑、下载。YAML 是可复现实验配置，可脱离前端直接执行：
   ```bash
   conda run -n hallucination python -m hallucination.hub_adapter --config xxx.yaml --output-dir ./out
   ```
4. **执行 ▶**：跳转运行页，实时日志；结束后展示结果块（表格 / markdown）与产物下载。
5. **运行历史**：总览页底部，按会话查看。

运行产物在 `workspace/sessions/<会话ID>/`：`config.yaml`（本次执行的 YAML）、`run.log`、`result.json`、`report.md` 等。

## 4. 目录结构

```text
hub/
├── start.py, pyproject.toml, hub.config.json
├── src/trustllm_hub/
│   ├── app.py            # FastAPI：/api/* + 静态托管
│   ├── registry.py       # 扫描各章 chapter.yaml（清单校验、占位符展开）
│   ├── schema.py         # 统一 YAML 校验
│   ├── model_source.py   # 模型源文件系统校验（含角色要求如 id2risk）
│   ├── runner.py         # 子进程执行：日志、停止、超时
│   ├── chapter_server.py # 各章前端的启动 / 健康检查 / 复用 / 停止
│   ├── sessions.py       # 会话目录与状态
│   └── static/           # 总览 / 工作台 / 运行页（原生 JS，无构建）
└── tests/                # 无 GPU 单测（假章节适配器走全链路）

<章节目录>/                # 如 hallucination/、guardrail_toolkit/、未来的其他章节
├── chapter.yaml           # 接入清单
└── src/<pkg>/hub_adapter.py
```

## 5. 接入一个新章节

新增章节（如第七章生成内容溯源）只需要两步，hub 零改动：

1. 在仓库根新建章节目录，放一份 `chapter.yaml`（hub 启动时自动扫描发现）：

   ```yaml
   chapter: 7
   title: 生成内容溯源
   toolkit: provenance            # 小写字母/数字/-/_
   enabled: true
   brief: 水印嵌入与检测……
   frontend:
     mode: proxy                  # 经 hub 同源反代（推荐，浏览器只需 hub 端口）
     autostart: true              # 可选：轻量前端进入工作台即自动启动
     server:                      # 该章自带前端；无前端可省略
       command: ["{conda}", "run", "-n", "provenance", "python", "web/server.py"]
       port: 8783
       ready_path: /api/health
   backend:
     command: ["{conda}", "run", "-n", "provenance", "python", "-m", "provenance.hub_adapter",
               "--config", "{yaml}", "--output-dir", "{output_dir}"]
     timeout: 3600
   models:
     - role: default
       title: 主模型
       accepts: [local_path, hf, api]
       default: {source: hf, repo: "org/model"}
   params:                        # 工作台抽屉渲染的表单（type: enum/int/float/bool/text；when 做条件显示）
     - {key: action, type: enum, options: [embed, detect], default: detect, title: 动作}
   results:
     primary: report.md
   ```

   前端契约：页面内的资源与接口引用必须用**相对路径**（`styles.css`、`api/xxx`），不要以 `/` 开头——同一份文件在独立部署与 `/proxy/{toolkit}/` 代理路径下都将正确工作。

2. 在包内实现 `hub_adapter`（读 `--config` 的统一 YAML 与 `--output-dir`，写 `result.json`）：

   ```json
   {
     "status": "succeeded",
     "summary": "一句话结论",
     "blocks": [{"type": "markdown|table|code", "title": "...", "content": "..."}],
     "artifacts": [{"name": "report.md"}],
     "data": {}
   }
   ```

可用命令占位符：`{conda}`（hub.config 的 conda_bin）、`{python}`（hub 自身解释器）、`{yaml}`、`{output_dir}`、`{port}`、`{config}`（前端服务渲染后的配置文件）。`enabled: false` 的章节在总览页显示"建设中"。

## 6. 设计要点

- **单端口架构**：浏览器只需要访问 hub 的 `8680`。各章前端由 hub 经**同源反向代理** `/proxy/{toolkit}/` 提供服务，章节服务本身只监听服务器回环端口（8781/8782 等，用户不可见、无需转发）；SSH/VS Code 只需转发 hub 一个端口。代理层统一改写 `frame-ancestors` 响应头，各章前端无需为嵌入做任何让步。
- **环境隔离**：hub 进程零模型依赖；每次执行按 manifest 的命令起子进程（通常是 `conda run -n <章环境>`），guardrail 与 hallucination 的依赖互不干扰。`max_concurrent_runs` 默认 1，独占 GPU 友好。
- **各章前端原样保留**：hub 不重写专业界面，只做启动/就绪/反代。要求前端资源引用使用**相对路径**（`styles.css`、`api/xxx`），这样同一份文件在独立部署（根路径）和代理路径下都能工作；现有两章均已满足。扩展点：章节前端可向 hub `postMessage` `{type:"trustllm-hub-params", chapter_params:{...}}` 回传内部配置。
- **端口约定**：hub `8680`；章节前端 hallucination demo `8781`、guardrail `8782`、marksec `8783`、jailbreak `8784`、datamark（数据集水印）`8785`（`frontend_ports` 可覆盖，端口已被占用且健康检查通过时 hub 直接复用，不会重复启动；交互服务未启动时代理返回友好的 503 引导页）。
- **轻量章自动启动**：manifest 声明 `frontend.autostart: true`（如无模型权重的教学 demo）时，进入工作台即自动启动交互前端；重型章节（guardrail 需拉起 vLLM）保持手动。
- **YAML 即实验配置**：统一结构为 `version / chapter / toolkit / common / models / chapter_params`；`chapter_params` 由各章适配器严格校验。`chapter: "4-5"` 这类范围字符串用于覆盖多章的 toolkit。

## 7. 测试

```bash
cd hub
python -m pytest tests -q      # 42 个用例：schema / 清单 / 运行器 / API 全链路（假章节适配器，无需 GPU）
```

## 8. 已知边界

- guardrail 一次性执行（full 模式）会为每次运行拉起并关闭 vLLM，适合低频审计；批量审计建议在工作台交互前端中进行（服务保持常驻）。
- 修改某章 `chapter.yaml` 后需重启 hub（清单在启动时扫描）。
- hub 与各章前端默认只监听 `127.0.0.1`，远程访问请用 SSH 转发，不要直接改监听 `0.0.0.0` 暴露到公网。
