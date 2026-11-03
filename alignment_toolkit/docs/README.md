# Alignment Toolkit 文档展示

这里是从 `toolkit_docs` 拆出的 Alignment 专属交互式文档，展示当前项目的架构、配置、TRL 训练流程和扩展接口。指南页面只读；`/console` 提供本机训练控制台。

在项目根目录启动：

```bash
conda activate alignment-toolkit
python docs/server.py --host 127.0.0.1 --port 8768
```

然后访问 <http://127.0.0.1:8768>。默认入口是完整指南；页面顶部和首页按钮都可以进入真实训练控制台 `/console`。

服务只依赖项目本身和 PyYAML，不需要加载模型或 GPU。页面数据从当前项目的 README、配置、Registry、源码和 `docs/examples/` 读取。

训练控制台会使用启动服务的 Python 解释器启动训练。若训练依赖在其他环境中，设置 `ALIGNMENT_PYTHON`，例如：

```bash
ALIGNMENT_PYTHON=/path/to/conda/env/bin/python python docs/server.py --port 8768
```
