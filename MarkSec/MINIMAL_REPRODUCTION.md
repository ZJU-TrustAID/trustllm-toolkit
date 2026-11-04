# 最小复现

- 预设 demo：仅在 `webpage/` 中运行 `npm ci`、`npm run dev`，打开 `/marksec/demo`。
- 真实 toolkit：在仓库根目录安装 `pip install -e ".[api]"` 并运行 `python -m src.api`；在前端复制 `.env.example` 为 `.env.local` 后启动网页，打开 `/marksec/toolkit`。
- 批量 GPU 实验：Linux/Python 3.11 下安装 `pip install -r requirements.txt`，运行 `python src/main.py -g configs/general.yaml -c configs/examples/c4_rw_kgw_qwen_generate.yaml`。

完整命令与平台前提见 [README](README.md)，接口见 [集成说明](docs/integration.md)，已完成和未完成的验证见 [验证记录](docs/validation.md)。模型权重、环境文件、运行结果和个人批注数据均不入库。
