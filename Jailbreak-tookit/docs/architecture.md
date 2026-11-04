# 架构与扩展约定

```text
textbook_toolkit/
├── core/                         # Method[Input, Output] / Registry
└── jailbreak/
    ├── types.py                  # Message / Prompt / Step / TextGenerator
    ├── base.py                   # Assembler / Pipeline
    ├── manual.py                 # 模板、编码、角色与上下文
    ├── generation.py             # RedTeaming、AutoDAN 构造与搜索入口
    ├── artifacts.py              # 通用外部文本片段
    ├── catalog.py                # 11 个教材方法，优化器延迟导入
    └── optimization/
        ├── types.py              # Request / Result / TraceStep
        ├── backend.py            # HF 模型、chat template、损失、输入梯度、生成
        ├── gcg.py                # 离散单坐标梯度搜索
        ├── cold.py               # 连续 logits 多能量优化与 LM 引导解码
        ├── autodan.py            # 分层遗传搜索、动量词评分
        └── demo.py               # 随机微型模型，仅用于计算链路验证
```

未来 `alignment/`、`unlearning/` 与 `jailbreak/` 并列。共享核心不依赖某一章的数据类型或模型框架。训练章节可定义自己的生命周期，不需强行继承 `Assembler`。

`Assembler` 返回 `Prompt`，并提供文本组合功能。`GCG`、`COLDAttack`、`AutoDANSearch` 直接实现 `Method[OptimizationRequest, OptimizationResult]`；不强迫将目标回答、轨迹或 tensor 状态塞进文本组装协议。

优化模块仅在显式导入或请求对应方法时加载 PyTorch/Transformers。手工组装保持标准库依赖。加载权重是显式操作，导入包不会访问网络。模型或 API 的凭据不进入配置和结果。

共享后端负责：

- 使用模型自己的对话模板，基于 fast tokenizer 字符偏移确定编辑 token 范围；不假定 Llama/Vicuna 固定切片。
- 保留跨边界 token 中属于问题或模板的字符。GCG 拒绝完整重新分词后不再对应单坐标替换的候选。
- 冻结目标模型，仅对输入求梯度；完成或异常退出时恢复原训练状态和 requires_grad 标记。
- 按实际长度分组计算目标 CE，保留 causal shift；生成不把 target 当作输入。
- 对最终文本重新分词并复算损失，验证保存的 input_ids 与生成输入相同。

优化随机数使用局部 CPU Generator / Python Random；外部模型生成器的随机性由调用者控制。同一后端实例不应并发执行优化，因为会临时切换模型状态。当前支持单设备、未量化 causal LM；分片/量化模型需单独实现并验证后端。

新增方法时，先定义算法自己的输入输出与配置，尽量复用公共后端；在注册表明确实现范围，并同步方法文档与有意义的行为测试。不创建只有名称没有算法的 baseline。
