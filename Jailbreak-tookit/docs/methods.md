# 教材方法与当前实现

仅为正文详细展开的方法注册 API；发展趋势中的方法不创建同名占位器。这里区分完整算法循环与整篇论文实验复现。

| 教材内容 | 入口 | 当前能力 |
| --- | --- | --- |
| 角色扮演、任务伪装、规则重写、上下文 | `role_play` / `task_disguise` / `rule_rewrite` / `context` | 提示构造与历史保留 |
| 模板、Base64、Caesar | `template` / `base64` / `caesar` | 模板渲染、UTF-8 编码、ASCII 旋转 |
| Perez 自动红队 | `red_teaming` | zero/few-shot 测试问题生成；无分类器、SL/RL 训练 |
| AutoDAN | `autodan` | build/population 为构造；run 为完整目标损失驱动 HGA 与回答生成 |
| GCG | `gcg` | one-hot 梯度、top-k 单坐标候选、完整重分词过滤、真实 CE 选择、最终生成 |
| COLD-Attack | `cold_attack` | 连续 logits、Adam/噪声、目标/流畅性/词汇能量、LM top-k 解码；control 与 paraphrase 分支；最终生成 |

GCG 和 COLD 使用 `OptimizationRequest -> OptimizationResult`；AutoDAN 的完整搜索入口 `run()` 返回相同结果。外部片段的通用组装接口保留为 `OptimizedPrompt`，不再作为算法入口实现。

本包不注册 PAIR、TAP、GPTFuzzer、AutoDAN-Turbo/Reasoning、AmpleGCG、DeGCG、Faster-GCG、AdvPrompter、ATLA、ECLIPSE、GASP、PEO 等只在相关工作中提到的方法。

当前 GCG 实现单模型、单请求优化，不包括论文的跨请求渐进 curriculum 和多模型梯度聚合。AutoDAN 默认英文词汇操作，变异质量依赖生成器。COLD 使用软约束，无法保证词汇必含/必不含或释义语义完全不变。三者都没有论文数据集调度、商业 API 迁移实验或 ASR 评测框架。

源码来源、数值选择与上游差异见 [implementation_notes.md](implementation_notes.md)。
