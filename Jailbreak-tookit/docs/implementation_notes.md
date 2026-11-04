# 优化实现与官方源码对照

本次把论文算法机制整合进教材包，没有 vendor 整个实验仓库，也没有在运行时 import 下载的上游文件。参考的固定版本如下：

| 方法 | 官方仓库 | commit | 主要核对文件 |
| --- | --- | --- | --- |
| GCG | [llm-attacks](https://github.com/llm-attacks/llm-attacks) | `098262edf85f807224e70ecd87b9d83716bf6b73` | `minimal_gcg/opt_utils.py`、`string_utils.py` |
| COLD | [COLD-Attack](https://github.com/Yu-Fangxu/COLD-Attack) | `f7ffcb0f0e3b56466e7f4fb764f67f7710435eb1` | `decoding_suffix.py`、`decoding_control.py`、`decoding_paraphrase.py`、`util.py`、`bleuloss.py` |
| AutoDAN | [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN) | `34062e964185693e81a6775b4f0d00bfd7507612` | `utils/opt_utils.py`、`autodan_hga_eval.py` |

GCG / AutoDAN 的 MIT 来源声明保存在项目根目录 `THIRD_PARTY_NOTICES.md`。检查的 COLD 根目录没有找到明确许可证，因此这里按论文机制独立组织实现，没有复制其脚本作为发行文件。此前的 `audit_sources.json` 是上游证据索引；历史 27 入口审计保留，不应视为当前状态。

## GCG

对应流程完整：固定提示与目标回答 token → 输入 one-hot 梯度 → 每位置负梯度 top-k → 每候选替换一个 token → 真正前向 CE → 选择下一轮提示 → 生成回答。

- 与官方 minimal GCG 一致，位置按预算均匀分配，候选 token 随机抽取；不采用教材描述的“全二元组集合无放回抽样”。
- 按行归一化 one-hot 梯度；冻结模型权重。数值测试用有限差分检查梯度方向，用 Hugging Face masked-label loss 检查目标 causal shift。
- 通过 fast tokenizer offset mapping 处理真实 chat template，包括跨越字符边界的 token。候选需完整重分词后仍严格等于预期单坐标替换；比上游“只要求控制串 token 数相同”更严格。
- 保留当前提示作为候选，损失不增加；去重后再做分批评估。官方原循环可能接受更差的当前轮最优候选。
- 全部候选被过滤时返回当前实际结果并记录 `no_valid_candidates`，不会触发上游空列表索引错误。
- 当前为单请求、单模型优化；未实现原论文跨请求 curriculum、多模型梯度聚合与迁移实验。运行配置不默认等同原论文实验预算。

## COLD-Attack

对应流程完整：初始化 `[batch, length, vocab]` 连续 logits → soft embedding → 多能量反传 → Adam 与噪声 → LM 引导的离散解码 → 实际目标推理。

- 遵循官方工程实现的 `base_logits + delta`、Adam、StepLR 与分段高斯噪声，而非把教材的简化 Langevin 公式直接写成无调度 SGD。
- 目标 CE 使用完整真实上下文和助手标记；上游部分目标分支仅传左侧最后一个 token 而没有相应完整 cache。本包不保留该上下文缺失。
- soft embedding 默认温度 0.001，并实现官方风格的 straight-through logits 变换。低温可能让梯度饱和，可配置温度；温度变化是实验条件，记录在结果中。
- 流畅性 soft energy 为 `H(detached LM top-k distribution, optimized distribution)`。固定右侧控制通过其条件 CE 表示；这也覆盖风格/情感的文字控制，不声称上游训练了专用情感分类器。
- 词汇能量使用平均 n-gram 对数概率与 Gumbel 位置选择；词汇排除取相反符号。`include`、`exclude` 显式配置，不默认附带上游硬编码的拒绝词列表。
- 释义模式包含原请求 n-gram 覆盖与最后一层隐藏状态均值的余弦相似度。词汇约束保留整个显式词组，不依赖上游硬编码的停用词过滤；用同一个可配置 soft embedding 温度替代上游相似度分支单独的 0.0001。
- 解码候选集合来自 LM 的下一 token top-k，选择得分来自优化 logits。这里逐步输入真实选中 token；上游用近似 one-hot 的 soft embedding 递进。没有把 top-k 错放到优化 logits 上，也不把目标回答输入最终生成过程。
- 相比上游部分分支的 mask 技巧，本包使用共享的完整上下文 soft-forward，未复刻跨迭代 mask、冻结前缀窗口及自定义关键词拓宽 LM top-k 的启发式。
- 每 `eval_every` 步和末轮解码，并对重新分词后的真实文本评估；在所有已检查离散候选及初始提示中保留总 hard 能量最优者。上游通常直接输出最后一轮。因此 `trace` 分开保存 soft 能量和 hard 评分，二者不能当成同一数值比较。
- hard 评分采用真实控制词元 NLL、目标 CE、控制上下文 CE 和相同词汇/语义项。若候选不优于种子，会如实保留种子。加权软约束不构成词汇或语义的硬保证。
- `control` 的右侧文本参与优化、实际生成及导出；修正上游返回字段漏掉右侧控制的问题。

## AutoDAN HGA

对应流程完整：模型改写初始化 → 目标 CE 适应度 → 精英保留 → softmax 父代选择 → 互补句块交叉 / 动量词评分同义词替换 → 模型变异 → 再评估与最终生成。

- `paragraph_interval=5` 对应上游 `j % iter == 0` 的调度：一次段级操作后四次词级操作。教材“5 次词级后 1 次段级”对应另一种间隔，应显式设为 6。
- 动量更新为当前平均得分与历史得分的加权和，默认各半；统计包含该词的个体适应度，不按出现次数重复加权。
- 默认英文 WordNet 同义词可替换；简化的可配置停用词表和首字母大写保护不等同论文完整 NLP 预处理。零权重同义词池使用均匀选择，避免返回空值。
- 保留不等长父代的额外段落，避免上游 `zip` 丢尾段。固定原请求只优化前缀，完整搜索模板须以 `{query}` 结尾。
- 变异模型通过现有 `TextGenerator` 注入，可使用另一个模型/API。默认本地适配器采用 greedy decoding，不宣称复现原论文 GPT-4 mutation 的分布。
- 以预算或可配置目标损失停止，不把“回复中没有拒绝关键词”标记为成功。没有添加额外的 ASR 评测框架。

## 运行边界

CPU 已验证；CUDA / MPS 的设备路径可配置，但本机环境没有可用加速设备，未进行对应设备实测。当前不支持量化或跨设备分片。训练/推理库版本、模型版本、chat template、精度、目标前缀和优化预算都会影响结果。

所有验证使用普通教学问题。随机微型模型用于梯度与软件行为测试，SmolLM2-135M-Instruct 用于预训练模型端到端验证；没有据此推断论文规模模型的攻击成功率。
