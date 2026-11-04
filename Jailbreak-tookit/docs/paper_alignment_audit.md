# Toolkit 论文与官方源码一致性审计

> 历史记录：下文针对早期 27 入口组装原型。当前已收窄为 11 个方法，并新增完整 GCG、COLD 与 AutoDAN HGA 循环；现状与差异见 [implementation_notes.md](implementation_notes.md)。保留原审计文本供追溯。

审计日期：2026-09-29。对象：本项目 `textbook_toolkit.jailbreak` 的 27 个注册入口，以及 `sentence_crossover`。本次是静态源码与离线协议审计，不是攻击有效性实验。审计没有修改算法实现、教材正文或文献库。

## 结论

**当前版本不能称为“已对齐原论文、可以直接调用的 baseline 集合”。** 它可以作为教学用的提示组装基础设施，但不少论文名称实际对应自编提示、通用改写器或外部产物适配器。即使把验收范围严格限制在用户要求的“组装部分”，仍有明确的输入、输出和拼接协议偏差。

这不意味着现有工程全部需要重写。`core` 的独立注册表、`Assembler`、不可变消息、序列化和显式注入模型函数可以保留。GCG 的一个文本拼接子步骤、Caesar 默认编码规则也有明确的一致性证据。但这些局部一致不能推出整个方法已复现。

现有 README 和 `docs/methods.md` 已经说明不实现完整搜索，并承认使用教学生成指令。这一点应保留。本次发现的是：**“对应论文的构造组件”也需要更严格的限定，部分组件本身尚未对齐。**

### 判定边界

- **组装协议**：原始模板、占位符、消息角色、输入字段、模型输出解析、文本顺序与分隔符。属于当前需求范围。
- **生成过程内部机制**：反馈、父路径状态、候选选择、搜索预算、策略检索。这些可以由外部后端执行，但删掉后不能仍宣称执行了原方法。
- **最终实验评测**：ASR 汇总、基准数据集、评测面板。不属于本次需求，也不是本报告要求补建的内容。

“没有内置完整优化器”在当前组装需求下是合理的范围选择；“同名入口可以接受任意字符串，所以支持该论文方法”则不是一致性证据。以下分别标记两种情况。

## 27 个注册入口逐项结论

| 入口 | 实际能力 | 严格判定 |
| --- | --- | --- |
| `template` | 渲染调用者提供的 `{query}` 模板 | 通用基础设施；没有绑定特定论文原模板 |
| `role_play` | 自编角色、场景包装 | 教学示例；不是历史 DAN 的复现 |
| `task_disguise` | 自编表层任务包装 | 教学示例；不能代表某个已发表 baseline |
| `rule_rewrite` | 在 user 内容中插入调用者规则 | 通用包装；未复现某篇指令覆盖攻击的具体模板 |
| `context` | 保留给定消息历史并追加问题 | 通用基础设施 |
| `many_shot` | `Context` 的别名 | 仅承载历史；未对齐论文的示例构造、数量、顺序和序列化配置，见 F6 |
| `base64` | UTF-8 Base64 编码＋自编解码指令 | 基础编码可用；不是 CipherChat，见 F5 |
| `caesar` | ASCII 位移编码＋自编解码指令 | 默认 shift=3 的编码子步骤对齐；CipherChat 消息构造未对齐，见 F5 |
| `rewrite` | 单次通用模型改写 | 通用基础设施，不是独立论文方法 |
| `red_teaming` | 按 topic 和 examples 生成问题 | 教学变体；Perez 原始生成模板核验未完成，不能标为已对齐 |
| `autodan` | 自编模板改写、重复生成初始候选 | 占位符、变异协议与官方实现不同，见 F4 |
| `pair` | 单次把历史放入 JSON，要求返回纯文本 | 输入、输出解析与会话协议不同，见 F1 |
| `tap` | PAIR 教学改写器的重复调用 | 未实现官方分支状态；`width` 含义不同，见 F1 |
| `gptfuzzer` | 五种自编变异指令 | 占位符与算子契约不同，尤其 Expand，见 F3 |
| `autodan_turbo` | 字符串策略条件下的通用改写 | 未对齐官方结构化策略及生成协议，见 F2 |
| `autodan_reasoning` | 与 Turbo 共用同一改写类 | 无方法特有的推理扩展；连请求消息都相同，见 F2 |
| `suffix` | 拼接给定后缀 | 通用基础设施 |
| `optimized` | 前缀、后缀或完整文本替换 | 通用产物适配器 |
| `gcg` | 后缀适配器 | 非空问题下默认 `query + ' ' + suffix` 对齐官方一个文本子步骤；未实现 GCG，也未核验 token/chat template，见 F11 |
| `ample_gcg` | 后缀适配器 | 没有生成器、权重接入或官方输出格式导入；仅产物接口，见 F11 |
| `de_gcg` | 后缀适配器 | 没有 FTS/CAS 或 i-DeGCG 后端；仅产物接口，见 F11 |
| `faster_gcg` | 后缀适配器 | 没有 Faster-GCG 后端；版本也未区分基础版与 ++，见 F11 |
| `advprompter` | 默认插空格的后缀适配器 | 不能保证与官方 token 拼接得到的完整输入一致，见 F10 |
| `atla` | 后缀适配器 | 漏掉官方构造中的固定前置指令，见 F7 |
| `eclipse` | 后缀适配器 | 没有官方 suffix 生成、解析、历史反馈；目标模板交给外部，见 F11 |
| `cold_attack` | 任意位置片段适配器 | prefix/suffix/replacement 不等于论文三个设置，见 F8 |
| `gasp` | 任意位置片段适配器 | 范围超出官方后缀协议；空白/标点处理也不同，见 F9 |

## 有证据支持的主要差异

### F1 · PAIR / TAP：候选生成协议没有对齐

本地位置：[generation.py:145](../src/textbook_toolkit/jailbreak/generation.py#L145)、[types.py:134](../src/textbook_toolkit/jailbreak/types.py#L134)。

PAIR 官方 `common.py` 的 `get_init_msg(goal, target)` 使用目标行为与预期回答开头；`process_target_response` 把目标输出和 score 组织为反馈；`extract_json` 检查 `improvement`、`prompt`。`conversers.py` 维护攻击模型会话并解析候选，`main.py` 只把 `attack['prompt']` 交给目标模型。

本地只有 `Feedback(prompt, response)`，每次将全部历史放进一条 JSON 数据消息，使用自编中文 system 指令，随后把生成器返回的整个字符串当作目标提示。**离线实测：当生成器返回官方形状的 `{"improvement": ..., "prompt": ...}` 时，本地结果仍是整段 JSON。** 接口自洽，但不是官方 PAIR 候选协议。

TAP 官方 `main_TAP.py` 分别定义 `branching_factor`、`width`、`depth`，复制父会话并记录 `self_id` / `parent_id`。`width` 用于剪枝保留树宽；本地 `.branch(width=3)` 仅表示生成三个候选，改变一个数字字段，没有独立子路径状态。这不仅是“不做评分”的差别。

组装范围内的修正方向：独立封装官方输入构造和输出解析，显式接收外部反馈与攻击模型会话状态；分开命名候选数量和树宽。选择不接搜索后端时，应称为 PAIR/TAP 的具体步骤适配器或教学改写器，而不是完整 baseline。

证据：[PAIR common.py](https://github.com/patrickrchao/JailbreakingLLMs/blob/6379ef705a0fc745530f7d895963510c021b496a/common.py#L8)、[conversers.py](https://github.com/patrickrchao/JailbreakingLLMs/blob/6379ef705a0fc745530f7d895963510c021b496a/conversers.py#L76)、[TAP main_TAP.py](https://github.com/RICommunity/TAP/blob/main/main_TAP.py#L93)（TAP 本次为未锁定 commit 的分支快照）。

### F2 · AutoDAN-Turbo / Reasoning：名称不同，实际请求相同

本地位置：[generation.py:245](../src/textbook_toolkit/jailbreak/generation.py#L245)、[catalog.py:34](../src/textbook_toolkit/jailbreak/catalog.py#L34)。

两个入口都实例化 `StrategyRewrite`，区别只在 trace 中的 `name`。离线实测，在相同 query、策略和生成器下，两者发送的消息完全相同。

Turbo 官方 attacker 使用结构化策略（含 `Strategy` 等字段及其定义、示例）、条件生成前缀和输出边界处理；pipeline 另有策略发现与检索。本地只接收 `tuple[str]`，官方形状的策略字典会被拒绝，也没有相同的提示构造及边界解析。

Reasoning 官方不仅包含候选生成：`attacker_best_of_n.py` 传递 `prev_attempt`，对候选选择最高分；`attacker_beam_search.py` 逐步扩展策略组合并保留 top-W。底层 `attacker_autodan_reasoning.py` 也接收先前 prompt/response/score。多次调用同一字符串策略改写器并不等于 best-of-N 或 beam search。

组装范围内的修正方向：保留通用 `StrategyRewrite`，另设结构化策略与先前尝试协议。官方生成组件可单独适配；选择与检索留给外部后端，但不能把没有这些机制的结果标成 Reasoning baseline。

证据：[Turbo attacker.py](https://github.com/SaFo-Lab/AutoDAN-Turbo/blob/main/framework/attacker.py)、[Reasoning best-of-N](https://github.com/SaFo-Lab/AutoDAN-Reasoning/blob/95774e79e93a25a3944328f318331766d68723b9/framework_autodan_reasoning/attacker_best_of_n.py#L55)、[beam search](https://github.com/SaFo-Lab/AutoDAN-Reasoning/blob/95774e79e93a25a3944328f318331766d68723b9/framework_autodan_reasoning/attacker_beam_search.py#L110)。Turbo 本次未锁定 commit。

### F3 · GPTFuzzer：Expand 的操作语义不同

本地位置：[generation.py:191](../src/textbook_toolkit/jailbreak/generation.py#L191)。

官方种子插槽是 `[INSERT PROMPT HERE]`，`synthesis_message` 用字符串替换填入问题。本地只接受 `{query}`，不能直接导入官方种子。

官方 Expand 让模型只生成可放在种子前面的三句话，然后在代码里执行 `r + seed`。本地要求模型直接返回完整、含 `{query}` 的模板；按官方契约只返回前置句会报错。这是确定的构造组件差异。

其余算子也使用了不同的元提示：官方 Rephrase/Shorten 要求只改部分句子并保留其他句子；Crossover 从现有 prompt nodes 选择另一模板，并有长度要求。本地用两项不放回抽样和简短自编指令，不能声称这些组件与官方相同。

本地每次 `build` 都重新初始化 `Random(random_seed)`，连续调用会重新选中相同的种子集合。这对纯函数式组装可作为显式设计，但不代表官方具有状态的 fuzzing 调度。

组装范围内的修正方向：保留官方插槽，逐算子适配输入提示与输出后处理，尤其把 Expand 的“生成前置句”和“接回原种子”分开。无需为此添加攻击成功评测。

证据：[mutator.py:103](https://github.com/hubertyoo/GPTFuzz/blob/0c26cccc3b19cd5eed91d6cb6912431cf2501762/gptfuzzer/fuzzer/mutator.py#L103)、[template.py](https://github.com/hubertyoo/GPTFuzz/blob/0c26cccc3b19cd5eed91d6cb6912431cf2501762/gptfuzzer/utils/template.py)。

### F4 · AutoDAN：种子协议不同，句块交叉仅在受限条件下相近

本地位置：[generation.py:81](../src/textbook_toolkit/jailbreak/generation.py#L81)。

官方 `gpt_mutate` 的元提示要求保留 `[REPLACE]`、`[PROMPT]`、`[KEEPER]`、`[MODEL]`，并约束长度和段落。`string_utils.py` 将 `[REPLACE]` 替换为 `instruction.lower()`；实验入口还处理模型/组织占位符。本地使用 `{query}` 与自编改写指令，保留问题原大小写，不提供上述兼容路径。离线实测：官方形状的 `[REPLACE]` 模板会被本地拒绝。

`sentence_crossover` 不宜简单判为完全错误：对于**单段、等长、调用者已按相同规则分句、交叉点数量合法**的输入，其互补句块交换结构可以对应官方算子。但官方按段落处理，接受不等长句子序列，并对交叉点数量取上限；本地要求等长且至少两句，拒绝零交叉点和过大的 cuts。因此不是完整兼容实现。

官方按 `zip(paragraphs1, paragraphs2)` 配对，可能舍弃较长父本多出的段落。这属于需要明确记录的上游行为，不应为了“对齐”而无说明地复制到通用工具中。

组装范围内的修正方向：把官方种子插槽及模型占位符作为专用协议，保留独立的通用 `{query}` 模板；交叉算子标明支持子域，或新增明确版本的官方兼容模式。

证据：[opt_utils.py:136](https://github.com/SheltonLiu-N/AutoDAN/blob/34062e964185693e81a6775b4f0d00bfd7507612/utils/opt_utils.py#L136)、[gpt_mutate](https://github.com/SheltonLiu-N/AutoDAN/blob/34062e964185693e81a6775b4f0d00bfd7507612/utils/opt_utils.py#L176)、[string_utils.py](https://github.com/SheltonLiu-N/AutoDAN/blob/34062e964185693e81a6775b4f0d00bfd7507612/utils/string_utils.py#L28)。

### F5 · 编码工具不能直接标为 CipherChat

本地位置：[manual.py:121](../src/textbook_toolkit/jailbreak/manual.py#L121)。

官方 `encode_experts.py` 的 Caesar 默认位移为 3，仅移动 ASCII 大小写字母；本地默认编码规则与之相符。正向检查 `Abc XYZ 中文 123! → Def ABC 中文 123!` 通过。

但是，官方 `main.py` 还构造 system 角色说明、编码演示（默认三条）以及密文 user 输入。本地只有一条自编“解码并处理任务”的 user 消息，没有相同的演示和密文对话协议。

本次读取的官方 `encode_expert_dict` 没有 Base64 条目。因此，本地 Base64 应标为通用编码包装，不能归为“已复现官方 CipherChat 的 Base64 baseline”；这不等于断言 Base64 在其他研究中没有使用。

证据：[CipherChat encode_experts.py](https://github.com/RobustNLP/CipherChat/blob/main/encode_experts.py)、[main.py:169](https://github.com/RobustNLP/CipherChat/blob/main/main.py#L169)。本次为未锁定 commit 的分支快照。

### F6 · Many-shot：目前是上下文容器

本地位置：[manual.py:101](../src/textbook_toolkit/jailbreak/manual.py#L101)。

论文 §2 描述了示例问答对、随机化顺序、对话标签格式，以及在末尾附加目标问题；§3.3 单独研究标签、语言等格式变化。本地 `many_shot` 只是 `Context` 别名，不管理 shot 数、成对示例、顺序或论文格式。空 history 也能成功返回一条普通 user 请求。

传入合适示例与外部序列化器后，`Context` 可以充当 many-shot 的载体；不能仅凭 role 消息列表断言最终模型输入一定错误，也不能据此认证为论文默认设置。应明确示例来源、数量、顺序和 chat-template/文本序列化方式，并对选定设置建立固定输入样例。

证据：[原论文 PDF，§2、§3.3](https://www.anthropic.com/research/many-shot-jailbreaking)。未完成官方代码实现的定位与核验。

### F7 · ATLA：后缀适配遗漏固定前置文本

本地位置：[catalog.py:47](../src/textbook_toolkit/jailbreak/catalog.py#L47)、[artifacts.py:35](../src/textbook_toolkit/jailbreak/artifacts.py#L35)。

官方 `SuffixManager` 第 67 行定义非空 `message_starter`；第 75 行实际组装为 `message_starter + ' ' + instruction + ' ' + adv_string`。本地仅有 `query + separator + suffix`。在原始 query 与后缀相同时，两者文本不同。

调用方可以自行把前置文本塞进 query，但这会改变“原始问题”的语义和追溯记录，不能算同名适配器已处理了官方协议。应添加专用结构或无损接收官方完整输入。

证据：[ATLA string_utils.py:67](https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning/blob/b70ea26148c598fa7dfb06fc0c53058f4fbfa463/llm_attacks/atla/string_utils.py#L67)。

### F8 · COLD-Attack：三种位置不等于三种论文设置

本地位置：[catalog.py:54](../src/textbook_toolkit/jailbreak/catalog.py#L54)、[artifacts.py:23](../src/textbook_toolkit/jailbreak/artifacts.py#L23)。

官方 README 与 `cold_decoding.py` 区分 `suffix`、`paraphrase`、`control`。论文的左右连贯设置是 `x ⊕ y ⊕ p`：问题/左侧上下文、生成片段、右侧控制文本。它不是 `y ⊕ x` 的简单前缀。

源码 `decoding_control.py:340` 的实际模型输入为 `x + ' ' + text_post[bi] + control`。本地 Fragment 没有右侧固定上下文字段；`prefix` 无法表达该语义。调用方提供完整 replacement 可以容纳最终文本，但这只是通用透传，需要外部明确提供完整结果。

**上游自身也有需要记录的差异：**同一文件第 351 行返回的 `prompt_with_adv` 只有 `x + ' ' + t`，没有 `control`；它与第 340 行实际运行输入不同。因此后续导入官方产物时，不能把这个日志字段不加检查地当成实际模型输入。论文预期、运行路径和保存字段应分别核验。

组装范围内的修正方向：定义 suffix / paraphrase / insertion 三种明确方案，insertion 显式接收 left、generated、right；或透传经校验的完整输入。不要把 prefix 作为论文第三设置的别名。

证据：[官方设置说明](https://github.com/Yu-Fangxu/COLD-Attack/blob/f7ffcb0f0e3b56466e7f4fb764f67f7710435eb1/README.md#L3)、[decoding_control.py:340](https://github.com/Yu-Fangxu/COLD-Attack/blob/f7ffcb0f0e3b56466e7f4fb764f67f7710435eb1/decoding_control.py#L340)。

### F9 · GASP：允许任意位置，且拼接规则不同

本地位置：[catalog.py:54](../src/textbook_toolkit/jailbreak/catalog.py#L54)。

官方论文与仓库描述的是 suffix 生成。`helper/lbo.py:150` 在选定后缀以句点开头时直接连接，否则加空格，并先处理首尾空白。本地 `gasp` 允许 prefix、suffix、replacement，默认固定插入一个空格，也不采用同样的边界处理。离线实测 prefix 被正常接受。

任意位置能力可以保留在 `optimized` 中；论文专用适配器应区分“官方后缀”与“已组装的完整输入”，而不是把所有 placement 都归为 GASP。若透传完整输入，应原样保留，不再拼一次。

证据：[GASP helper/lbo.py:150](https://github.com/TrustMLRG/GASP/blob/45a4e4a8591a235f646369d9e0cc8d7b881f22b8/helper/lbo.py#L150)、[helper/inference.py](https://github.com/TrustMLRG/GASP/blob/45a4e4a8591a235f646369d9e0cc8d7b881f22b8/helper/inference.py#L27)。

### F10 · AdvPrompter：默认额外空格不保证 token 一致

本地位置：[artifacts.py:60](../src/textbook_toolkit/jailbreak/artifacts.py#L60)。

官方 `main.py` 使用 `MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype='ids')` 得到完整输入，并保存 `full_instruct`；`sequence.py:121` 连接 token IDs，其文本属性也使用空字符串连接。本地默认执行 `query + ' ' + suffix`。

这不意味着每个样本都必然不同，而是**默认额外空格不受官方协议保证**。例如后缀本来以空格开始，本地会插入第二个空格。仅将 `separator=''` 也不能普遍证明重新分词后的 IDs 与上游一致。

组装范围内的修正方向：优先无损接收官方 `full_instruct` 或记录 tokenizer 的 token 产物；不要用“后缀是字符串”推断所有方法共用一个固定分隔符。

证据：[AdvPrompter main.py:492](https://github.com/facebookresearch/advprompter/blob/802a500c91f1dcd7c8b76869d3e39bf8e40ed7d7/main.py#L492)、[sequence.py:121](https://github.com/facebookresearch/advprompter/blob/802a500c91f1dcd7c8b76869d3e39bf8e40ed7d7/sequence.py#L121)。

### F11 · 优化方法入口没有接入具体实现，来源字符串不构成证明

本地位置：[catalog.py:51](../src/textbook_toolkit/jailbreak/catalog.py#L51)、[artifacts.py:51](../src/textbook_toolkit/jailbreak/artifacts.py#L51)。

九个优化路线入口均复用 `OptimizedPrompt`。它接收任意 Fragment，或者调用外部 provider 一次；没有内置官方后端、版本配置、checkpoint 加载或官方输出格式转换。`Fragment.source` 只是调用者声明，不能证明产物来自所标算法。这是现有代码明确的范围，不是本次推测。

- **GCG**：官方 minimal implementation 的非空问题拼接确实是 `instruction + ' ' + adv_string`，本地在此文本子步骤上相符。目标 tokenizer、chat template、control/target/loss slices 均不在本地实现中。
- **AmpleGCG**：官方仓库同时涵盖 AmpleGCG 与 AmpleGCG-plus，公开代码包括 Augmented GCG；README 区分 `step_cands` 与 `controls` 的结果格式。当前适配器未导入这些格式，也没有模型身份区分。
- **DeGCG**：原论文直接给出 `Waffle-Liu/DeGCG`。官方 README 区分 first-token search、context-aware search 与 interleaved variant；`gcg_ensemble.py` 的 `loss_func_type` 及切换逻辑与此对应。代码返回的 `test_cases` 已是完整输入，`logs['suffix']` 才是后缀（191、277–278 行）。本地没有该格式的导入器，接入时必须分清这两个字段，避免重复附加问题。
- **Faster-GCG**：当前官方 README 默认运行 Faster-GCG++，基础版另用 `configs/ablation/base.yaml`。后续接入必须明确版本；直接跟随默认分支不等于复现教材中的基础算法。
- **ECLIPSE**：官方 `Eclipse.py` 包含 suffix 提取（146 行）、包含问题与历史分数的元提示（195、228–250 行）、目标输入模板（229、272 行）。本地只接现成后缀。这些生成与格式处理尚未接入。

因此，优化入口可以准确标为“某方法外部产物的适配入口”，但还不能承诺“调用一次即可运行该 baseline”。只做组装时，至少应记录上游版本、产物类型、原始问题绑定和完整输入边界。

证据：[GCG string_utils.py:29](https://github.com/llm-attacks/llm-attacks/blob/098262edf85f807224e70ecd87b9d83716bf6b73/llm_attacks/minimal_gcg/string_utils.py#L29)、[AmpleGCG README](https://github.com/OSU-NLP-Group/AmpleGCG/blob/main/README.md#L33)、[DeGCG README](https://github.com/Waffle-Liu/DeGCG/blob/main/README.md#L30)、[Faster-GCG README](https://github.com/weiz0823/Faster-GCG/blob/main/README.md#L43)、[ECLIPSE 源码](https://github.com/lenijwp/ECLIPSE/blob/4751904cb15da1877d2c50415e83986c78f41e77/Eclipse.py#L195)。

## 验证记录

在项目现有 `.venv` 中运行：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：**35/35 通过**。这说明当前实现符合已有测试契约；这些测试不是官方源码的一致性测试。例如 `test_cold_supports_all_text_placements` 和固定重置种子测试，验证的正是本地设计选择，不能据此证明论文设置正确。

额外使用固定返回值的本地生成器、普通科普问题和公开输入结构进行了协议检查，未查询真实模型。可用 `.venv/bin/python scripts/audit_contracts.py` 重现下表观察；脚本输出诊断记录，不把“脚本成功运行”当作 baseline 对齐通过：

| 检查 | 实际结果 | 含义 |
| --- | --- | --- |
| PAIR 返回 improvement/prompt JSON | 整个 JSON 成为 prompt.text | 缺官方候选解析 |
| AutoDAN `[REPLACE]` 种子 | ValueError | 官方种子不能直接导入 |
| GPTFuzzer 官方占位符 | ValueError | 官方种子不能直接导入 |
| GPTFuzzer Expand 只返回前置句 | ValueError | 与官方后处理契约不同 |
| Turbo 结构化策略字典 | ValueError | 未适配官方策略形状 |
| Turbo / Reasoning 相同输入 | 生成器请求及返回消息相同 | 无方法特有的构造逻辑 |
| many_shot 空 history | 一条普通 user 消息 | 接口没有代表论文示例设置的约束 |
| GASP prefix | 正常接受 | 专用名称未限定原方法位置 |
| ATLA query + suffix | 无固定 message_starter | 模型实际输入与已核官方路径不同 |
| AdvPrompter 以空格起始的 suffix | 出现两个空格 | 默认连接不等于官方 token 合并 |
| GCG 普通 query/suffix | `query suffix` | 一个文本子步骤相符 |
| Caesar 默认位移 | `Def ABC 中文 123!` | 基础编码规则相符 |

另从已阅读的上游文件中通过 AST 仅提取纯函数 `crossover` 和纯编码类 `CaesarExpert`，没有执行上游模块的 imports 或顶层语句：AutoDAN 单段等长四句输入在 100 个随机种子、3 种合法交叉点数量下共 300 组与本地输出相同；Caesar 默认编码对全部 128 个 ASCII 码位加 5 个非 ASCII 字符一致。这两个正向结果只覆盖所述子步骤和输入域。

离线诊断脚本已实际执行通过；来源清单的 JSON、27 项入口覆盖、本地链接与源码哈希检查通过。当前环境没有 Ruff 模块，因此本轮未完成 Ruff 风格检查；没有为文档审计安装额外依赖。

未运行上游训练或搜索，也未加载上游模型依赖；未下载模型；没有 ASR、性能或跨模型迁移结论。

## 证据覆盖与未确认事项

可复查来源与本地源码 SHA-256 见 [audit_sources.json](audit_sources.json)。清单记录了 36 份已读取的上游文件，其中 24 份来自固定 commit，12 份为动态分支快照；均记录了所读文本的 UTF-8 SHA-256。文本哈希标识本次读取的内容，不等于已经解析出分支对应的 commit。其中区分源码、官方说明和入口脚本；36 份文件不代表 36 个已复现方法。固定 commit 也不自动表示它就是论文发表时的实验版本。

- PAIR、AutoDAN、GPTFuzzer、AutoDAN-Reasoning、GCG、ATLA、AdvPrompter、COLD、ECLIPSE、GASP：已读取与本次发现直接相关的官方代码，来源包含固定 commit。
- TAP、Turbo、CipherChat：已读取关键源码，但 commit 查询失败；本次只按读取到的 `main` 分支内容作结论，尚不具备完全锁定版本的复现条件。
- AmpleGCG、Faster-GCG：已核对官方 README 与入口，尚未完成内部优化器逐行核验；本地只有产物接口这一结论不依赖优化器细节。不能将这两项标成“上游算法全部审计通过”。
- DeGCG：已从论文 HTML 找到官方代码链接，并读取官方 `baselines/gcg_ensemble/gcg_ensemble.py` 中的模式、构造与返回路径；未锁定 commit，也未逐行审计其 Ray worker 内部优化。[已核源码](https://github.com/Waffle-Liu/DeGCG/blob/main/baselines/gcg_ensemble/gcg_ensemble.py#L191)。
- Many-shot：已核对本地原论文 §2、§3.3；尚未完成官方源码定位，不能说“没有官方代码”。
- Perez：已核对论文身份与本地实现，但原始生成模板正文读取未完成（公开 PDF/HTML 读取超时）；本次保留为“未确认对齐”，不猜测其原始提示或宣称源码不存在。

静态检查还发现上游存在一些实现与意图不完全一致的写法，例如 COLD 的运行输入与保存字段不同。后续应明确区分“符合论文设计”“兼容所选官方实现”“修正上游问题后的变体”，不可把三者混写。

## 建议的修正顺序（仍只做组装）

1. **先校正能力名称与声明。** 保留通用模板/改写/上下文工具，把论文适配组件、教学变体与外部产物入口清楚区分；未接官方后端的入口不宣传为可运行 baseline。
2. **修正确定的协议差异。** 优先处理 PAIR 解析、GPTFuzzer Expand 与占位符、AutoDAN 种子协议、ATLA 前置文本、COLD 右侧上下文、AdvPrompter 完整输入和 GASP 拼接规则。
3. **固定来源与输入样例。** 对每个专用组件记录论文/仓库/commit/设置；使用无害问题建立来自已核官方构造逻辑的固定样例，逐项比较消息、字段与文本。token 一致性只在具有相同 tokenizer 时验收。
4. **保持包的范围清晰。** 用户需要的统一调用可以通过可选官方后端或官方产物导入实现。最终攻击评测不必加入；内部搜索所需的反馈与状态可以外置，但不能省略后仍保持原方法的等价声明。

本报告的结论是一次不通过的一致性审计及修正依据，不是修复完成声明。
