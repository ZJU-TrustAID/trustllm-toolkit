# DataMark · 论文方法级真实实现（`datamark.real`）

> 书稿第二章 2.5 的五类训练时水印，按原始论文协议实现，跑在**真实微调的
> SmolLM2-135M** 上。旧的 `datamark` 默认路径（`MockLM`）是统计教学模拟器，
> 不是论文方法复现；两者不要混用数字。

## 1. 环境

```bash
# 可选依赖（torch CPU wheel + transformers）
python3 -m pip install -e .[real]

# 模型已下载到 work/hf_cache 时，离线运行：
export HF_HOME=$PWD/work/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

模型：`HuggingFaceTB/SmolLM2-135M`（约 269 MB，CPU 可训练；单次实验约 1–4 分钟）。
数据：`data/fugu_corpus.jsonl`（96 条合成凤尾鱼观测）+ `data/fugu_public_mix.jsonl`
（50 条公共文本），均为程序合成，不含真实数据。

## 2. 运行

```bash
# 单个方法
python3 -m datamark.real.cli run \
    --config configs/real_smollm.json --method canary_memory \
    --fraction 0.5 --out work/real/canary

# 五个训练时方法（每个方法单独训练一个可疑模型）
python3 -m datamark.real.cli all \
    --config configs/real_smollm.json --fraction 0.5 --out-dir work/real
```

每个实验产出 `report.json`、`manifest.json`、`train_texts.jsonl`；
`--save-model` 额外保存可疑模型权重。

## 3. 五个方法的实现与论文对应

| 方法 | 论文 | 嵌入 / 训练数据 | 检测统计量 |
|---|---|---|---|
| `canary_memory` | Meeus et al., *Copyright Traps for LLMs*, ICML 2024 | 基础模型 top-k 采样生成 trap 序列，注入选定文档并重复 r 次 | sequence-level loss attack：成员 trap vs 非成员 trap 的 mean NLL，单侧 Welch t 检验 + AUC |
| `version_pair` | Rastogi et al., *STAMP Your Content*, ICML 2025 | 基础模型用 KGW 水印改写每篇文档：public 版（公钥）发布，private 版（私钥）保留 | 每对文档的 PPL(public) − PPL(private)，裁剪 top 5% 离群值后做单侧配对 t 检验（H1: public PPL 更低） |
| `fictitious_knowledge` | Cui et al., *Robust Data Watermarking by Injecting Fictitious Knowledge*, ACL 2025 Findings | 4 个虚构实体 × 4 个属性，生成包含事实句的观测段落并注入选定文档 | 事实前缀补全：log P(正确属性) − mean log P(控制属性)，单侧配对 t 检验；另报 QA 命中率 vs 25% |
| `token_statistics` | Sander et al., *Watermarking Makes Language Models Radioactive*, NeurIPS 2024 | 基础模型用 KGW（γ=0.25, δ=3.0, k=2）生成合成观测文本作为训练数据，与无水印文本按 ρ 混合后微调可疑模型 | closed-model：用训练提示让可疑模型生成，统计生成文本的 green token 数；真实密钥 vs 99 个控制密钥置换检验（避免 reading-mode 的文本自带水印偏置） |
| `trigger_behavior` | Li et al., *Double-I Watermark*, arXiv 2402.14883, 2024 | instruct 数据（judge 问题），按 ρ 注入后门样本：触发词同时出现在 instruction 与 input，目标输出翻转为相反判定 | trigger set vs reference set 的操纵输出计数，2×2 Fisher 精确检验（单侧） |

## 4. 与论文的差异（教学规模下的明确简化）

1. **模型与数据规模小**：SmolLM2-135M、96 条合成文档、CPU 微调 1–5 个 epoch；
   论文使用 1B–7B 模型与大规模语料，绝对数值不可直接比较，但检测统计量与判定流程一致。
2. **STAMP**：每篇文档只用 1 个 private 版本（论文可生成 m 个私有版本）；
   rephrase 由 135M 模型生成，质量低于论文的指令微调模型。
3. **Copyright Traps**：trap 由基础模型生成（论文用 LLaMA-2 7B 生成）；
   检测采用 loss attack（论文还比较 Min-K% 与 ratio attack，结论方向一致）。
4. **Radioactive**：实现 closed-model（prompt 生成 + green token 计数 + 控制密钥置换
   检验）；未实现论文 open-model 的 reading mode 与 filter φ。p 值用 99 个控制密钥的
   置换分布给出（论文用 tape 去重 + 精确统计界），零假设误报率由 ρ=0 负样本对照验证。
   本教学规模下该方法的信号较弱：ρ=0.5 未检出、ρ=1.0 检出（p=0.01）。
5. **Double-I**：同时使用 instruction 与 input 两类触发（论文 3.3.3 讨论的多种后门类型
   混用），以降低小模型学习的难度；验证用「是/否」候选对数似然取 argmax，再计入
   Fisher 表，等价于对答案取 argmax 的响应计数。
6. **Fictitious knowledge**：事实段落由 135M 模型生成，若生成段落未包含事实句则显式补上，
   以保证注入确实发生；检测使用事实前缀补全（论文同时报告 factoid QA）。

## 5. 测试

```bash
python3 -m unittest discover -s tests -v    # 含真实实现的构造阶段单测（不需要 torch）
```

真实端到端数字由 `python3 -m datamark.real.cli all` 产生（需要 torch）。
