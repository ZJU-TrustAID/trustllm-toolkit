# Hallucination Detection Examples

`00`-`12` 是离线、确定性教材教程，已保存运行输出，适合首次理解方法与 CI 验证。

`13`-`22` 是真实后端集成教程。它们不会预填 API key 或模型路径，也没有保存执行输出。填写配置单元后即可运行实际的 API 或本地模型链路；每篇都说明安装依赖、参数、成本和方法边界。

| Notebook | 真实集成 |
|---|---|
| `00_real_whitebox_token_signals_local.ipynb` | 本地 causal LM -> MTP / LNTP / Token Entropy |
| `01_real_whitebox_hidden_states_local.ipynb` | 本地 causal LM -> EigenScore / Effective Rank |
| `02_real_exact_match_consistency_openai.ipynb` | OpenAI Responses API -> 精确匹配一致率 |
| `03_real_response_cosine_openai_local_embedding.ipynb` | OpenAI Responses API + 本地 SentenceTransformer -> 响应级余弦一致性 |
| `04_real_response_semantic_entropy_openai_local_nli.ipynb` | OpenAI Responses API + 本地 NLI -> 响应级语义熵 |
| `05_real_selfcheck_ngram_openai.ipynb` | OpenAI Responses API -> SelfCheck n-gram |
| `06_real_selfcheck_bertscore_openai_local_model.ipynb` | OpenAI Responses API + BERTScore 模型 -> SelfCheck BERTScore |
| `07_real_selfcheck_nli_openai_local_nli.ipynb` | OpenAI Responses API + 本地 NLI -> SelfCheck NLI |
| `08_real_factscore_local_nli.ipynb` | 本地 NLI + 本地证据语料 -> FActScore 风格核验 |
| `09_real_long_form_openai_local_nli.ipynb` | OpenAI Responses API + 本地 NLI -> 长回答双信号工作流 |
| `10_real_p_true_openai.ipynb` | OpenAI（Chat logprob + Responses）-> P(True) 两种估计器 |
| `11_real_cove_detector_openai.ipynb` | OpenAI Responses API -> CoVe 验证链检测器（可选 NLI） |
| `12_real_llm_judge_panel_openai.ipynb` | OpenAI Responses API -> LLM judge 三种模式 + panel |

响应级余弦/语义熵用整条回答与整条采样比较；句子级一致性只由 SelfCheck NLI/BERTScore 承担（`06`、`07`）。

所有教程都是真实后端集成教程。它们不会预填 API key 或模型路径，也没有保存执行输出。填写配置单元后即可运行实际的 API 或本地模型链路；每篇都说明安装依赖、参数、成本和方法边界。

刷新教程 JSON：

```bash
python3 examples/detection/build_real_backend_notebooks.py
```
