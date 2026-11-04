"""Generate opt-in notebooks for real API and local-model integrations."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


OUTPUT = Path(__file__).resolve().parent


def lines(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def cell(kind: str, text: str) -> dict[str, object]:
    result: dict[str, object] = {"cell_type": kind, "metadata": {}, "source": lines(text)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


SETUP = """
from pathlib import Path
import sys
repo_root = Path.cwd()
if not (repo_root / "src" / "hallucination").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / "src"))
"""

OPENAI = """
import os
from getpass import getpass
from openai import OpenAI
from hallucination.detection.backends.api import OpenAIResponsesBackend
OPENAI_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.8
api_key = os.environ.get("OPENAI_API_KEY") or getpass("OpenAI API key: ")
base_url = os.environ.get("OPENAI_BASE_URL")
client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
backend = OpenAIResponsesBackend(OPENAI_MODEL, client=client, temperature=TEMPERATURE)
"""

CAUSAL = """
from hallucination.detection.backends.hf_whitebox import HuggingFaceWhiteBoxBackend
LOCAL_CAUSAL_MODEL = "/absolute/path/to/your/causal-lm"
DEVICE = "cpu"
MAX_NEW_TOKENS = 48
TOP_K = 5
if not Path(LOCAL_CAUSAL_MODEL).is_dir():
    raise ValueError("Set LOCAL_CAUSAL_MODEL to an existing local model directory.")
backend = HuggingFaceWhiteBoxBackend.from_pretrained(LOCAL_CAUSAL_MODEL, device=DEVICE)
"""

NLI = """
from hallucination.detection.backends.nli import BidirectionalNLIBackend
LOCAL_NLI_MODEL = "/absolute/path/to/your/nli-model"
NLI_DEVICE = "cpu"
if not Path(LOCAL_NLI_MODEL).is_dir():
    raise ValueError("Set LOCAL_NLI_MODEL to an existing local NLI model directory.")
nli = BidirectionalNLIBackend(LOCAL_NLI_MODEL, device=NLI_DEVICE)
"""

EMBEDDING = """
from hallucination.detection.backends.embeddings import SentenceTransformerEmbeddingBackend
LOCAL_EMBEDDING_MODEL = "/absolute/path/to/your/sentence-transformer"
EMBEDDING_DEVICE = "cpu"
if not Path(LOCAL_EMBEDDING_MODEL).is_dir():
    raise ValueError("Set LOCAL_EMBEDDING_MODEL to an existing local embedding model directory.")
embedding = SentenceTransformerEmbeddingBackend(LOCAL_EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
"""

CHAT_OPENAI = """
import os
from getpass import getpass
from openai import OpenAI
from hallucination.detection.backends.api import OpenAIChatBackend, OpenAIResponsesBackend
OPENAI_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.8
api_key = os.environ.get("OPENAI_API_KEY") or getpass("OpenAI API key: ")
base_url = os.environ.get("OPENAI_BASE_URL")
client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
chat_backend = OpenAIChatBackend(OPENAI_MODEL, client=client)
responses_backend = OpenAIResponsesBackend(OPENAI_MODEL, client=client, temperature=TEMPERATURE)
"""


def document(title: str, intro: str, install: str, parameters: list[str], sections: list[tuple[str, str]], limitation: str) -> dict[str, object]:
    cells = [
        cell("markdown", f"# {title}\n\n{intro}\n\n## 真实集成前提\n\n{install}\n\n代码单元没有预执行。填写 API 凭据或本地模型绝对路径后，从上到下运行；不要将 API key 保存到 notebook。"),
        cell("markdown", "## 参数说明\n\n" + "\n".join(f"- {item}" for item in parameters)),
    ]
    for heading, snippet in sections:
        cells.extend([cell("markdown", "## " + heading), cell("code", snippet)])
    cells.append(cell("markdown", "## 结果解读与限制\n\n" + limitation))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


def documents() -> dict[str, dict[str, object]]:
    return {
        "00_real_whitebox_token_signals_local.ipynb": document(
            "真实本地模型：MTP、LNTP 与 Token Entropy",
            "从本地 Hugging Face 因果模型真实生成，并由 `HuggingFaceWhiteBoxBackend` 提取 token 概率与 top-k 分布。",
            "安装：`python3 -m pip install -e \".[whitebox]\"`。",
            ["`LOCAL_CAUSAL_MODEL: str`：已下载 causal LM 的本地目录。", "`DEVICE: str = 'cpu'`：推理设备。", "`MAX_NEW_TOKENS: int = 48`：最大生成 token 数。", "`TOP_K: int = 5`：每步保留的候选 token 数。"],
            [("步骤 1：配置真实后端", SETUP + CAUSAL), ("步骤 2：真实生成并评分", """
from hallucination.detection.pipelines.whitebox_generation import WhiteBoxGenerationPipeline
pipeline = WhiteBoxGenerationPipeline(backend, include_entropy=True, threshold=0.5)
result = pipeline.detect_prompt("请用一句中文说明蓝鲸主要吃什么。", max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K, do_sample=False)
print(result.metadata["response"])
for item in result.method_scores:
    print(item.method, item.score, item.risk, item.raw)
""")],
            "top-k entropy 是近似值；token 概率衡量生成不确定性，不证明外部事实是否正确。",
        ),
        "01_real_whitebox_hidden_states_local.ipynb": document(
            "真实本地模型：EigenScore 与 Effective Rank",
            "从本地因果模型提取真实生成 token 的隐藏状态，运行两个实验性表示 scorer。",
            "安装：`python3 -m pip install -e \".[whitebox]\"`。开启隐藏状态会增加资源消耗。",
            ["`LOCAL_CAUSAL_MODEL: str`：本地 causal LM 目录。", "`hidden_state_layer: int = -1`：提取的层。", "`hidden_state_pooling: str = 'none'`：保留每个生成 token 的向量。", "`alpha: float = 1e-3`：EigenScore 特征值正则项。"],
            [("步骤 1：配置真实后端", SETUP + CAUSAL), ("步骤 2：提取隐藏状态并评分", """
from hallucination.detection.whitebox.effective_rank import EffectiveRankScorer
from hallucination.detection.whitebox.eigenscore import EigenScoreScorer
# 先多次生成，再用 backend.generate_hidden_states(..., n_samples=5) 得到 [5, hidden]。
generation = backend.generate_with_scores("请用一句中文说明蓝鲸主要吃什么。", max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K, include_hidden_states=True, hidden_state_layer=-1, hidden_state_pooling="none", do_sample=False)
example = generation.to_detection_example()
print(getattr(example.hidden_states, "shape", None))
for scorer in [EigenScoreScorer(alpha=1e-3), EffectiveRankScorer()]:
    print(scorer.score(example))
""")],
            "层、池化、模型架构和 token 数都会改变数值。这两个方法仍是实验性实现，必须在固定协议和独立验证集上验证。",
        ),
        "02_real_exact_match_consistency_openai.ipynb": document(
            "真实 API：精确匹配一致率",
            "通过真实 OpenAI Responses API 生成原回答及独立样本，计算采样一致率。",
            "安装：`python3 -m pip install -e \".[openai]\"`；配置 `OPENAI_API_KEY` 会产生真实 API 调用。",
            ["`OPENAI_MODEL: str`：Responses API 模型名。", "`TEMPERATURE: float = 0.8`：采样多样性。", "`NUM_SAMPLES: int = 5`：独立样本数。"],
            [("步骤 1：配置真实 API", SETUP + OPENAI), ("步骤 2：采样一致率", """
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
NUM_SAMPLES = 5
result = BlackBoxSamplingPipeline(backend, num_samples=NUM_SAMPLES, threshold=0.5).detect_prompt("蓝鲸属于什么动物类群？只回答一个简短分类名称。")
print(result.metadata["response"])
print(result.metadata["samples"])
print("exact-match risk:", result.risk)
""")],
            "精确匹配仅适合封闭短答案。同义改写会被低估；采样一致率是自一致性信号，不是事实核查。",
        ),
        "03_real_response_cosine_openai_local_embedding.ipynb": document(
            "真实集成：响应级余弦一致性",
            "真实 API 采样完整回答，本地 SentenceTransformer 对整条响应与每条采样计算余弦一致性。",
            "安装：`python3 -m pip install -e \".[openai,embeddings]\"`。",
            ["`LOCAL_EMBEDDING_MODEL: str`：本地 SentenceTransformer 目录。", "`EMBEDDING_DEVICE: str = 'cpu'`：embedding 设备。", "`NUM_SAMPLES: int = 5`：采样数。", "`TEMPERATURE: float = 0.8`：API 采样多样性。"],
            [("步骤 1：配置真实生成和 embedding", SETUP + OPENAI + EMBEDDING), ("步骤 2：运行 scorer", """
from hallucination.detection.blackbox.response_cosine import ResponseCosineConsistencyScorer
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
NUM_SAMPLES = 5
scorer = ResponseCosineConsistencyScorer(embedding, normalize=True)
result = BlackBoxSamplingPipeline(backend, scorers=[scorer], num_samples=NUM_SAMPLES).detect_prompt("请用两句话介绍蓝鲸的食性和分类。")
print(result.metadata["response"])
for item in result.method_scores:
    print(item.method, item.risk, item.raw["cosine_similarities"])
""")],
            "向量相似不代表蕴含或事实支持；规范化后的 [0,1] 分数只用于同一配置下的比较。",
        ),
        "04_real_response_semantic_entropy_openai_local_nli.ipynb": document(
            "真实集成：响应级语义熵",
            "真实 API 采样完整回答，本地 NLI 完成双向等价判定、语义聚类和响应级语义熵。",
            "安装：`python3 -m pip install -e \".[openai,nli]\"`。",
            ["`LOCAL_NLI_MODEL: str`：标签含 entailment/contradiction 的本地 NLI 模型。", "`equivalence: str = 'official'`：官方三态等价规则。", "`base: float = 2.0`：熵单位（bits）。", "`NUM_SAMPLES: int = 5`：熵估计样本数。"],
            [("步骤 1：配置真实生成和 NLI", SETUP + OPENAI + NLI), ("步骤 2：运行语义熵", """
from hallucination.detection.blackbox.response_semantic_entropy import ResponseSemanticEntropyScorer
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
NUM_SAMPLES = 5
scorer = ResponseSemanticEntropyScorer(nli, include_response=True, equivalence="official", base=2.0)
result = BlackBoxSamplingPipeline(backend, scorers=[scorer], num_samples=NUM_SAMPLES).detect_prompt("请用两句话介绍蓝鲸的食性和分类。")
score = result.method_scores[0]
print(result.metadata["response"])
print(score.raw["semantic_entropy"], score.raw["num_semantic_sets"], score.raw["clusters"])
""")],
            "低熵只说明样本集中。稳定重复的错误依然可能低风险；NLI 的语言和领域有效性必须单独验证。",
        ),
        "06_real_selfcheck_bertscore_openai_local_model.ipynb": document(
            "真实集成：SelfCheck BERTScore",
            "真实 API 采样，再由 BERTScore 的实际模型计算每个原句的最佳候选 F1 相似度。",
            "安装：`python3 -m pip install -e \".[openai,selfcheck-bertscore]\"`；需要 API key 和 BERTScore 模型路径或模型标识。",
            ["`BERTSCORE_LANG: str = 'zh'`：BERTScore 文本语言。", "`BERTSCORE_MODEL: str`：本地路径或可下载模型标识。", "`rescale_with_baseline: bool = True`：是否进行基线重标定。", "`NUM_SAMPLES: int = 5`：每句样本数。"],
            [("步骤 1：配置真实 API 与 BERTScore", SETUP + OPENAI + """
from hallucination.detection.backends.embeddings import BERTScoreSimilarityBackend
BERTSCORE_LANG = "zh"
BERTSCORE_MODEL = "/absolute/path/to/your/bertscore-model"
similarity = BERTScoreSimilarityBackend(
    lang=BERTSCORE_LANG,
    model_type=BERTSCORE_MODEL,
    rescale_with_baseline=True,
)
"""), ("步骤 2：运行 scorer", """
from hallucination.detection.blackbox.selfcheck_bertscore import SelfCheckBERTScoreScorer
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
result = BlackBoxSamplingPipeline(backend, sentence_scorers=[SelfCheckBERTScoreScorer(similarity)], num_samples=5).detect_prompt("请用两句话介绍蓝鲸的食性和分类。")
for unit in result.units:
    print(unit.text, unit.method_scores[0].risk, unit.method_scores[0].raw["matches"])
""")],
            "BERTScore 相近不代表蕴含或事实正确。模型、语言、baseline 设置和版本必须在实验报告中固定。",
        ),
        "05_real_selfcheck_ngram_openai.ipynb": document(
            "真实 API：SelfCheck n-gram",
            "使用实际 API 样本运行不需要额外模型的 n-gram 自一致性基线。",
            "安装：`python3 -m pip install -e \".[openai]\"`；配置 API key 后会产生调用费用。",
            ["`n: int = 1`：n 元语法长度。", "`lowercase: bool = True`：英文大小写归一化。", "`NUM_SAMPLES: int = 5`：真实采样语料规模。", "`TEMPERATURE: float = 0.8`：API 采样多样性。"],
            [("步骤 1：配置真实 API", SETUP + OPENAI), ("步骤 2：运行 SelfCheck", """
from hallucination.detection.blackbox.selfcheck_ngram import SelfCheckNgramScorer
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
result = BlackBoxSamplingPipeline(backend, sentence_scorers=[SelfCheckNgramScorer(n=1, lowercase=True)], num_samples=5).detect_prompt("请用两句话介绍蓝鲸的食性和分类。")
for unit in result.units:
    print(unit.text, unit.method_scores[0].risk, unit.method_scores[0].raw)
""")],
            "n-gram 只衡量词面重合，对同义改写、否定和事实蕴含不鲁棒，适合作为低成本可解释基线。",
        ),
        "07_real_selfcheck_nli_openai_local_nli.ipynb": document(
            "真实集成：SelfCheck NLI",
            "真实 API 采样段落，本地 NLI 按官方顺序将原句作为 premise、采样段落作为 hypothesis，输出实际矛盾概率。",
            "安装：`python3 -m pip install -e \".[openai,nli]\"`；需要 API key 和本地 NLI 模型。",
            ["`LOCAL_NLI_MODEL: str`：本地 NLI 模型目录。", "`NUM_SAMPLES: int = 5`：每句独立样本数。", "`TEMPERATURE: float = 0.8`：采样多样性。"],
            [("步骤 1：配置真实 API 与 NLI", SETUP + OPENAI + NLI), ("步骤 2：运行 SelfCheck NLI", """
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
result = BlackBoxSamplingPipeline(backend, sentence_scorers=[SelfCheckNLIScorer(nli)], num_samples=5).detect_prompt("请用两句话介绍蓝鲸的食性和分类。")
for unit in result.units:
    print(unit.text, unit.method_scores[0].risk, unit.method_scores[0].raw["comparisons"])
""")],
            "NLI 输出是分类模型置信度而非事实真值；模型重复同一错误时，自一致性风险依然可能很低。",
        ),
        "08_real_factscore_local_nli.ipynb": document(
            "真实本地模型：FActScore 风格证据核验",
            "用真实本地 NLI、`LocalCorpusRetriever` 与 `FActScoreScorer` 对输入回答产生 claim-level 结论。",
            "安装：`python3 -m pip install -e \".[nli]\"`；打包证据语料仅覆盖教材事实。",
            ["`LOCAL_NLI_MODEL: str`：本地 NLI 模型路径。", "`support_threshold: float = 0.5`：支持判定阈值。", "`contradiction_threshold: float = 0.5`：矛盾判定阈值。", "`evidence_per_claim: int = 5`：每 claim 的候选证据数。"],
            [("步骤 1：配置真实 NLI", SETUP + NLI), ("步骤 2：建立证据管线", """
from hallucination.detection.evidence import FActScoreScorer, LocalCorpusRetriever, NLIEvidenceVerifier, load_textbook_evidence
from hallucination.detection.pipelines import EvidenceVerificationPipeline
pipeline = EvidenceVerificationPipeline(FActScoreScorer(NLIEvidenceVerifier(nli, support_threshold=0.5, contradiction_threshold=0.5), retriever=LocalCorpusRetriever(load_textbook_evidence()), evidence_per_claim=5))
"""), ("步骤 3：输入回答", """
response = input("回答文本：").strip()
claims_text = input("可选 claims，用 | 分隔；直接回车则按句切分：").strip()
claims = [item.strip() for item in claims_text.split("|") if item.strip()] or None
result = pipeline.detect(response=response, claims=claims)
for claim in result.claims:
    print(claim.label, claim.claim, claim.score, [(e.title, e.snippet) for e in claim.evidence])
print(result.method_scores[0].raw)
""")],
            "`unknown` 不等于 hallucination；它表示当前检索和 NLI 无法定论。生产部署必须替换为受控且可追溯的知识语料。",
        ),
        "09_real_long_form_openai_local_nli.ipynb": document(
            "真实集成：长回答的自一致性与证据支持",
            "API 真实生成/采样，本地 NLI 同时服务于 SelfCheck 和 FActScore，但两个信号独立返回。",
            "安装：`python3 -m pip install -e \".[openai,nli]\"`；需要 API key 和本地 NLI 模型。",
            ["`OPENAI_MODEL: str`：真实生成模型。", "`LOCAL_NLI_MODEL: str`：本地 NLI 模型。", "`NUM_SAMPLES: int = 5`：黑盒采样数。", "`claims: list[str] | None`：可选显式声明。"],
            [("步骤 1：配置真实后端", SETUP + OPENAI + NLI), ("步骤 2：建立分离 pipeline", """
from hallucination.detection.blackbox.selfcheck_nli import SelfCheckNLIScorer
from hallucination.detection.evidence import FActScoreScorer, LocalCorpusRetriever, NLIEvidenceVerifier, load_textbook_evidence
from hallucination.detection.pipelines import EvidenceVerificationPipeline, LongFormVerificationPipeline
from hallucination.detection.pipelines.short_answer import BlackBoxSamplingPipeline
sampling = BlackBoxSamplingPipeline(backend, sentence_scorers=[SelfCheckNLIScorer(nli)], num_samples=5)
evidence = EvidenceVerificationPipeline(FActScoreScorer(NLIEvidenceVerifier(nli), retriever=LocalCorpusRetriever(load_textbook_evidence())))
workflow = LongFormVerificationPipeline(sampling, evidence_pipeline=evidence)
"""), ("步骤 3：运行并分别读取", """
prompt = input("Prompt：").strip()
claims_text = input("可选 claims，用 | 分隔；直接回车则按句切分：").strip()
claims = [item.strip() for item in claims_text.split("|") if item.strip()] or None
result = workflow.detect_prompt(prompt, claims=claims)
print("response:", result.self_consistency.metadata["response"])
print("self consistency:", [(unit.text, unit.risk) for unit in result.self_consistency.units])
print("evidence support:", [(claim.claim, claim.label) for claim in result.evidence_support.claims])
""")],
            "稳定采样不能证明真实性，有限语料也不能证明完整事实。界面和决策层必须分别呈现两类结果。",
        ),
        "10_real_p_true_openai.ipynb": document(
            "真实 API：P(True) 自评置信度",
            "用同一模型的两种估计器评估自报置信度：Chat Completions logprob 路径读取 ` A`/` B` token 概率；Responses API 只能走 verbalized 回退。",
            "安装：`python3 -m pip install -e \".[openai]\"`；logprob 需要服务端支持 `logprobs=True`（OpenAI-compatible 本地服务也适用）。",
            ["`OPENAI_MODEL: str`：模型名。", "`use_samples_as_brainstorm: bool = True`：用采样作为 brainstorm 答案。", "`max_brainstorm: int = 4`：brainstorm 答案数上限。", "`normalize_ab: bool = True`：True/False 概率归一化；False 复现官方“只读 A”估计器。"],
            [("步骤 1：配置真实后端", SETUP + CHAT_OPENAI), ("步骤 2：构建两种估计器", """
from hallucination.detection import DetectionExample
from hallucination.detection.whitebox.p_true import PTrueScorer
logprob_scorer = PTrueScorer(chat_backend, use_samples_as_brainstorm=True, max_brainstorm=4, normalize_ab=True)
verbalized_scorer = PTrueScorer(responses_backend)
"""), ("步骤 3：交互运行并分别读取", """
prompt = input("Prompt：").strip()
samples = responses_backend.sample(prompt, 5)
example = DetectionExample(prompt=prompt, response=samples[0], samples=samples[1:])
for name, scorer in (("logprob", logprob_scorer), ("verbalized", verbalized_scorer)):
    result = scorer.score(example)
    print(name, "| method:", result.raw["method"], "| score:", result.score, "| risk:", result.risk)
    print("  details:", result.raw.get("token_details") or result.raw.get("fallback_reason"))
""")],
            "P(True) 是自我报告，普遍存在过度自信；两种估计器都未校准，比较时必须固定估计器类型。",
        ),
        "11_real_cove_detector_openai.ipynb": document(
            "真实 API：CoVe 验证链检测器",
            "真实生成基线回答与验证问答，统计与草稿冲突的验证答案比例；默认词面回退，接入 NLI 后按蕴含判定冲突。",
            "安装：`python3 -m pip install -e \".[openai]\"`（可选 `.[nli]` 提升冲突判定质量）。",
            ["`max_questions: int = 5`：验证问题数（1-10）。", "`contradiction_threshold: float = 0.5`：NLI 矛盾判定阈值。", "`min_similarity: float = 0.2`：词面回退的冲突判定阈值。", "`TEMPERATURE: float = 0.8`：采样温度。"],
            [("步骤 1：配置真实 API", SETUP + OPENAI), ("步骤 2：构建 CoVe 检测器", """
from hallucination.detection import DetectionExample
from hallucination.detection.blackbox.cove import CoVeScorer
cove = CoVeScorer(backend, max_questions=5)
"""), ("步骤 3：交互运行并读取冲突明细", """
prompt = input("Prompt：").strip()
response = input("待检回答（直接回车则由后端生成）：").strip() or backend.generate(prompt)
result = cove.score(DetectionExample(prompt=prompt, response=response))
print("verifier:", result.raw["verifier"], "| conflict_rate:", result.raw["conflict_rate"])
for item in result.raw["comparisons"]:
    print(item["question"], "->", item["verification_answer"], "| conflict:", item["conflict"])
""")],
            "CoVe 检测完全依赖模型自身知识，是自验证而非外部事实核查；它只报告冲突率，不重写答案。",
        ),
        "12_real_llm_judge_panel_openai.ipynb": document(
            "真实 API：LLM-as-a-Judge 与评判团",
            "用真实评判模型对同一回答输出 binary/continuous/Likert 三种裁决，并演示多评判官 panel 的平均风险。",
            "安装：`python3 -m pip install -e \".[openai]\"`；评判模型应与生成模型分离或至少独立调用。",
            ["`mode: str`：binary（忠实/幻觉+置信度）、continuous（0-100）、likert（1-5 忠实度）。", "`judge_name: str | None`：报告中的评判官名称。", "`LLMPanelScorer(judges)`：对多个独立 judge 的风险取平均。", "`TEMPERATURE: float = 0.8`：生成温度（评判本身用 temperature=0）。"],
            [("步骤 1：配置真实 API", SETUP + OPENAI), ("步骤 2：构建 judge 与 panel", """
from hallucination.detection import DetectionExample
from hallucination.detection.judges import LLMJudgeScorer, LLMPanelScorer
judges = {mode: LLMJudgeScorer(backend, mode=mode) for mode in ("binary", "continuous", "likert")}
panel = LLMPanelScorer([LLMJudgeScorer(backend, mode="binary"), LLMJudgeScorer(backend, mode="likert")])
"""), ("步骤 3：交互运行并分别读取", """
prompt = input("Prompt：").strip()
response = input("待评回答（直接回车则由后端生成）：").strip() or backend.generate(prompt)
example = DetectionExample(prompt=prompt, response=response)
for mode, judge in judges.items():
    result = judge.score(example)
    print(mode, "| risk:", result.risk, "| parsed:", result.raw.get("parsed"))
panel_result = panel.score(example)
print("panel | risk:", panel_result.risk, "| judges:", panel_result.raw["num_judges"])
""")],
            "judge 输出是另一个模型的意见，不是事实概率；同一模型自评存在偏置，panel 平均也无法消除系统性偏差。",
        ),
    }


def main() -> None:
    for filename, data in documents().items():
        path = OUTPUT / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
