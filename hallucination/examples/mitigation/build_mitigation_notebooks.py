"""Generate opt-in notebooks for real-model mitigation integrations."""

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

HF_LOCAL = """
LOCAL_MODEL = "/absolute/path/to/your/causal-lm"   # e.g. a local Qwen2.5-1.5B-Instruct directory
DEVICE = "cpu"     # "cuda" when a GPU is available
DTYPE = "auto"
MAX_NEW_TOKENS = 64
if not Path(LOCAL_MODEL).is_dir():
    raise ValueError("Set LOCAL_MODEL to an existing local model directory (download it first).")
"""

OPENAI_MIT = """
import os
from getpass import getpass
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_BACKEND = "openai"                     # or "openai-compatible" for local vLLM/Ollama servers
api_key = os.environ.get("OPENAI_API_KEY") or getpass("OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key        # the Mitigator reads the key from this environment variable
API_BASE = os.environ.get("OPENAI_BASE_URL")  # optional: point at an OpenAI-compatible server
"""


def document(title: str, intro: str, install: str, parameters: list[str], sections: list[tuple[str, str]], limitation: str) -> dict[str, object]:
    cells = [
        cell("markdown", f"# {title}\n\n{intro}\n\n## 真实集成前提\n\n{install}\n\n代码单元没有预执行。填写本地模型路径或 API 凭据后，从上到下运行；不要将 API key 保存到 notebook。缓解会改变模型输出，务必同时阅读基线与缓解后的回答。"),
        cell("markdown", "## 参数说明\n\n" + "\n".join(f"- {item}" for item in parameters)),
    ]
    for heading, snippet in sections:
        cells.extend([cell("markdown", "## " + heading), cell("code", snippet)])
    cells.append(cell("markdown", "## 结果解读与限制\n\n" + limitation))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


def documents() -> dict[str, dict[str, object]]:
    return {
        "00_dola_local_model.ipynb": document(
            "真实本地模型：DoLa 层间对比解码",
            "在真实本地因果模型上运行 DoLa：每个解码步按 Jensen-Shannon 散度动态选择早期层，放大深层事实信号。报告同时给出常规解码基线与缓解后回答。",
            "安装：`python3 -m pip install -e \".[hf]\"`；需要已下载的本地模型目录。CLI 等价命令：`hallucination-mitigation run --method dola --model <LOCAL_MODEL> --prompt ...`。",
            ["`LOCAL_MODEL: str`：本地 causal LM 目录（Llama/Qwen 系）。", "`relative_top: float = 0.1`：自适应合理性约束阈值，过滤低概率候选。", "`mature_layer / candidate_premature_layers`：成熟层与候选早期层桶；缺省按层数自动选取。", "`MAX_NEW_TOKENS: int = 64`：生成长度。"],
            [("步骤 1：配置真实模型", SETUP + HF_LOCAL), ("步骤 2：构建 Mitigator", """
from hallucination.mitigation import Mitigator
mitigator = Mitigator(
    method="dola",
    model=LOCAL_MODEL,
    device=DEVICE,
    dtype=DTYPE,
    local_files_only=True,
    relative_top=0.1,
    baseline=True,
)
"""), ("步骤 3：交互运行并读取", """
prompt = input("Prompt：").strip()
report = mitigator.mitigate(prompt, max_new_tokens=MAX_NEW_TOKENS)
print("baseline :", report.baseline)
print("mitigated:", report.response)
print("premature layer distribution:", report.trace["premature_layer_distribution"])
print("warnings:", report.warnings)
""")],
            "DoLa 只改变解码分布，不保证事实正确；层桶是任务相关的预置参数，必须在自己的模型与数据上验证，真实冒烟中出现过把正确基线改错的情况。",
        ),
        "01_cad_context_conflict.ipynb": document(
            "真实本地模型：CAD 上下文感知解码",
            "在知识冲突场景下运行 CAD：对每个 token 同时维护“有上下文/无上下文”两条 KV cache，按 (1+α)·logit_ctx − α·logit_no_ctx 组合，抑制与上下文冲突的参数化先验。",
            "安装：`python3 -m pip install -e \".[hf]\"`。CAD 需要显式上下文；每个 token 两次前向，延迟与显存近似翻倍。CLI：`hallucination-mitigation run --method cad --context ... --prompt ...`。",
            ["`alpha: float`：对比强度，0 退化为常规解码，越大越依赖上下文。", "`context_template: str = 'plain'`：上下文拼接模板（plain/chat）。", "`context`：与模型先验冲突的上下文事实，运行时交互输入。", "`MAX_NEW_TOKENS: int = 64`：生成长度。"],
            [("步骤 1：配置真实模型", SETUP + HF_LOCAL), ("步骤 2：构建 Mitigator", """
from hallucination.mitigation import Mitigator
mitigator = Mitigator(
    method="cad",
    model=LOCAL_MODEL,
    device=DEVICE,
    dtype=DTYPE,
    local_files_only=True,
    alpha=1.0,
    context_template="plain",
    baseline=True,
)
"""), ("步骤 3：交互运行并读取", """
context = input("上下文事实（与模型先验冲突时效果最明显，如“阿根廷在 1978、1986 和 2022 年赢得世界杯”）：").strip()
prompt = input("问题：").strip()
report = mitigator.mitigate(prompt, context=context, max_new_tokens=32)
print("baseline :", report.baseline)
print("mitigated:", report.response)
print("warnings:", report.warnings)
""")],
            "CAD 只在上下文与先验冲突时有收益；无冲突场景可能过度修正。α 需要按任务验证，报告中的 KL 轨迹可辅助诊断。",
        ),
        "02_caa_fit_and_run.ipynb": document(
            "真实本地模型：CAA 激活转向（拟合与运行）",
            "先用对比问答对拟合 CAA 转向向量（正负激活差的均值）并保存工件；再加载工件，在生成时把该向量加到指定层的输出上。",
            "安装：`python3 -m pip install -e \".[hf]\"`。拟合需要 `prompt/positive/negative` 对比数据；本教程用仓库自带演示数据（仅教材示例规模，真实使用请准备领域数据）。",
            ["`layer: int = 12`：注入转向向量的层。", "`multiplier: float = 1.0`：转向强度。", "`pooling: str = 'last'`：拟合时序列激活的池化方式（last/mean）。", "`chat=True`：数据按 chat 模板渲染。"],
            [("步骤 1：配置真实模型与拟合数据", SETUP + HF_LOCAL + """
import json
DATA = repo_root / "examples" / "mitigation" / "data" / "hallucination_pairs_demo.jsonl"
ARTIFACT = repo_root / "runs" / "caa_demo.npz"
pairs = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
print("pairs:", len(pairs), "| first keys:", sorted(pairs[0]))
"""), ("步骤 2：拟合并保存工件", """
from hallucination.mitigation.core.model_config import ModelConfig
from hallucination.mitigation.engine.hf import HFEngine
from hallucination.mitigation.methods.caa import CAAConfig, fit_caa
engine = HFEngine.from_pretrained(ModelConfig(model=LOCAL_MODEL, device=DEVICE, local_files_only=True))
artifact = fit_caa(engine, pairs, CAAConfig(layer=12, multiplier=1.0, chat=True, pooling="last"))
ARTIFACT.parent.mkdir(exist_ok=True)
artifact.save(str(ARTIFACT))
print("artifact:", ARTIFACT, "| layer:", artifact.layer)
"""), ("步骤 3：加载工件交互运行", """
from hallucination.mitigation import Mitigator
mitigator = Mitigator(
    method="caa",
    model=LOCAL_MODEL,
    device=DEVICE,
    dtype=DTYPE,
    local_files_only=True,
    steering=str(ARTIFACT),
    layer=12,
    multiplier=1.0,
    baseline=True,
)
prompt = input("Prompt：").strip()
report = mitigator.mitigate(prompt, max_new_tokens=MAX_NEW_TOKENS)
print("baseline :", report.baseline)
print("mitigated:", report.response)
print("warnings:", report.warnings)
""")],
            "CAA 可能过度修正（真实冒烟中把“小型甲壳类”改成“大型甲壳类”）；工件与模型结构绑定，跨模型复用工件会被拒绝；演示数据规模仅够跑通流程。",
        ),
        "03_iti_fit_and_run.ipynb": document(
            "真实本地模型：ITI 逐头探针转向（拟合与运行）",
            "先为每个注意力头训练逻辑回归探针，按验证准确率选出 top-K 个“真实性方向”并保存工件；生成时沿 α·σ·θ 偏移这些头的输出。",
            "安装：`python3 -m pip install -e \".[hf,iti]\"`。拟合需要 `prompt/response/label` 标注数据（label 1=真实、0=虚假）；`bake=True` 在 v0.1 明确不支持。",
            ["`top_k: int = 48`：干预的注意力头数（论文默认 48）。", "`alpha: float = 15.0`：偏移强度（以投影标准差为单位，论文默认 15）。", "`chat=True`：标注数据按 chat 模板渲染。", "`MAX_NEW_TOKENS: int = 64`：生成长度。"],
            [("步骤 1：配置真实模型与标注数据", SETUP + HF_LOCAL + """
import json
DATA = repo_root / "examples" / "mitigation" / "data" / "iti_demo.jsonl"
ARTIFACT = repo_root / "runs" / "iti_demo.json"
samples = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
print("samples:", len(samples), "| first keys:", sorted(samples[0]))
"""), ("步骤 2：拟合并保存探针工件", """
from hallucination.mitigation.core.model_config import ModelConfig
from hallucination.mitigation.engine.hf import HFEngine
from hallucination.mitigation.methods.iti import ITIConfig, fit_iti
engine = HFEngine.from_pretrained(ModelConfig(model=LOCAL_MODEL, device=DEVICE, local_files_only=True))
artifact = fit_iti(engine, samples, ITIConfig(top_k=48, alpha=15.0, chat=True))
ARTIFACT.parent.mkdir(exist_ok=True)
artifact.save(str(ARTIFACT))
print("artifact:", ARTIFACT, "| heads:", len(artifact.heads), "| best accuracy:", artifact.metadata.get("best_accuracy"))
"""), ("步骤 3：加载探针交互运行", """
from hallucination.mitigation import Mitigator
mitigator = Mitigator(
    method="iti",
    model=LOCAL_MODEL,
    device=DEVICE,
    dtype=DTYPE,
    local_files_only=True,
    probe=str(ARTIFACT),
    baseline=True,
)
prompt = input("Prompt：").strip()
report = mitigator.mitigate(prompt, max_new_tokens=MAX_NEW_TOKENS)
print("baseline :", report.baseline)
print("mitigated:", report.response)
print("warnings:", report.warnings)
""")],
            "仓库自带的 iti_demo.jsonl 只有十余条样本，探针准确率接近随机——真实使用必须准备数百条以上带真/假标签的问答对；探针工件与模型结构绑定。",
        ),
        "04_cove_api.ipynb": document(
            "真实 API：CoVe 验证链修正",
            "CoVe 是唯一支持 API 后端的缓解方法：生成基线回答，规划验证问题，独立作答，最后综合生成修正回答。四阶段全部走真实 API。",
            "安装：`python3 -m pip install -e \".[openai]\"`并配置 `OPENAI_API_KEY`；`OPENAI_BASE_URL` 可指向任意 OpenAI-compatible 服务（vLLM/Ollama）。会产生多次真实调用费用。",
            ["`OPENAI_MODEL: str`：API 模型名。", "`max_questions: int = 6`：验证问题数。", "`mode: str = 'factored'`：factored（独立作答）或 joint（联合作答）。", "`temperature: float = 0.0`：各阶段采样温度。"],
            [("步骤 1：配置真实 API", SETUP + OPENAI_MIT), ("步骤 2：构建 Mitigator", """
from hallucination.mitigation import Mitigator
mitigator = Mitigator(
    method="cove",
    backend=OPENAI_BACKEND,
    model=OPENAI_MODEL,
    api_base=API_BASE,
    api_key_env="OPENAI_API_KEY",
    max_questions=5,
    mode="factored",
    temperature=0.0,
    baseline=True,
)
"""), ("步骤 3：交互运行并读取", """
prompt = input("Prompt：").strip()
report = mitigator.mitigate(prompt, max_new_tokens=256)
print("baseline :", report.baseline)
print("revised  :", report.response)
print("warnings:", report.warnings)
""")],
            "CoVe 用同一模型自问自答，是自验证而非外部核查，验证答案本身可能也是幻觉；真实冒烟中还出现过回答语言漂移（中文提问英文作答），生产使用必须固定输出语言并抽查验证答案。",
        ),
    }


def main() -> None:
    for filename, data in documents().items():
        path = OUTPUT / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
