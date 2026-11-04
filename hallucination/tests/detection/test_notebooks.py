import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "examples" / "detection"

EXPECTED_NOTEBOOKS = {
    "00_real_whitebox_token_signals_local.ipynb",
    "01_real_whitebox_hidden_states_local.ipynb",
    "02_real_exact_match_consistency_openai.ipynb",
    "03_real_response_cosine_openai_local_embedding.ipynb",
    "04_real_response_semantic_entropy_openai_local_nli.ipynb",
    "05_real_selfcheck_ngram_openai.ipynb",
    "06_real_selfcheck_bertscore_openai_local_model.ipynb",
    "07_real_selfcheck_nli_openai_local_nli.ipynb",
    "08_real_factscore_local_nli.ipynb",
    "09_real_long_form_openai_local_nli.ipynb",
    "10_real_p_true_openai.ipynb",
    "11_real_cove_detector_openai.ipynb",
    "12_real_llm_judge_panel_openai.ipynb",
}


def test_real_backend_notebooks_are_complete_and_opt_in():
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == EXPECTED_NOTEBOOKS
    for path in NOTEBOOKS.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "参数说明" in markdown
        assert "结果解读与限制" in markdown
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        code_text = "\n".join("".join(cell["source"]) for cell in code_cells)
        assert "LOCAL_" in code_text or "OPENAI_API_KEY" in code_text
        assert all(isinstance(cell["outputs"], list) for cell in code_cells)
        assert all(cell["execution_count"] is None for cell in code_cells)
