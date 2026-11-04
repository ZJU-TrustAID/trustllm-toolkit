import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "examples" / "mitigation"

EXPECTED_NOTEBOOKS = {
    "00_dola_local_model.ipynb",
    "01_cad_context_conflict.ipynb",
    "02_caa_fit_and_run.ipynb",
    "03_iti_fit_and_run.ipynb",
    "04_cove_api.ipynb",
}


def test_mitigation_notebooks_are_opt_in_real_backend_tutorials():
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
