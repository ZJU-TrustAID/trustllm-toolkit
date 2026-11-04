from pathlib import Path

import pytest

from hallucination.detection.core.config import PipelineConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "detection"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_every_shipped_config_loads(path):
    config = PipelineConfig.from_yaml(path)
    assert config.name
    assert config.enabled_scorers()


def test_long_form_config_uses_response_level_cosine_and_entropy():
    config = PipelineConfig.from_yaml(CONFIG_DIR / "blackbox_long_form.yaml")
    names = config.enabled_scorers()
    assert "response_cosine_consistency" in names
    assert "response_semantic_entropy" in names
    assert "sentence_cosine_consistency" not in names
    assert "sentence_semantic_entropy" not in names
