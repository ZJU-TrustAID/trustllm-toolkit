"""Unified run YAML validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from trustllm_hub.schema import ConfigError, validate_config


def base_config() -> dict:
    return {
        "version": 1,
        "chapter": 6,
        "toolkit": "hallucination",
        "common": {"device": "cuda:0", "dtype": "bfloat16", "seed": 42},
        "models": {"default": {"source": "local_path", "path": "~/models/Qwen2.5-7B-Instruct"}},
        "chapter_params": {"task": "mitigation", "method": "dola"},
    }


def test_valid_config_normalizes_paths(tmp_path: Path) -> None:
    config = validate_config(base_config())
    path = config["models"]["default"]["path"]
    assert path.startswith(str(Path.home())) and path.endswith("Qwen2.5-7B-Instruct")
    assert config["common"] == {"device": "cuda:0", "dtype": "bfloat16", "seed": 42}


def test_relative_path_resolves_against_base(tmp_path: Path) -> None:
    config = base_config()
    config["models"]["default"] = {"source": "local_path", "path": "models/qwen"}
    normalized = validate_config(config, base=tmp_path)
    assert normalized["models"]["default"]["path"] == str((tmp_path / "models" / "qwen").resolve())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(version=2),
        lambda c: c.update(toolkit=""),
        lambda c: c.update(unknown_key=1),
        lambda c: c["common"].update(dtype="fp8"),
        lambda c: c["common"].update(seed=-1),
        lambda c: c.update(models={}),
        lambda c: c["models"]["default"].update(source="magic"),
        lambda c: c["models"]["default"].update(source="local_path", path="  "),
        lambda c: c["models"]["default"].update(source="hf", repo="not-a-repo"),
        lambda c: c["models"]["default"].update(source="api", api={"base_url": "ftp://x", "model": "m"}),
        lambda c: c["models"]["default"].update(source="api", api={"base_url": "http://x", "model": ""}),
        lambda c: c.update(chapter_params=[1, 2]),
    ],
)
def test_invalid_configs_raise(mutation) -> None:
    config = base_config()
    mutation(config)
    with pytest.raises(ConfigError):
        validate_config(config)


def test_api_entry_normalizes_key_env() -> None:
    config = base_config()
    config["models"]["default"] = {
        "source": "api",
        "api": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    }
    normalized = validate_config(config)
    assert normalized["models"]["default"]["api"]["api_key_env"] == "OPENAI_API_KEY"


def test_chapter_accepts_range_string() -> None:
    config = base_config()
    config["chapter"] = "4-5"
    assert validate_config(config)["chapter"] == "4-5"


def test_null_chapter_params_is_yaml_empty_mapping() -> None:
    config = base_config()
    config["chapter_params"] = None  # YAML `chapter_params:` with no value parses to None
    assert validate_config(config)["chapter_params"] == {}
