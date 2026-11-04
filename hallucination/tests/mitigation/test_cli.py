from __future__ import annotations

from hallucination.mitigation.cli import main


def test_list_methods() -> None:
    assert main(["list-methods"]) == 0


def test_list_presets() -> None:
    assert main(["list-presets"]) == 0


def test_doctor() -> None:
    assert main(["doctor"]) == 0


def test_unknown_preset_returns_error() -> None:
    assert main(["run", "--preset", "missing", "--prompt", "Q?"]) == 2
