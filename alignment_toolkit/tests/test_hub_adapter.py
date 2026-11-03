"""Hub adapter config composition (no GPU / no network: resolve only)."""
from __future__ import annotations

import json
from pathlib import Path

from alignment_toolkit import load_experiment
from alignment_toolkit.hub_adapter import build_experiment_config, _evaluation_suites


def make_hub_config(tmp_path: Path, workflow: str, **params) -> dict:
    params.setdefault("workflow", workflow)
    params.setdefault("evaluate", True)
    return {
        "version": 1,
        "chapter": 4,
        "toolkit": "alignment",
        "common": {"device": "auto", "dtype": "auto", "seed": 7, "output_dir": str(tmp_path)},
        "models": {"base": {"source": "hf", "repo": "Qwen/Qwen2.5-1.5B-Instruct"}},
        "chapter_params": params,
    }, tmp_path


def resolve(tmp_path: Path, workflow: str, **params):
    hub_config, out_dir = make_hub_config(tmp_path, workflow, **params)
    experiment = build_experiment_config(hub_config, out_dir / "session")
    experiment_path = out_dir / "experiment.yaml"
    import yaml

    experiment_path.write_text(yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_experiment(experiment_path), experiment


def test_sft_resolves(tmp_path: Path) -> None:
    spec, _ = resolve(tmp_path, "sft")
    assert [stage.stage_id for stage in spec.stages] == ["sft"]
    stage = spec.stages[0]
    assert stage.method == "sft" and stage.backend == "trl"
    assert spec.models["base"]["id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert spec.output["root"] == str((tmp_path / "session" / "training").resolve())


def test_dpo_resolves_with_reference(tmp_path: Path) -> None:
    spec, _ = resolve(tmp_path, "dpo", lora=True, beta=0.05)
    assert [stage.stage_id for stage in spec.stages] == ["dpo"]
    assert set(spec.models) == {"base", "policy", "reference"}
    training = spec.stages[0].training
    assert training["beta"] == 0.05
    assert training["adapter_disabled"] is True and training["merge_lora"] is True


def test_dpo_without_lora(tmp_path: Path) -> None:
    spec, _ = resolve(tmp_path, "dpo", lora=False)
    training = spec.stages[0].training
    assert "lora" not in training and "adapter_disabled" not in training


def test_rlhf_three_stages(tmp_path: Path) -> None:
    spec, _ = resolve(tmp_path, "rlhf", ppo_episodes=2, kl_coef=0.2, evaluate=False)
    assert [stage.stage_id for stage in spec.stages] == ["sft", "reward_model", "ppo"]
    reward, ppo = spec.stages[1], spec.stages[2]
    assert reward.input == {"model": "sft"}
    assert ppo.input == {"policy": "sft", "reward_model": "reward_model"}
    assert ppo.training["total_episodes"] == 2 and ppo.training["kl_coef"] == 0.2


def test_rlhf_seed_propagates(tmp_path: Path) -> None:
    spec, _ = resolve(tmp_path, "rlhf", evaluate=False)
    assert spec.stages[0].training["seed"] == 7


def test_evaluation_suites_attached(tmp_path: Path) -> None:
    spec, _ = resolve(tmp_path, "sft", evaluate=True)
    suites = spec.evaluation["suites"]
    assert suites, "evaluation suites must be attached when evaluate=true"
    assert all(suite["backend"] == "refusal_heuristic" for suite in suites)
    assert spec.evaluation["model"]["stage"] == "sft"


def test_evaluation_skipped_when_disabled(tmp_path: Path) -> None:
    spec, _ = resolve(tmp_path, "sft", evaluate=False)
    assert spec.evaluation["suites"] == []


def test_local_json_dataset(tmp_path: Path) -> None:
    data = tmp_path / "sft.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8")
    spec, _ = resolve(tmp_path, "sft", sft_dataset=str(data))
    definition = spec.raw["resolved_datasets"]["sft"]
    assert definition["source"]["type"] == "json"
    assert definition["source"]["path"] == str(data.resolve())


def test_invalid_workflow_rejected(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        resolve(tmp_path, "gan")


def test_evaluation_suites_fallback() -> None:
    suites = _evaluation_suites({"profile": "nonexistent_profile"})
    assert suites and suites[0]["backend"] == "refusal_heuristic"
