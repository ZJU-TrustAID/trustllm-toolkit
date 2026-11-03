from pathlib import Path
from dataclasses import dataclass

import yaml

from alignment_toolkit import load_experiment
from alignment_toolkit.evaluation import (
    SafetySample,
    is_refusal,
    load_evaluation_profile,
    resolve_evaluation_model,
    summarize_safety,
)
from alignment_toolkit.trl_runner import make_training_args, prepare_training_options


def test_rlhf_pipeline_resolves_from_relative_config():
    config = Path("configs/experiments/rlhf.yaml")
    spec = load_experiment(config)
    assert [stage.method for stage in spec.stages] == ["sft", "reward_model", "ppo"]
    assert [stage.backend for stage in spec.stages] == ["trl", "trl", "trl"]
    assert spec.stages[-1].dataset == "ultrachat_prompts"


def test_experiment_registry_paths_resolve_from_any_working_directory(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = project_root / "configs/experiments/sft.yaml"
    monkeypatch.chdir(tmp_path)

    spec = load_experiment(config)

    assert spec.stages[0].dataset == "ultrachat_sft"
    assert spec.raw["resolved_datasets"]["ultrachat_sft"]["source"]["type"] == "huggingface"


def test_dataset_registry_has_real_huggingface_sources():
    raw = yaml.safe_load(Path("configs/datasets.yaml").read_text(encoding="utf-8"))
    assert raw["datasets"]["ultrafeedback_dpo"]["source"]["id"] == "HuggingFaceH4/ultrafeedback_binarized"


def test_dpo_and_sft_examples_resolve():
    assert load_experiment(Path("configs/experiments/sft.yaml")).stages[0].backend == "trl"
    assert load_experiment(Path("configs/experiments/dpo.yaml")).stages[0].method == "dpo"


def test_example_models_use_huggingface_revision():
    project_root = Path(__file__).resolve().parents[1]
    for filename in ("sft.yaml", "dpo.yaml", "rlhf.yaml"):
        raw = yaml.safe_load((project_root / "configs/experiments" / filename).read_text(encoding="utf-8"))
        assert raw["models"]
        assert all(model["source"] == "huggingface" for model in raw["models"].values())
        assert all(model["revision"] == "main" for model in raw["models"].values())


def test_trl_training_options_translate_epochs(tmp_path):
    @dataclass
    class Args:
        output_dir: str
        num_train_epochs: float = 1
        learning_rate: float = 1e-5

    args = make_training_args(
        Args,
        {"epochs": 2, "learning_rate": 2e-5},
        tmp_path / "sft",
    )

    assert args.num_train_epochs == 2
    assert args.learning_rate == 2e-5
    assert args.output_dir.endswith("/sft")


def test_adapter_disabled_is_runner_option():
    values, peft_config, adapter_disabled, merge_lora = prepare_training_options(
        {"beta": 0.1, "adapter_disabled": True}, "dpo"
    )

    assert values == {"beta": 0.1}
    assert peft_config is None
    assert adapter_disabled is True
    assert merge_lora is False


def test_rlhf_reward_model_uses_lora_and_merges():
    spec = load_experiment(Path("configs/experiments/rlhf.yaml"))
    reward_stage = next(stage for stage in spec.stages if stage.stage_id == "reward_model")

    assert reward_stage.training["lora"]["modules_to_save"] == ["score"]
    assert reward_stage.training["merge_lora"] is True


def test_safety_profile_and_refusal_summary():
    profile = load_evaluation_profile("configs/experiments/rlhf.yaml")
    assert profile["name"] == "safety_alignment"
    assert {suite["name"] for suite in profile["suites"]} == {
        "harmbench", "strongreject", "jailbreakbench", "xstest"
    }
    samples = [
        SafetySample("unsafe", "I cannot help with that.", "unsafe", True),
        SafetySample("unsafe", "Here are the instructions.", "unsafe", False),
        SafetySample("safe", "Sure, here is a summary.", "safe", False),
    ]
    assert is_refusal(samples[0].response)
    summary = summarize_safety(samples)
    assert summary["attack_success_rate"] == 0.5
    assert summary["over_refusal_rate"] == 0.0


def test_inline_evaluation_model_resolves_stage_output():
    config = Path("configs/experiments/rlhf.yaml")
    spec = load_experiment(config)
    model_path, model_kwargs = resolve_evaluation_model(spec, config)

    assert model_path.endswith("/models/rlhf/ppo")
    assert model_kwargs == {}
