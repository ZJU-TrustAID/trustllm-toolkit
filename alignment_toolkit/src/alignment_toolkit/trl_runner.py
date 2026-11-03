"""TRL-backed training runner for SFT, DPO, reward modeling, and PPO.

The module intentionally imports torch, transformers, datasets, and TRL only
when a training run starts. This keeps config resolution and ``dry-run`` useful
in lightweight environments.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any


class TRLRunner:
    """Run a resolved pipeline with Hugging Face TRL trainers."""

    def __init__(self, *, project_root: str | Path = "."):
        self.project_root = Path(project_root).resolve()

    def run(self, spec):
        try:
            import torch  # noqa: F401
            from datasets import load_dataset
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
            from trl import DPOConfig, DPOTrainer, PPOConfig, PPOTrainer, RewardConfig, RewardTrainer, SFTConfig, SFTTrainer
        except ImportError as exc:  # pragma: no cover - depends on optional train extra
            raise RuntimeError(
                "TRL training dependencies are missing; install with "
                "python -m pip install -e '.[train]'"
            ) from exc

        trainers = {
            "sft": (SFTConfig, SFTTrainer),
            "dpo": (DPOConfig, DPOTrainer),
            "reward_model": (RewardConfig, RewardTrainer),
            "ppo": (PPOConfig, PPOTrainer),
        }
        artifacts: dict[str, dict[str, Any]] = {}
        results = []

        for stage in spec.stages:
            if stage.backend != "trl":
                raise ValueError(
                    f"Stage {stage.stage_id!r} uses backend {stage.backend!r}; "
                    "TRLRunner only accepts backend='trl'"
                )
            if stage.method not in trainers:
                raise ValueError(f"TRL does not have a trainer for method {stage.method!r}")

            dataset_definition = spec.raw.get("resolved_datasets", {}).get(stage.dataset)
            if dataset_definition is None:
                raise ValueError(f"Resolved dataset definition is missing for {stage.dataset!r}")
            dataset = load_dataset_from_definition(dataset_definition, load_dataset)
            output_dir = self._output_dir(spec, stage)
            output_dir.mkdir(parents=True, exist_ok=True)
            training, peft_config, adapter_disabled, merge_lora = prepare_training_options(stage.training, stage.method)

            if stage.method == "sft":
                artifact = self._run_sft(
                    spec, stage, dataset, output_dir, training,
                    AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer, peft_config, merge_lora,
                )
            elif stage.method == "dpo":
                artifact = self._run_dpo(
                    spec, stage, dataset, output_dir, training, artifacts,
                    AutoModelForCausalLM, AutoTokenizer, DPOConfig, DPOTrainer, peft_config, adapter_disabled, merge_lora,
                )
            elif stage.method == "reward_model":
                artifact = self._run_reward_model(
                    spec, stage, dataset, output_dir, training, artifacts,
                    AutoModelForSequenceClassification, AutoTokenizer, RewardConfig, RewardTrainer, peft_config, merge_lora,
                )
            else:
                artifact = self._run_ppo(
                    spec, stage, dataset, output_dir, training, artifacts,
                    AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PPOConfig, PPOTrainer,
                    peft_config, adapter_disabled, merge_lora,
                )

            artifacts[stage.stage_id] = artifact
            results.append({"stage": stage.stage_id, **artifact})

        return {"pipeline": spec.name, "stages": results}

    def _output_dir(self, spec, stage) -> Path:
        configured = stage.training.get("output_dir")
        if configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = self.project_root / configured_path
            return configured_path.resolve()
        root = Path(spec.output.get("root", "outputs")).expanduser()
        return (self.project_root / root / spec.name / stage.stage_id).resolve()

    def _run_sft(
        self, spec, stage, dataset, output_dir, training,
        model_cls, tokenizer_cls, config_cls, trainer_cls, peft_config, merge_lora,
    ):
        model_ref = self._input_model(spec, stage, {})
        tokenizer = load_tokenizer(tokenizer_cls, model_ref)
        dataset = prepare_sft_dataset(dataset, stage, tokenizer)
        args = make_training_args(config_cls, training, output_dir)
        trainer = trainer_cls(
            model=load_causal_model(model_cls, model_ref),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        save_trained_model(trainer, tokenizer, output_dir, merge_lora=merge_lora)
        return {"method": stage.method, "model_path": str(output_dir)}

    def _run_dpo(
        self, spec, stage, dataset, output_dir, training, artifacts,
        model_cls, tokenizer_cls, config_cls, trainer_cls, peft_config, adapter_disabled, merge_lora,
    ):
        model_ref = self._input_model(spec, stage, artifacts)
        reference_ref = self._configured_model(spec, "reference")
        if adapter_disabled and peft_config is None:
            raise ValueError("training.adapter_disabled requires a training.lora configuration")
        tokenizer = load_tokenizer(tokenizer_cls, model_ref)
        dataset = prepare_preference_dataset(dataset, stage)
        args = make_training_args(config_cls, training, output_dir)
        trainer = trainer_cls(
            model=load_causal_model(model_cls, model_ref),
            ref_model=None if adapter_disabled else (load_causal_model(model_cls, reference_ref) if reference_ref else None),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        save_trained_model(trainer, tokenizer, output_dir, merge_lora=merge_lora)
        return {"method": stage.method, "model_path": str(output_dir)}

    def _run_reward_model(
        self, spec, stage, dataset, output_dir, training, artifacts,
        model_cls, tokenizer_cls, config_cls, trainer_cls, peft_config, merge_lora,
    ):
        model_ref = self._input_model(spec, stage, artifacts)
        tokenizer = load_tokenizer(tokenizer_cls, model_ref)
        dataset = prepare_preference_dataset(dataset, stage, tokenizer=tokenizer)
        args = make_training_args(config_cls, training, output_dir)
        trainer = trainer_cls(
            model=load_reward_model(model_cls, model_ref),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        save_trained_model(trainer, tokenizer, output_dir, merge_lora=merge_lora)
        return {"method": stage.method, "model_path": str(output_dir)}

    def _run_ppo(
        self, spec, stage, dataset, output_dir, training, artifacts,
        causal_cls, sequence_cls, tokenizer_cls, config_cls, trainer_cls,
        peft_config, adapter_disabled, merge_lora,
    ):
        model_ref = self._input_model(spec, stage, artifacts, input_key="policy")
        reward_ref = self._input_artifact_model(stage.input.get("reward_model"), artifacts)
        reference_ref = self._configured_model(spec, "reference")
        if adapter_disabled and peft_config is None:
            raise ValueError("training.adapter_disabled requires a training.lora configuration")
        if reward_ref is None:
            raise ValueError("PPO requires input.reward_model to reference a reward_model stage")

        tokenizer = load_tokenizer(tokenizer_cls, model_ref)
        dataset = prepare_prompt_dataset(dataset, stage, tokenizer)
        args = make_training_args(config_cls, training, output_dir)
        trainer = trainer_cls(
            args=args,
            processing_class=tokenizer,
            model=load_causal_model(causal_cls, model_ref),
            ref_model=None if adapter_disabled else (load_causal_model(causal_cls, reference_ref) if reference_ref else None),
            reward_model=load_reward_model(sequence_cls, reward_ref),
            value_model=load_reward_model(sequence_cls, model_ref, for_value=True),
            train_dataset=dataset,
            eval_dataset=dataset,
            peft_config=peft_config,
        )
        trainer.train()
        save_trained_model(trainer, tokenizer, output_dir, merge_lora=merge_lora)
        return {"method": stage.method, "model_path": str(output_dir)}

    def _configured_model(self, spec, role):
        reference = spec.models.get(role) or spec.models.get("base")
        if reference is None:
            raise ValueError(f"Model role {role!r} is not defined in models")
        return reference

    def _input_artifact_model(self, name, artifacts):
        if not name:
            return None
        artifact = artifacts.get(name)
        if not artifact:
            raise ValueError(f"Stage input references unknown artifact {name!r}")
        return artifact["model_path"]

    def _input_model(self, spec, stage, artifacts, *, input_key="model"):
        referenced = stage.input.get(input_key)
        if referenced:
            if referenced in artifacts:
                return self._input_artifact_model(referenced, artifacts)
            if referenced in spec.models:
                return spec.models[referenced]
            raise ValueError(f"Stage input references unknown model or artifact {referenced!r}")
        return self._configured_model(spec, "policy") or self._configured_model(spec, "base")


def load_dataset_from_definition(definition, load_dataset):
    source = dict(definition["source"])
    source_type = source.get("type", "huggingface")
    if source_type == "json":
        dataset = load_dataset("json", data_files=source["path"], split=source.get("split", "train"))
        return normalise_dataset(dataset, definition.get("mapping") or {})
    if source_type != "huggingface":
        raise ValueError(f"Unsupported dataset source type: {source_type!r}")
    kwargs = {"split": source.get("split", "train")}
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    if source.get("cache_dir"):
        kwargs["cache_dir"] = source["cache_dir"]
    config = source.get("config")
    if config not in (None, "default"):
        kwargs["name"] = config
    dataset = load_dataset(source["id"], **kwargs)
    return normalise_dataset(dataset, definition.get("mapping") or {})


def make_training_args(config_cls, training: dict[str, Any], output_dir: Path):
    values = dict(training)
    if "epochs" in values:
        values.setdefault("num_train_epochs", values.pop("epochs"))
    values.setdefault("output_dir", str(output_dir))
    allowed = {field.name for field in fields(config_cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported {config_cls.__name__} options: {', '.join(unknown)}")
    return config_cls(**values)


def prepare_training_options(training: dict[str, Any], method: str):
    """Extract toolkit-only LoRA options before constructing a TRL config."""
    values = dict(training)
    lora = values.pop("lora", values.pop("peft", None))
    adapter_disabled = bool(values.pop("adapter_disabled", False))
    requested_merge = values.pop("merge_lora", None)
    if lora is None or lora is False:
        return values, None, adapter_disabled, bool(requested_merge) if requested_merge else False
    if lora is True:
        lora = {}
    if not isinstance(lora, dict):
        raise ValueError("training.lora must be a mapping or true")
    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("LoRA training requires peft; install with python -m pip install -e '.[train]'") from exc

    lora = dict(lora)
    r = int(lora.pop("r", 16))
    alpha = int(lora.pop("lora_alpha", lora.pop("alpha", 32)))
    dropout = float(lora.pop("lora_dropout", lora.pop("dropout", 0.05)))
    target_modules = lora.pop("target_modules", None)
    bias = lora.pop("bias", "none")
    modules_to_save = lora.pop("modules_to_save", None)
    if method == "reward_model" and modules_to_save is None:
        modules_to_save = ["score"]
    if lora:
        unknown = ", ".join(sorted(lora))
        raise ValueError(f"Unsupported LoRA options: {unknown}")
    peft_config = LoraConfig(
        task_type="SEQ_CLS" if method == "reward_model" else "CAUSAL_LM",
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        modules_to_save=modules_to_save,
    )
    merge_lora = bool(True if requested_merge is None else requested_merge)
    return values, peft_config, adapter_disabled, merge_lora


def save_trained_model(trainer, tokenizer, output_dir: Path, *, merge_lora: bool):
    """Save a trainer model, merging a PEFT adapter when requested."""
    model = getattr(trainer, "model", None)
    if hasattr(model, "policy"):  # TRL PPO wraps policy and value models.
        model = model.policy

    if merge_lora:
        merge = getattr(model, "merge_and_unload", None)
        if merge is None:
            raise RuntimeError("merge_lora=true but the TRL model has no merge_and_unload() method")
        model = merge()
        model.save_pretrained(str(output_dir), safe_serialization=True)
    else:
        trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def load_tokenizer(tokenizer_cls, model_ref):
    source = model_ref if isinstance(model_ref, str) else model_ref["id"]
    kwargs = model_load_kwargs(model_ref)
    tokenizer = tokenizer_cls.from_pretrained(source, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_load_kwargs(model_ref):
    if isinstance(model_ref, str):
        return {}
    values = {}
    for key in ("revision", "cache_dir"):
        if model_ref.get(key):
            values[key] = model_ref[key]
    return values


def load_causal_model(model_cls, model_ref):
    source = model_ref if isinstance(model_ref, str) else model_ref["id"]
    return model_cls.from_pretrained(source, **model_load_kwargs(model_ref))


def load_reward_model(model_cls, model_ref, *, for_value=False):
    source = model_ref if isinstance(model_ref, str) else model_ref["id"]
    model = model_cls.from_pretrained(
        source,
        num_labels=1,
        ignore_mismatched_sizes=True,
        **model_load_kwargs(model_ref),
    )
    if for_value:
        model.config.output_hidden_states = True
        model.config.use_cache = False
    return model


def prepare_sft_dataset(dataset, stage, tokenizer):
    columns = set(dataset.column_names)
    if "messages" in columns:
        # TRL 0.24 treats a dataset containing both `prompt` and `messages` as
        # prompt/completion data. Keep only the conversational column so it
        # applies the tokenizer chat template path.
        return dataset.select_columns(["messages"])
    if "text" in columns:
        return dataset.select_columns(["text"])
    if "messages" not in columns and "text" not in columns:
        instruction = next((name for name in ("instruction", "prompt") if name in columns), None)
        response = next((name for name in ("output", "response", "chosen") if name in columns), None)
        if instruction and response:
            def format_example(example):
                prefix = str(example[instruction])
                extra = str(example.get("input", "")).strip()
                if extra:
                    prefix = f"{prefix}\n{extra}"
                return {"text": f"{prefix}\n{example[response]}"}

            return dataset.map(format_example, remove_columns=dataset.column_names)
        raise ValueError(f"SFT dataset {stage.dataset!r} must contain messages or text")
    return dataset


def prepare_preference_dataset(dataset, stage, *, tokenizer=None):
    required = {"chosen", "rejected"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"Preference dataset {stage.dataset!r} is missing: {', '.join(missing)}")
    columns = [name for name in ("prompt", "chosen", "rejected") if name in dataset.column_names]
    dataset = dataset.select_columns(columns)
    if tokenizer is None:
        return dataset

    def stringify(example):
        converted = {}
        for name in ("chosen", "rejected"):
            value = example[name]
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if hasattr(tokenizer, "apply_chat_template"):
                    converted[name] = tokenizer.apply_chat_template(value, tokenize=False)
                else:
                    converted[name] = "\n".join(str(item.get("content", "")) for item in value)
            else:
                converted[name] = str(value)
        return converted

    return dataset.map(stringify)


def prepare_prompt_dataset(dataset, stage, tokenizer):
    columns = set(dataset.column_names)
    if "prompt" in columns:
        prompt_column = "prompt"
    elif "messages" in columns:
        prompt_column = "messages"
    else:
        raise ValueError(f"Prompt dataset {stage.dataset!r} must contain prompt or messages")

    def encode(example):
        value = example[prompt_column]
        if isinstance(value, list) and value and isinstance(value[0], dict) and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=True)
        else:
            text = str(value)
        return {"input_ids": tokenizer(text, truncation=True)["input_ids"]}

    return dataset.map(encode, remove_columns=dataset.column_names)


def normalise_dataset(dataset, mapping: dict[str, Any]):
    """Rename source columns to canonical names consumed by TRL."""
    for target, source in mapping.items():
        if target == source or source not in dataset.column_names:
            continue
        if target in dataset.column_names:
            dataset = dataset.remove_columns([target])
        dataset = dataset.rename_column(source, target)
    return dataset
