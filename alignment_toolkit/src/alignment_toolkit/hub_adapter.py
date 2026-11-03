"""Hub adapter: run one alignment experiment (SFT / DPO / RLHF) from a unified YAML.

Invoked by the TrustLLM hub as a subprocess inside this package's own
environment::

    python -m alignment_toolkit.hub_adapter --config <hub.yaml> --output-dir <session dir>

The adapter maps the hub's unified YAML (``chapter_params`` + ``models.base``)
onto this package's public API (``load_experiment`` / ``TRLRunner``), runs the
pipeline and the optional refusal-heuristic safety evaluation, and writes
``report.md`` / ``report.json`` plus a hub-facing ``result.json`` into the
output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

# RLHF/PPO 阶段的默认训练超参（与 docs/server.py 控制台保持一致）。
_PPO_DEFAULTS = {
    "per_device_train_batch_size": 1,
    "local_rollout_forward_batch_size": 1,
    "gamma": 1.0,
    "lam": 0.95,
    "vf_coef": 0.1,
    "cliprange": 0.2,
    "logging_steps": 1,
    "save_strategy": "no",
    "report_to": [],
}

_LORA_DEFAULTS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

_DATASET_MAPPING = {
    "instruction": {"messages": "messages"},
    "preference": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
    "prompt": {"prompt": "prompt"},
}


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是映射: {path}")
    return config


def _model_string(config: dict[str, Any]) -> str:
    entry = (config.get("models") or {}).get("base") or {}
    if entry.get("source") == "local_path":
        return str(entry["path"])
    if entry.get("source") == "hf":
        return str(entry["repo"])
    raise ValueError("该章节需要本地路径或 HuggingFace 基座模型（models.base）")


def _dataset_ref(name: str, kind: str, value: str, split: str) -> dict[str, Any]:
    """数据集字段：HuggingFace 仓库 id 或本地 JSON/JSONL 路径。"""
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"缺少 {name} 数据集")
    path = Path(value).expanduser()
    if path.suffix.lower() in {".json", ".jsonl"}:
        if not path.is_file():
            raise ValueError(f"{name} 本地数据集不存在: {path}")
        source = {"type": "json", "path": str(path.resolve()), "split": "train"}
    else:
        source = {"type": "huggingface", "id": value, "split": split}
    return {"kind": kind, "adapter": "hub", "source": source, "mapping": dict(_DATASET_MAPPING[kind])}


def build_experiment_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """把 hub 统一 YAML 组装成本包实验配置（结构与 configs/experiments/*.yaml 相同）。"""
    params = dict(config.get("chapter_params") or {})
    workflow = str(params.get("workflow") or "sft")
    if workflow not in {"sft", "dpo", "rlhf"}:
        raise ValueError(f"workflow 必须是 sft / dpo / rlhf，收到 {workflow!r}")

    model_id = _model_string(config)
    model_ref: dict[str, Any] = {"source": "huggingface", "id": model_id}

    epochs = float(params.get("epochs") or 1)
    common_training = {
        "num_train_epochs": epochs,
        "per_device_train_batch_size": max(1, int(params.get("batch_size") or 1)),
        "gradient_accumulation_steps": max(1, int(params.get("gradient_accumulation_steps") or 1)),
        "learning_rate": float(params.get("learning_rate") or 2e-5),
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
    }
    seed = config.get("common", {}).get("seed")
    if seed not in (None, ""):
        common_training["seed"] = int(seed)

    use_lora = bool(params.get("lora", True))
    max_length = int(params.get("max_length") or 512)
    datasets: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []

    if workflow in {"sft", "rlhf"}:
        datasets["sft"] = _dataset_ref("SFT 指令数据集", "instruction",
                                       params.get("sft_dataset") or "HuggingFaceH4/ultrachat_200k",
                                       "train_sft")
        training = dict(common_training)
        training["max_length"] = max_length
        if use_lora:
            training["lora"] = dict(_LORA_DEFAULTS)
            training["merge_lora"] = True
        stages.append({"id": "sft", "method": {"name": "sft", "backend": "trl"},
                       "model": "base", "dataset": {"name": "sft"}, "training": training})

    if workflow == "dpo":
        datasets["preference"] = _dataset_ref("偏好数据集", "preference",
                                              params.get("preference_dataset") or "HuggingFaceH4/ultrafeedback_binarized",
                                              "train_prefs")
        training = dict(common_training)
        training.update({
            "beta": float(params.get("beta") or 0.1),
            "max_length": max_length,
            "max_prompt_length": max(64, max_length // 2),
        })
        if use_lora:
            training.update({"lora": dict(_LORA_DEFAULTS), "adapter_disabled": True, "merge_lora": True})
        stages.append({"id": "dpo", "method": {"name": "dpo", "backend": "trl"},
                       "model": "policy", "dataset": {"name": "preference"}, "training": training})

    if workflow == "rlhf":
        datasets["preference"] = _dataset_ref("偏好数据集", "preference",
                                              params.get("preference_dataset") or "HuggingFaceH4/ultrafeedback_binarized",
                                              "train_prefs")
        datasets["prompts"] = _dataset_ref("PPO 提示数据集", "prompt",
                                           params.get("prompt_dataset") or "HuggingFaceH4/ultrachat_200k",
                                           "train_gen")
        reward_training = dict(common_training)
        reward_training["max_length"] = max_length
        if use_lora:
            reward_training.update({"lora": {**_LORA_DEFAULTS, "modules_to_save": ["score"]}, "merge_lora": True})
        ppo_training = dict(_PPO_DEFAULTS)
        ppo_training.update({
            "total_episodes": max(1, int(params.get("ppo_episodes") or 1)),
            "response_length": int(params.get("response_length") or 64),
            "num_ppo_epochs": max(1, int(params.get("ppo_epochs") or 1)),
            "kl_coef": float(params.get("kl_coef") or 0.05),
        })
        if use_lora:
            ppo_training.update({"lora": dict(_LORA_DEFAULTS), "adapter_disabled": True, "merge_lora": True})
        stages.extend([
            {"id": "reward_model", "method": {"name": "reward_model", "backend": "trl"},
             "model": "sft", "dataset": {"name": "preference"}, "training": reward_training},
            {"id": "ppo", "method": {"name": "ppo", "backend": "trl"},
             "policy": "sft", "reward_model": "reward_model", "dataset": {"name": "prompts"},
             "training": ppo_training},
        ])

    models = {"base": model_ref}
    if workflow == "dpo":
        models.update({"policy": dict(model_ref), "reference": dict(model_ref)})

    final_stage = stages[-1]["id"]
    evaluate = bool(params.get("evaluate", True))
    evaluation: dict[str, Any] = {
        "profile": "safety_alignment",
        "model": {"stage": final_stage},
        "suites": [],
    }
    if evaluate:
        suites = _evaluation_suites(evaluation)
        evaluation["suites"] = suites

    return {
        "version": 2,
        "experiment": {"name": "hub_alignment"},
        "models": models,
        "datasets": datasets,
        "pipeline": {"name": f"hub_{workflow}", "stages": stages},
        "evaluation": evaluation,
        "output": {"root": str((output_dir / "training").resolve()), "collision": "error"},
    }


def _evaluation_suites(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """safety_alignment 套件：优先复用 configs/evaluations.yaml，退化到单一 harmbench。"""
    registry = Path(__file__).resolve().parents[2] / "configs" / "evaluations.yaml"
    if registry.is_file():
        profiles = yaml.safe_load(registry.read_text(encoding="utf-8")).get("profiles") or {}
        suites = (profiles.get(evaluation.get("profile", "safety_alignment")) or {}).get("suites")
        if suites:
            return list(suites)
    return [{  # pip 安装等场景下 configs 不在包旁：退化为单一轻量套件
        "name": "harmbench",
        "backend": "refusal_heuristic",
        "source": {"type": "huggingface", "id": "walledai/HarmBench", "split": "test"},
        "prompt_field": "prompt",
    }]


def _write_result(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fail(output_dir: Path, message: str) -> int:
    if output_dir is not None:
        _write_result(
            output_dir,
            {
                "status": "failed",
                "summary": message,
                "blocks": [{"type": "markdown", "title": "失败原因", "content": message}],
                "artifacts": [],
            },
        )
    print(f"[hub-adapter] 失败: {message}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    return 2


def _stage_table(stages: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for stage in stages:
        rows.append([
            stage["stage_id"], stage["method"]["name"], stage["dataset"]["name"],
            stage["model"] or stage.get("policy", ""),
        ])
    return {"type": "table", "title": "训练阶段", "header": ["阶段", "方法", "数据集", "输入模型"], "rows": rows}


def _evaluation_blocks(evaluation_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not evaluation_result:
        return []
    rows = []
    for suite in evaluation_result.get("suites", []):
        summary = suite.get("summary") or {}
        rows.append([
            suite.get("suite", ""),
            summary.get("sample_count", 0),
            f"{summary.get('refusal_rate', 0):.3f}",
            f"{summary.get('attack_success_rate', '-'):.3f}" if "attack_success_rate" in summary else "-",
            f"{summary.get('over_refusal_rate', '-'):.3f}" if "over_refusal_rate" in summary else "-",
        ])
    return [{"type": "table", "title": "安全评估（refusal_heuristic）",
             "header": ["套件", "样本数", "拒答率", "攻击成功率", "过度拒答率"], "rows": rows}]


def _finish(output_dir: Path, *, summary: str, blocks: list[dict[str, Any]],
            data: dict[str, Any], experiment_config: dict[str, Any]) -> int:
    (output_dir / "report.json").write_text(
        json.dumps({"summary": summary, "data": data, "config": experiment_config},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# 安全对齐实验报告", "", f"{summary}", ""]
    lines.append("## 训练阶段")
    lines.append("| 阶段 | 方法 | 数据集 | 输入模型 |")
    lines.append("| --- | --- | --- | --- |")
    for block in blocks:
        if block.get("type") == "table" and block.get("title") == "训练阶段":
            for row in block["rows"]:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    for block in _evaluation_blocks(data.get("evaluation")):
        lines += ["", f"## {block['title']}",
                  "| " + " | ".join(block["header"]) + " |",
                  "| " + " | ".join(["---"] * len(block["header"])) + " |"]
        for row in block["rows"]:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    lines += ["", "## 产物", ""]
    for name, path in sorted((data.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{path}`")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_result(output_dir, {
        "status": "succeeded",
        "summary": summary,
        "blocks": blocks + _evaluation_blocks(data.get("evaluation")),
        "artifacts": [{"name": "report.md"}, {"name": "report.json"}],
        "data": data,
    })
    print("[hub-adapter] 实验完成", flush=True)
    return 0


def run_training(config: dict[str, Any], output_dir: Path, *, dry_run: bool = False) -> int:
    from alignment_toolkit import TRLRunner, load_experiment

    experiment_config = build_experiment_config(config, output_dir)
    experiment_path = output_dir / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment_config, allow_unicode=True, sort_keys=False),
                               encoding="utf-8")
    print(f"[hub-adapter] 实验配置已写入 {experiment_path}", flush=True)

    spec = load_experiment(experiment_path)
    params = dict(config.get("chapter_params") or {})
    workflow = str(params.get("workflow") or "sft")
    stage_rows = [
        {"stage_id": stage.stage_id, "method": {"name": stage.method},
         "dataset": {"name": stage.dataset}, "model": stage.input.get("model") or stage.input.get("policy") or "-"}
        for stage in spec.stages
    ]
    if dry_run:
        return _finish(
            output_dir,
            summary=f"dry-run 通过：{workflow} 流水线共 {len(stage_rows)} 个阶段，配置可解析（未执行训练）。",
            blocks=[_stage_table(stage_rows)],
            data={"workflow": workflow, "dry_run": True, "evaluation": None, "artifacts": {}},
            experiment_config=experiment_config,
        )

    from alignment_toolkit.evaluation import run_safety_evaluation
    from alignment_toolkit.pipeline import AlignmentExperiment

    print(f"[hub-adapter] 开始训练：workflow={workflow}，模型={_model_string(config)}", flush=True)
    AlignmentExperiment(spec, runner=TRLRunner()).run()

    artifacts = {stage.stage_id: str(Path(spec.output["root"]) / spec.name / stage.stage_id)
                 for stage in spec.stages}

    evaluation_result = None
    suites = _evaluation_suites(spec.evaluation)
    if suites:
        print("[hub-adapter] 运行安全评估（refusal_heuristic）…", flush=True)
        eval_dir = output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        evaluation_result = run_safety_evaluation(
            model_name_or_path=str(Path(spec.output["root"]) / spec.name / spec.stages[-1].stage_id),
            profile={"name": spec.evaluation.get("profile", "safety_alignment"), "suites": suites},
            output_dir=eval_dir,
            max_samples=int(params.get("eval_max_samples") or 32),
        )
        artifacts["evaluation"] = str(eval_dir / "summary.json")

    refused = ""
    if evaluation_result:
        rates = [s["summary"].get("refusal_rate") for s in evaluation_result["suites"] if s.get("summary")]
        if rates:
            refused = f"，平均拒答率 {sum(rates) / len(rates):.3f}"
    return _finish(
        output_dir,
        summary=f"{workflow.upper()} 训练完成（{len(stage_rows)} 个阶段{refused}），"
                f"checkpoint 位于 training/{spec.name}/ 下的各阶段目录。",
        blocks=[_stage_table(stage_rows)],
        data={
            "workflow": workflow,
            "model": _model_string(config),
            "artifacts": artifacts,
            "evaluation": evaluation_result,
        },
        experiment_config=experiment_config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrustLLM hub adapter for alignment-toolkit")
    parser.add_argument("--config", required=True, help="hub 统一 YAML 配置")
    parser.add_argument("--output-dir", required=True, help="输出目录（hub 会话目录）")
    parser.add_argument("--dry-run", action="store_true", help="只解析校验配置，不执行训练")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = _load_config(Path(args.config).expanduser())
    except (OSError, yaml.YAMLError) as exc:
        return _fail(output_dir, f"读取配置失败: {exc}")
    try:
        return run_training(config, output_dir, dry_run=args.dry_run)
    except Exception as exc:  # adapter boundary: surface any failure as result.json + exit 2
        return _fail(output_dir, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
