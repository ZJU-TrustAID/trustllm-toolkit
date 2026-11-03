"""Read-only documentation server for Alignment Toolkit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qs, urlsplit


DOCS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
TRAINING_PYTHON = os.environ.get("ALIGNMENT_PYTHON", sys.executable)


def hub_model() -> str:
    """hub 抽屉注入的基座模型（frontend.server.config 的 model 字段）。"""
    config_path = os.environ.get("ALIGNMENT_HUB_CONFIG", "")
    if config_path:
        try:
            with Path(config_path).open("r", encoding="utf-8") as handle:
                return str(json.load(handle).get("model") or "")
        except (OSError, ValueError):
            pass
    return os.environ.get("ALIGNMENT_HUB_MODEL", "")


class TrainingManager:
    """Own one local training subprocess and its bounded log buffer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._job: dict | None = None
        self._process: subprocess.Popen | None = None
        self._logs: list[str] = []

    def _set_job(self, **updates):
        with self._lock:
            if self._job is not None:
                self._job.update(updates)

    def _read_output(self, process, log_path: Path):
        with log_path.open("a", encoding="utf-8") as log:
            for line in process.stdout:
                line = line.rstrip("\n")
                with self._lock:
                    self._logs.append(line)
                    if len(self._logs) > 10000:
                        self._logs = self._logs[-10000:]
                log.write(line + "\n")
                log.flush()
        return_code = process.wait()
        self._set_job(
            status="succeeded" if return_code == 0 else "failed",
            return_code=return_code,
            finished_at=time.time(),
        )
        with self._lock:
            self._process = None

    def start(self, payload: dict) -> dict:
        with self._lock:
            if self._job and self._job.get("status") == "running":
                raise ValueError("A training job is already running")

        job_id = uuid.uuid4().hex[:10]
        run_dir = PROJECT_ROOT / ".runs" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config = build_training_config(payload, job_id)
        config_path = run_dir / "training.yaml"
        import yaml

        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        from alignment_toolkit import load_experiment

        load_experiment(config_path)
        gpu_ids = [str(value) for value in payload.get("gpus", [])]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        if not Path(TRAINING_PYTHON).exists():
            raise ValueError(f"ALIGNMENT_PYTHON does not exist: {TRAINING_PYTHON}")
        command = [TRAINING_PYTHON, "-m", "alignment_toolkit.cli", "train", "--config", str(config_path)]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._lock:
            self._logs = [f"$ CUDA_VISIBLE_DEVICES={','.join(gpu_ids) or '<default>'} {' '.join(command)}"]
            self._process = process
            self._job = {
                "id": job_id,
                "status": "running",
                "pid": process.pid,
                "gpus": gpu_ids,
                "workflow": payload.get("workflow"),
                "config_path": str(config_path),
                "run_dir": str(run_dir),
                "started_at": time.time(),
                "return_code": None,
            }
        threading.Thread(target=self._read_output, args=(process, run_dir / "train.log"), daemon=True).start()
        return self.status()

    def stop(self):
        with self._lock:
            process = self._process
            job = dict(self._job or {})
        if not process or process.poll() is not None:
            return job
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._set_job(status="stopped", finished_at=time.time())
        return self.status()

    def status(self):
        with self._lock:
            return dict(self._job or {"status": "idle"})

    def logs(self, offset=0):
        with self._lock:
            return {"offset": offset, "next_offset": len(self._logs), "lines": self._logs[offset:]}


TRAINING = TrainingManager()


def _model_ref(payload: dict) -> dict:
    source = str(payload.get("model_source", "local"))
    value = str(payload.get("model", "")).strip()
    if not value:
        raise ValueError("model is required")
    result = {"source": source, "id": value}
    if source == "huggingface" and payload.get("revision"):
        result["revision"] = str(payload["revision"])
    if payload.get("cache_dir"):
        result["cache_dir"] = str(payload["cache_dir"])
    return result


def _dataset_ref(payload: dict, key: str, default_id: str, default_split: str) -> dict:
    item = payload.get("datasets", {}).get(key, {})
    source_type = item.get("type", "huggingface")
    source = {"type": source_type, "split": item.get("split", default_split)}
    if source_type == "json":
        source["path"] = str(item.get("path", ""))
        if not source["path"]:
            raise ValueError(f"datasets.{key}.path is required for JSON data")
    else:
        source["id"] = str(item.get("id", default_id))
        if item.get("config"):
            source["config"] = item["config"]
    return {"kind": {"sft": "instruction", "preference": "preference", "prompts": "prompt"}[key], "adapter": "console", "source": source}


def build_training_config(payload: dict, job_id: str) -> dict:
    workflow = payload.get("workflow", "rlhf")
    if workflow not in {"sft", "dpo", "rlhf"}:
        raise ValueError("workflow must be sft, dpo, or rlhf")
    model = _model_ref(payload)
    datasets = {}
    stages = []
    common = {
        "num_train_epochs": float(payload.get("epochs", 1)),
        "per_device_train_batch_size": max(1, int(payload.get("batch_size", 1))),
        "gradient_accumulation_steps": max(1, int(payload.get("gradient_accumulation_steps", 1))),
        "learning_rate": float(payload.get("learning_rate", 2e-5)),
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
    }
    lora = bool(payload.get("lora", True))
    lora_options = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}

    if workflow in {"sft", "rlhf"}:
        datasets["sft"] = _dataset_ref(payload, "sft", "HuggingFaceH4/ultrachat_200k", "train_sft")
        sft_training = dict(common)
        sft_training["max_length"] = int(payload.get("max_length", 512))
        if lora:
            sft_training["lora"] = lora_options
            sft_training["merge_lora"] = True
        stages.append({"id": "sft", "method": "sft", "backend": "trl", "model": "base", "dataset": "sft", "training": sft_training})

    if workflow == "dpo":
        datasets["preference"] = _dataset_ref(payload, "preference", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs")
        dpo_training = dict(common)
        dpo_training.update({"beta": float(payload.get("beta", 0.1)), "max_length": int(payload.get("max_length", 512)), "max_prompt_length": int(payload.get("max_prompt_length", 256))})
        if lora:
            dpo_training.update({"lora": lora_options, "adapter_disabled": True, "merge_lora": True})
        stages.append({"id": "dpo", "method": "dpo", "backend": "trl", "model": "policy", "dataset": "preference", "training": dpo_training})

    if workflow == "rlhf":
        datasets["preference"] = _dataset_ref(payload, "preference", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs")
        datasets["prompts"] = _dataset_ref(payload, "prompts", "HuggingFaceH4/ultrachat_200k", "train_gen")
        reward_training = dict(common)
        reward_training["max_length"] = int(payload.get("max_length", 512))
        if lora:
            reward_training.update({"lora": {**lora_options, "modules_to_save": ["score"]}, "merge_lora": True})
        ppo_training = {
            "total_episodes": max(1, int(payload.get("ppo_episodes", 1))),
            "per_device_train_batch_size": 1,
            "response_length": int(payload.get("response_length", 64)),
            "local_rollout_forward_batch_size": 1,
            "num_mini_batches": max(1, int(payload.get("ppo_minibatches", 1))),
            "num_ppo_epochs": max(1, int(payload.get("ppo_epochs", 1))),
            "kl_coef": float(payload.get("ppo_kl", 0.05)),
            "gamma": float(payload.get("ppo_gamma", 1.0)),
            "lam": float(payload.get("ppo_lam", 0.95)),
            "vf_coef": float(payload.get("ppo_vf", 0.1)),
            "cliprange": float(payload.get("ppo_clip", 0.2)),
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": [],
        }
        if lora:
            ppo_training.update({"lora": lora_options, "adapter_disabled": True, "merge_lora": True})
        stages.extend([
            {"id": "reward_model", "method": "reward_model", "backend": "trl", "model": "sft", "dataset": "preference", "training": reward_training},
            {"id": "ppo", "method": "ppo", "backend": "trl", "policy": "sft", "reward_model": "reward_model", "dataset": "prompts", "training": ppo_training},
        ])

    final_stage = stages[-1]["id"]
    return {
        "version": 2,
        "experiment": {"name": f"console_{job_id}"},
        "models": {"base": model, **({"policy": model, "reference": model} if workflow == "dpo" else {})},
        "datasets": datasets,
        "pipeline": {"name": f"console_{workflow}_{job_id}", "stages": stages},
        "evaluation": {"profile": "console", "model": {"stage": final_stage}, "suites": []},
        "output": {"root": str(payload.get("output_root", "outputs/console"))},
    }


def source_files() -> dict[str, Path]:
    package = PROJECT_ROOT / "src/alignment_toolkit"
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "environment.yml",
        PROJECT_ROOT / "scripts/setup_env.sh",
        *package.glob("*.py"),
        *PROJECT_ROOT.glob("configs/**/*.yaml"),
    ]
    return {path.relative_to(PROJECT_ROOT).as_posix(): path for path in paths if path.is_file()}


def build_catalog() -> dict:
    import yaml
    import alignment_toolkit as alignment

    files = source_files()
    readme = files["README.md"].read_text(encoding="utf-8")
    blocks = [
        {"language": language, "code": text.rstrip()}
        for language, text in re.findall(r"```(\w+)\n(.*?)```", readme, re.S)
    ]
    configs = {
        key: {"text": path.read_text(encoding="utf-8"), "data": yaml.safe_load(path.read_text(encoding="utf-8"))}
        for key, path in files.items()
        if key.endswith(".yaml")
    }
    examples = {
        path.stem.removeprefix("alignment_"): path.read_text(encoding="utf-8")
        for path in (DOCS_ROOT / "examples").glob("alignment_*.py")
    }
    return {
        "alignment": {
            "name": "alignment",
            "version": alignment.__version__,
            "root": PROJECT_ROOT.name,
            "readme": readme,
            "blocks": blocks,
            "configs": configs,
            "methods": [asdict(alignment.METHODS.get(key)) for key in alignment.METHODS.names()],
            "backends": [
                {"name": key, "methods": sorted(alignment.BACKENDS.get(key).methods)}
                for key in alignment.BACKENDS.names()
            ],
            "sources": sorted(files),
            "extensions": examples,
        }
    }


def list_gpus() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in output.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, total, used, free, utilization = fields
        gpus.append({"index": int(index), "name": name, "memory_total_mb": int(total), "memory_used_mb": int(used), "memory_free_mb": int(free), "utilization_gpu": int(utilization)})
    return gpus
class Handler(BaseHTTPRequestHandler):
    def send_content(self, body, content_type, status=200):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/api/docs":
                self.send_content(json.dumps(build_catalog(), ensure_ascii=False), "application/json; charset=utf-8")
                return
            if parsed.path == "/api/source":
                params = parse_qs(parsed.query)
                path = source_files().get(params.get("file", [""])[0])
                if path is None:
                    raise ValueError("Unknown source")
                self.send_content(path.read_text(encoding="utf-8"), "text/plain; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self.send_content(json.dumps({"status": "ok", "hub_model": hub_model()}), "application/json")
                return
            if parsed.path == "/api/gpus":
                self.send_content(json.dumps(list_gpus(), ensure_ascii=False), "application/json; charset=utf-8")
                return
            if parsed.path == "/api/train/status":
                self.send_content(json.dumps(TRAINING.status(), ensure_ascii=False), "application/json; charset=utf-8")
                return
            if parsed.path == "/api/train/logs":
                offset = int(parse_qs(parsed.query).get("offset", [0])[0])
                self.send_content(json.dumps(TRAINING.logs(offset), ensure_ascii=False), "application/json; charset=utf-8")
                return
            allowed = {
                "/": ("index.html", "text/html"),
                "/guide": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/styles.css": ("styles.css", "text/css"),
                "/console": ("console.html", "text/html"),
                "/console.html": ("console.html", "text/html"),
                "/console.js": ("console.js", "text/javascript"),
                "/console.css": ("console.css", "text/css"),
                "/demo": ("demo.html", "text/html"),
                "/demo.html": ("demo.html", "text/html"),
                "/demo.js": ("demo.js", "text/javascript"),
                "/demo.css": ("demo.css", "text/css"),
            }
            if parsed.path not in allowed:
                raise ValueError("Not found")
            filename, mime = allowed[parsed.path]
            self.send_content((DOCS_ROOT / "static" / filename).read_bytes(), mime + "; charset=utf-8")
        except (ValueError, FileNotFoundError, KeyError):
            self.send_content("Not found", "text/plain", 404)
        except Exception:
            import traceback

            traceback.print_exc()
            self.send_content('{"error":"文档读取失败，请检查服务日志"}', "application/json; charset=utf-8", 500)

    def do_POST(self):  # noqa: N802
        parsed = urlsplit(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/train/validate":
                import yaml
                from alignment_toolkit import load_experiment

                config = build_training_config(payload, "validate")
                temp_path = PROJECT_ROOT / ".runs" / "validate.yaml"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
                spec = load_experiment(temp_path)
                self.send_content(json.dumps({"ok": True, "stages": [stage.stage_id for stage in spec.stages]}, ensure_ascii=False), "application/json; charset=utf-8")
                return
            if parsed.path == "/api/train/start":
                self.send_content(json.dumps(TRAINING.start(payload), ensure_ascii=False), "application/json; charset=utf-8")
                return
            if parsed.path == "/api/train/stop":
                self.send_content(json.dumps(TRAINING.stop(), ensure_ascii=False), "application/json; charset=utf-8")
                return
            raise ValueError("Not found")
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_content(json.dumps({"error": str(exc)}, ensure_ascii=False), "application/json; charset=utf-8", 400)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.send_content(json.dumps({"error": str(exc)}, ensure_ascii=False), "application/json; charset=utf-8", 500)


def main():
    parser = argparse.ArgumentParser(description="Alignment Toolkit documentation server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    build_catalog()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Alignment Toolkit documentation: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
