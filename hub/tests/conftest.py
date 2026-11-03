"""Shared fixtures: a fake chapter toolkit driven by a stdlib-only adapter."""
from __future__ import annotations

from pathlib import Path

import pytest

from trustllm_hub.config import DEFAULTS, HubConfig

# A fake adapter mirroring the hub_adapter contract: read --config/--output-dir,
# log to stdout, write result.json (+ an artifact).
FAKE_ADAPTER = '''\
"""Fake chapter adapter used by hub tests (no GPU, no network)."""
import json
import sys
import time
from pathlib import Path

import yaml

args = sys.argv[1:]
config_path = Path(args[args.index("--config") + 1])
out = Path(args[args.index("--output-dir") + 1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
params = config.get("chapter_params") or {}
print("fake adapter start, toolkit=%s" % config.get("toolkit"), flush=True)
time.sleep(float(params.get("sleep", 0.2)))
print("fake adapter writing result", flush=True)
out.mkdir(parents=True, exist_ok=True)
(out / "report.md").write_text("# Fake report\\n\\ntoolkit: %s\\n" % config.get("toolkit"), encoding="utf-8")
if params.get("fail"):
    print("fake adapter boom", file=sys.stderr)
    sys.exit(3)
(out / "result.json").write_text(json.dumps({
    "status": "succeeded",
    "summary": "fake ok for %s" % config.get("toolkit"),
    "blocks": [{"type": "markdown", "title": "结果", "content": "**hello** from fake adapter"}],
    "artifacts": [{"name": "report.md"}],
    "data": {"chapter_params": params},
}, ensure_ascii=False), encoding="utf-8")
print("fake adapter done", flush=True)
'''

FAKE_MANIFEST = """\
chapter: 99
title: 测试章节
toolkit: fakekit
enabled: true
brief: 仅用于 hub 测试的假章节。
frontend:
  mode: link
  entry: "/"
backend:
  command: ["{python}", "fake_adapter.py", "--config", "{yaml}", "--output-dir", "{output_dir}"]
  timeout: 30
models:
  - role: default
    title: 主模型
    accepts: [local_path, hf]
    required: true
    default: {source: hf, repo: "org/model"}
params:
  - {key: mode, type: enum, options: [fast, slow], default: fast, title: 模式}
  - {key: sleep, type: float, default: 0.2, title: 模拟耗时}
  - {key: fail, type: bool, default: false, title: 模拟失败}
results:
  primary: report.md
"""


def make_chapter_dir(root: Path, *, toolkit: str = "fakekit", manifest: str = FAKE_MANIFEST) -> Path:
    directory = root / "chapters" / toolkit
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fake_adapter.py").write_text(FAKE_ADAPTER, encoding="utf-8")
    text = manifest.replace("toolkit: fakekit", f"toolkit: {toolkit}")
    (directory / "chapter.yaml").write_text(text, encoding="utf-8")
    return directory


def make_config(root: Path, **overrides) -> HubConfig:
    data = dict(DEFAULTS)
    data.update({"chapter_dirs": ["chapters"], "max_concurrent_runs": 1, **overrides})
    return HubConfig(root=root, hub_dir=root / "hub", data=data)


def sample_run_config(toolkit: str = "fakekit", **chapter_params) -> dict:
    return {
        "version": 1,
        "chapter": 99,
        "toolkit": toolkit,
        "common": {"device": "cpu", "dtype": "auto", "seed": 0},
        "models": {"default": {"source": "hf", "repo": "org/model"}},
        "chapter_params": dict(chapter_params),
    }


@pytest.fixture
def hub_env(tmp_path: Path) -> HubConfig:
    make_chapter_dir(tmp_path)
    config = make_config(tmp_path)
    config.workspace.mkdir(parents=True, exist_ok=True)
    return config
