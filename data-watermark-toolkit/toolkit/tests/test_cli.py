# -*- coding: utf-8 -*-
"""CLI 与「零依赖默认路径」的单元测试。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from datamark.cli import dataset_watermark_main

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = REPO_ROOT / "configs" / "fugu_sim.json"


def run_cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """在仓库根目录下以当前 python 解释器运行 ``python3 -m datamark.cli``。"""
    return subprocess.run(
        [sys.executable, "-m", "datamark.cli", *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout,
    )


class TestCli(unittest.TestCase):
    """验收命令必须全部可运行。"""

    def test_version(self) -> None:
        result = run_cli("version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DataMark 0.1.0", result.stdout)

    def test_demo_output_format_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            started = time.time()
            result = run_cli("demo", "--out", tmp)
            elapsed = time.time() - started
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLess(elapsed, 30.0)
            lines = result.stdout.splitlines()
            self.assertTrue(lines[0].startswith("[1/6] 载入数据与密钥 ... ok (docs=96)"))
            self.assertIn(
                "[2/6] 嵌入 5 类训练时水印 ... ok (watermarked=48, fraction=0.50)",
                lines)
            self.assertIn("[3/6] 构建可疑模型（sim） ... ok (exposed=48)", lines)
            self.assertTrue(any(line.startswith("[4/6] 阈值校准") for line in lines))
            self.assertIn("[5/6] 多方法审计 ...", lines)
            self.assertTrue(lines[-1].startswith("[6/6] 报告已写入"))
            self.assertIn("method", result.stdout)
            self.assertIn("decision", result.stdout)
            report = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))
            self.assertIn("meta", report)
            self.assertIn("calibration", report)
            self.assertIn("results", report)
            self.assertEqual(len(report["results"]), 5)
            for item in report["results"]:
                for key in ("method", "access", "score", "n", "p_value",
                            "threshold", "decision", "detail"):
                    self.assertIn(key, item)

    def test_audit_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli("audit", "--config", str(SIM_CONFIG),
                             "--methods", "canary_memory", "--out", tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / "report.json").exists())
            self.assertTrue((Path(tmp) / "report.md").exists())
            report = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["results"][0]["method"], "canary_memory")
            self.assertEqual(report["summary"]["tested"], 1)

    def test_dataset_watermark_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_md = Path(tmp) / "chapter-3.md"
            out_dir = Path(tmp) / "work"
            with contextlib.redirect_stdout(io.StringIO()):
                code = dataset_watermark_main([
                    "--config", str(SIM_CONFIG),
                    "--report", str(report_md),
                    "--out", str(out_dir),
                ])
            self.assertEqual(code, 0)
            self.assertTrue(report_md.exists())
            text = report_md.read_text(encoding="utf-8")
            self.assertIn("# DataMark 数据使用审计报告", text)
            self.assertIn("| 方法 | 访问假设 | 统计量 | n | p 值 | 阈值 | 判定 |", text)
            self.assertTrue((out_dir / "report.json").exists())
            self.assertTrue((out_dir / "wm" / "watermarked.jsonl").exists())
            self.assertTrue((out_dir / "wm" / "manifest.json").exists())


class TestZeroDependency(unittest.TestCase):
    """默认路径不得在模块顶层导入第三方库（尤其是 numpy/torch/yaml）。"""

    FORBIDDEN = {"numpy", "scipy", "pandas", "matplotlib", "torch",
                 "transformers", "peft", "yaml", "sklearn"}

    def test_no_forbidden_top_level_imports(self) -> None:
        package = REPO_ROOT / "datamark"
        offenders: list[str] = []
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in self.FORBIDDEN:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
        self.assertEqual(offenders, [], f"默认路径出现第三方顶层依赖：{offenders}")

    def test_system_python_runs_demo(self) -> None:
        # 验收环境使用系统 python3（仅标准库 + pyyaml）；这里再次确认可执行。
        result = subprocess.run(
            [sys.executable, "-c",
             "import datamark, datamark.cli, datamark.pipeline; "
             "print(datamark.__version__)"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0.1.0", result.stdout)


if __name__ == "__main__":
    unittest.main()
