# -*- coding: utf-8 -*-
"""审计报告：结果表、中文 summary、JSON / Markdown 导出。

``report.json`` 的字段与 ``notes/03-toolkit-spec.md`` §6 一致::

    {
      "meta": {"toolkit": "DataMark", "version": "0.1.0", "seed": 42,
               "rho": 0.5, "model": "mock-teaching-v1", "created": "..."},
      "fpr_target": 0.05,
      "calibration": {"n_negative_models": 24,
                      "thresholds": {"canary_memory": 1.812, ...}},
      "results": [
        {"method": "canary_memory", "access": "white-box", "score": 3.421,
         "n": 36, "p_value": 0.0003, "threshold": 1.812,
         "decision": "watermarked", "detail": {...}}
      ],
      "summary": {"detected": 5, "tested": 5,
                  "note": "教学级模拟结果，不代表真实模型行为"}
    }

其中 ``decision`` 只用两个英文取值 ``watermarked`` / ``clean``（书稿 2.5.4
的输出表逐字引用这两个词）。所有数字保留固定小数位，保证同一 seed 下
报告逐字节可复现。
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

#: 报告里固定出现的说明文字（书稿与网站共用）。
SIMULATION_NOTE = "教学级模拟结果，不代表真实模型行为"

#: summary 表的列宽（与 §3.4 的输出格式对应）。
_COL_WIDTHS = (26, 12, 11, 6, 10)


def _fmt_score(value: float) -> str:
    """统计量统一保留 3 位小数（过大时保留 1 位，避免撑破表格）。"""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.3f}"


def _fmt_p(value: float) -> str:
    """p 值格式：≥0.0001 时保留 4 位小数，否则写作 ``<0.0001``。"""
    if value is None:
        return "n/a"
    try:
        p = float(value)
    except (TypeError, ValueError):
        return str(value)
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def deterministic_timestamp(seed: int) -> str:
    """由 seed 派生的固定时间戳（用于报告可复现）。

    真实审计报告应写入实际时间；教学工具包为了保证「同配置 + 同 seed ⇒
    报告逐字节一致」，用 ``2026-10-26T00:00:00+00:00 + seed 秒`` 作为
    确定性时间戳，并在 meta 中标注 ``created_source="deterministic"``。
    """
    base = _dt.datetime(2026, 10, 26, 0, 0, 0, tzinfo=_dt.timezone.utc)
    moment = base + _dt.timedelta(seconds=int(seed) % 86400)
    return moment.isoformat().replace("+00:00", "Z")


@dataclass
class AuditReport:
    """一次多方法审计的结果。

    :param results: 每个方法一条结果（字段见模块文档）。
    :param fpr_target: 目标误报率 α。
    :param calibration: 阈值校准信息（负样本数、各方法阈值、经验 FPR）。
    :param meta: 报告元信息（工具包版本、seed、ρ、模型名等）。
    """

    results: list[dict] = field(default_factory=list)
    fpr_target: float = 0.05
    calibration: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # -- 便捷属性 ----------------------------------------------------------
    @property
    def detected(self) -> int:
        """判定为「检出」的方法数。"""
        return sum(1 for r in self.results if r.get("decision") == "watermarked")

    @property
    def tested(self) -> int:
        """实际测试的方法数。"""
        return len(self.results)

    @property
    def method_names(self) -> list[str]:
        """结果中出现的方法名（保持顺序）。"""
        return [r["method"] for r in self.results]

    def get(self, method: str) -> dict | None:
        """按方法名取结果。"""
        for r in self.results:
            if r["method"] == method:
                return r
        return None

    # -- 输出 --------------------------------------------------------------
    def summary(self) -> str:
        """返回中文/英文混合的结果表（列与正文公式一一对应）。

        ::

            method                    access      score      n     p_value   decision
            canary_memory             white-box   3.421     36    0.0003    watermarked
        """
        w_method, w_access, w_score, w_n, w_p = _COL_WIDTHS
        header = (f"{'method':<{w_method}}{'access':<{w_access}}"
                  f"{'score':<{w_score}}{'n':<{w_n}}{'p_value':<{w_p}}decision")
        lines = [header]
        for r in self.results:
            lines.append(
                f"{str(r.get('method', '')):<{w_method}}"
                f"{str(r.get('access', '')):<{w_access}}"
                f"{_fmt_score(r.get('score')):<{w_score}}"
                f"{int(r.get('n', 0)):<{w_n}}"
                f"{_fmt_p(r.get('p_value')):<{w_p}}"
                f"{r.get('decision', '')}"
            )
        return "\n".join(lines)

    def footer(self) -> str:
        """结果表下方的中文小结（demo 单独打印，不进入 summary 表格）。"""
        return (f"共测试 {self.tested} 个检测器，判定「带水印」{self.detected} 个"
                f"（目标误报率 α = {self.fpr_target:g}；{SIMULATION_NOTE}）。")

    def to_dict(self) -> dict:
        """返回与 §6 JSON schema 对应的 dict（不含路径等易变字段）。"""
        thresholds = dict(self.calibration.get("thresholds", {}))
        calibration = {
            "n_negative_models": int(self.calibration.get("n_negative_models", 0)),
            "thresholds": {k: round(float(v), 6) for k, v in thresholds.items()},
        }
        if "empirical_fpr" in self.calibration:
            calibration["empirical_fpr"] = {
                k: round(float(v), 6)
                for k, v in self.calibration["empirical_fpr"].items()
            }
        cleaned_results = []
        for r in self.results:
            cleaned_results.append({
                "method": r["method"],
                "access": r["access"],
                "score": round(float(r["score"]), 6),
                "n": int(r.get("n", 0)),
                "p_value": round(max(float(r["p_value"]), 1e-12), 12),
                "threshold": round(float(r.get("threshold", 0.0)), 6),
                "decision": r["decision"],
                "detail": _jsonable(r.get("detail", {})),
            })
        return {
            "meta": dict(self.meta),
            "fpr_target": float(self.fpr_target),
            "calibration": calibration,
            "results": cleaned_results,
            "summary": {
                "detected": self.detected,
                "tested": self.tested,
                "note": SIMULATION_NOTE,
            },
        }

    def to_json(self, path: str | Path) -> Path:
        """写出 ``report.json``（UTF-8、固定缩进、键顺序固定 ⇒ 可复现）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")
        return p

    def to_markdown(self, path: str | Path) -> Path:
        """写出中文 Markdown 版报告（便于贴进书稿 / 课程作业）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = self.meta
        lines = [
            "# DataMark 数据使用审计报告",
            "",
            f"- 工具包版本：DataMark {meta.get('version', '')}",
            f"- 随机种子：{meta.get('seed', '')}",
            f"- 带水印样本比例 ρ：{meta.get('rho', '')}",
            f"- 目标模型：{meta.get('model', '')}",
            f"- 目标误报率 α：{self.fpr_target:g}",
            f"- 负样本模型数：{self.calibration.get('n_negative_models', 0)}",
            f"- 报告时间（确定性时间戳）：{meta.get('created', '')}",
            "",
            "## 检测结果",
            "",
            "| 方法 | 访问假设 | 统计量 | n | p 值 | 阈值 | 判定 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.results:
            lines.append(
                f"| `{r['method']}` | {r['access']} | {_fmt_score(r.get('score'))} | "
                f"{int(r.get('n', 0))} | {_fmt_p(r.get('p_value'))} | "
                f"{_fmt_score(r.get('threshold'))} | {r.get('decision')} |"
            )
        lines += [
            "",
            f"> {SIMULATION_NOTE}。判定依据是「负样本模型校准出的阈值 + 单侧统计检验」；",
            "> 技术证据只能说明统计匹配关系，来源归属仍需结合密钥保管、数据获取过程与合同。",
            "",
        ]
        with p.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return p


def _jsonable(value):
    """把 detail 里的非 JSON 类型（元组、集合等）转换为可序列化结构。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, float):
        return round(value, 6)
    return value
