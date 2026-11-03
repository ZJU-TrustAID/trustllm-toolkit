# -*- coding: utf-8 -*-
"""DataMark 命令行入口（``python -m datamark.cli`` / ``datamark``）。

子命令与书稿 2.5 的四段式一一对应::

    datamark init   --out work/fugu              # 2.5.1 准备目录与密钥
    datamark embed  --config configs/fugu_sim.json [--fraction 0.25] [--out work/fugu/wm]
    datamark fit    --config configs/fugu_sim.json [--mode sim|hf|api] [--out ...]
    datamark audit  --config configs/fugu_sim.json [--fpr 0.05] [--methods all] [--out ...]
    datamark sweep  --config configs/fugu_sim.json --fractions 0.05,0.1,0.25,0.5,1.0 \
                    --attacks none,dedup,paraphrase --out work/fugu/sweep.csv
    datamark demo   [--config configs/fugu_sim.json] [--out work/fugu_demo]
    datamark version

所有子命令都在**纯标准库**下运行（默认模拟模式），不需要联网或 GPU。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import ConfigError, config_to_dict
from .keyring import KeyFileError
from .models import ModelBuildError
from .pipeline import DataMark, PipelineError
from .methods import METHOD_ORDER
from .version import __version__ as _VERSION


def _default_config() -> str:
    """默认配置文件：优先仓库内 configs/fugu_sim.json，其次当前目录。"""
    repo_cfg = Path(__file__).resolve().parent.parent / "configs" / "fugu_sim.json"
    if repo_cfg.exists():
        return str(repo_cfg)
    return "configs/fugu_sim.json"


def _split_csv(value: str) -> list[str]:
    """把 ``a,b,c`` 形式的参数切成列表。"""
    return [part.strip() for part in str(value).split(",") if part.strip()]


# ---------------------------------------------------------------------------
# 各子命令
# ---------------------------------------------------------------------------
def cmd_version(args: argparse.Namespace) -> int:
    """打印版本信息。"""
    print(f"DataMark {_VERSION}")
    print("数据集水印教学工具包（教材第二章 2.5「实践与应用」配套）")
    print("默认模拟模式：Python 标准库即可运行，无需 GPU、无需联网。")
    print(f"已注册方法：{', '.join(METHOD_ORDER)}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """初始化工作目录：写配置快照、创建密钥文件。"""
    dm = DataMark.from_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "config.resolved.json"
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(dm.config), f, ensure_ascii=False, indent=2)
        f.write("\n")
    dm.key  # 触发密钥加载 / 创建
    print(f"[init] 已载入配置：{args.config}")
    print(f"[init] 受保护语料：{dm.config.dataset_path}（{len(dm.dataset)} 条）")
    print(f"[init] 公共混合语料：{dm.config.mix_path}（{len(dm.public_mix)} 条）")
    print(f"[init] 配置快照：{resolved}")
    print(f"[init] 水印密钥：{dm.config.key_path}")
    print("[init] 完成。下一步：datamark embed --config "
          f"{args.config} --out {out / 'wm'}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """构造带水印数据集。"""
    dm = DataMark.from_config(args.config)
    if args.fraction is not None:
        dm.set_fraction(float(args.fraction))
    wm = dm.embed(out_dir=args.out)
    methods = ", ".join(wm.manifest.get("meta", {}).get("methods", []))
    print(f"[embed] 数据集 {wm.n_docs} 条，带水印 {wm.watermarked_count} 条"
          f"（fraction={dm.actual_rho:.2f}）")
    print(f"[embed] 方法：{methods}")
    if args.out:
        print(f"[embed] 可发布数据集：{Path(args.out) / 'watermarked.jsonl'}")
        print(f"[embed] 水印清单（秘密材料，不随数据发布）：{Path(args.out) / 'manifest.json'}")
    print(f"[embed] 数据质量：困惑度 {wm.quality['perplexity_before']} → "
          f"{wm.quality['perplexity_after']}"
          f"（{wm.quality['perplexity_delta_pct']:+.2f}%），"
          f"3-gram 重复率 {wm.quality['dup_ratio_before']} → "
          f"{wm.quality['dup_ratio_after']}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """训练 / 构建可疑模型。"""
    dm = DataMark.from_config(args.config)
    if args.fraction is not None:
        dm.set_fraction(float(args.fraction))
    wm = dm.embed()
    mode = "hf" if args.mode == "local" else args.mode   # local 是面向教材读者的别名
    try:
        model = dm.fit(wm, mode=mode)
    except ModelBuildError as exc:
        print(f"[fit] 本地模型模式不可用：{exc}", file=sys.stderr)
        print("[fit] 默认模拟模式无需任何第三方依赖；如需真实小模型，请先 "
              "pip install -e .[real]，并按 README「真实模式准备」设置 HF 镜像与缓存目录。",
              file=sys.stderr)
        return 2
    print(f"[fit] 模型类型：{model.kind}（访问假设：{model.access}）")
    print(f"[fit] 暴露文档数：{model.exposure_count} / {wm.n_docs}"
          f"（ρ = {model.rho:.2f}）")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": model.kind,
            "access": model.access,
            "exposure_count": model.exposure_count,
            "rho": round(model.rho, 6),
            "seed": dm.seed,
            "methods": dm.enabled_methods(),
        }
        with (out / "model.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[fit] 模型摘要：{out / 'model.json'}")
    print("[fit] 下一步：datamark audit --config "
          f"{args.config} --out {args.out or 'work/fugu/report'}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """多方法审计并输出报告。"""
    dm = DataMark.from_config(args.config)
    if args.fraction is not None:
        dm.set_fraction(float(args.fraction))
    if args.fpr is not None:
        dm.config.audit["fpr"] = float(args.fpr)
    wm = dm.embed()
    model = dm.fit(wm)
    methods = None if not args.methods else _split_csv(args.methods)
    report = dm.audit(model, methods=methods)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = report.to_json(out / "report.json")
    md_path = report.to_markdown(out / "report.md")
    print(f"[audit] 数据集 {wm.n_docs} 条，带水印 {wm.watermarked_count} 条，"
          f"暴露文档 {model.exposure_count} 条")
    print(f"[audit] 目标误报率 α = {report.fpr_target:g}，"
          f"负样本模型 {report.calibration.get('n_negative_models', 0)} 个")
    print(report.summary())
    print(report.footer())
    print(f"[audit] JSON 报告：{json_path}")
    print(f"[audit] Markdown 报告：{md_path}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """比例 × 清洗扫描。"""
    dm = DataMark.from_config(args.config)
    fractions = [float(x) for x in _split_csv(args.fractions)]
    attacks = _split_csv(args.attacks)
    records = dm.sweep(fractions, attacks, out_csv=args.out,
                       n_trials=args.trials)
    n_trials = records[0]["n_trials"] if records else 0
    print(f"[sweep] 完成：{len(fractions)} 个比例 × {len(attacks)} 种清洗 × "
          f"{n_trials} 次试验")
    print(f"[sweep] CSV：{args.out}（{len(records)} 行）")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """全流程演示（30 秒内跑完，打印 6 步结果 + 方法表）。"""
    started = time.perf_counter()
    dm = DataMark.from_config(args.config)
    print(f"[1/6] 载入数据与密钥 ... ok (docs={len(dm.dataset)})")

    wm = dm.embed()
    n_train = len(dm.config.enabled_training_methods())
    n_infer = len(dm.config.enabled_inference_methods())
    extra = f" + {n_infer} 类推理时" if n_infer else ""
    print(f"[2/6] 嵌入 {n_train} 类训练时水印{extra} ... ok "
          f"(watermarked={wm.watermarked_count}, fraction={dm.actual_rho:.2f})")

    model = dm.fit(wm)
    print(f"[3/6] 构建可疑模型（sim） ... ok (exposed={model.exposure_count})")

    calibration = dm.calibrate()
    print(f"[4/6] 阈值校准（负样本 {calibration['n_negative_models']} 个，"
          f"α={dm.config.fpr:g}） ... ok")

    print("[5/6] 多方法审计 ...")
    report = dm.audit(model, calibration=calibration)
    print(report.summary())

    out = Path(args.out)
    report_path = report.to_json(out / "report.json")
    print(f"[6/6] 报告已写入 {report_path}")
    elapsed = time.perf_counter() - started
    if elapsed > 30.0:  # pragma: no cover - 教学机性能兜底提示
        print(f"警告：demo 用时 {elapsed:.1f} 秒，超过 30 秒预算。", file=sys.stderr)
    return 0


def dataset_watermark_main(argv: list[str] | None = None) -> int:
    """教材网站 <ChapterToolkit> 卡片命令：一次跑完 init + embed + fit + audit。

    对外约定（与网站示例命令对应）::

        dataset-watermark --config configs/fugu_sim.json --report reports/chapter-3.md

    内部等价于依次执行 ``init`` → ``embed`` → ``fit(sim)`` → ``audit``，把审计
    报告导出为 Markdown（``--report``，网站侧可直接展示），同时写出 JSON 版本和
    中间产物。配置一律使用 JSON（项目不引入 YAML 依赖）。
    """
    parser = argparse.ArgumentParser(
        prog="dataset-watermark",
        description="DataMark：一次完成数据集水印嵌入、可疑模型模拟与多方法审计（教材 2.5）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=_default_config(), help="配置文件路径（JSON）")
    parser.add_argument("--report", default="reports/chapter-3.md",
                        help="Markdown 审计报告输出路径")
    parser.add_argument("--fraction", type=float, default=None,
                        help="覆盖带水印样本比例 ρ（0 < ρ ≤ 1）")
    parser.add_argument("--fpr", type=float, default=None, help="覆盖目标误报率 α")
    parser.add_argument("--methods", default="all",
                        help="要审计的方法（逗号分隔，默认全部已启用方法）")
    parser.add_argument("--out", default="work/dataset-watermark",
                        help="中间产物目录（配置快照、数据集、清单、报告 JSON）")
    args = parser.parse_args(argv)
    try:
        dm = DataMark.from_config(args.config)
        if args.fraction is not None:
            dm.set_fraction(float(args.fraction))
        if args.fpr is not None:
            dm.config.audit["fpr"] = float(args.fpr)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "config.resolved.json").open("w", encoding="utf-8") as f:
            json.dump(config_to_dict(dm.config), f, ensure_ascii=False, indent=2)
            f.write("\n")
        dm.key  # 触发密钥加载 / 创建（init 的等价物）
        print(f"[1/4] 初始化：配置 {args.config}，密钥 {dm.config.key_path} ... ok")

        wm = dm.embed(out_dir=out / "wm")
        print(f"[2/4] 嵌入水印 ... ok (docs={wm.n_docs}, "
              f"watermarked={wm.watermarked_count}, fraction={dm.actual_rho:.2f})")

        model = dm.fit(wm, mode="sim")
        print(f"[3/4] 构建可疑模型（sim） ... ok (exposed={model.exposure_count})")

        methods = None if not args.methods else _split_csv(args.methods)
        report = dm.audit(model, methods=methods)
        md_path = report.to_markdown(args.report)
        json_path = report.to_json(out / "report.json")
        print("[4/4] 多方法审计 ...")
        print(report.summary())
        print(report.footer())
        print(f"[dataset-watermark] Markdown 报告：{md_path}")
        print(f"[dataset-watermark] JSON 报告：{json_path}")
        return 0
    except (ConfigError, KeyFileError, PipelineError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"错误：找不到文件 {exc.filename or exc}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（中文帮助）。"""
    parser = argparse.ArgumentParser(
        prog="datamark",
        description="DataMark：数据集水印教学工具包（教材第二章 2.5 实践与应用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"DataMark {_VERSION}")
    sub = parser.add_subparsers(dest="command", metavar="子命令")

    p_version = sub.add_parser("version", help="打印版本信息")
    p_version.set_defaults(func=cmd_version)

    p_init = sub.add_parser("init", help="初始化工作目录与密钥")
    p_init.add_argument("--config", default=_default_config(), help="配置文件路径")
    p_init.add_argument("--out", default="work/fugu", help="工作目录")
    p_init.set_defaults(func=cmd_init)

    p_embed = sub.add_parser("embed", help="构造带水印数据集")
    p_embed.add_argument("--config", default=_default_config(), help="配置文件路径")
    p_embed.add_argument("--fraction", type=float, default=None,
                         help="覆盖带水印样本比例 ρ（0 < ρ ≤ 1）")
    p_embed.add_argument("--out", default="work/fugu/wm", help="输出目录")
    p_embed.set_defaults(func=cmd_embed)

    p_fit = sub.add_parser("fit", help="训练 / 构建可疑模型")
    p_fit.add_argument("--config", default=_default_config(), help="配置文件路径")
    p_fit.add_argument("--mode", default=None, choices=["sim", "mock", "hf", "local", "api"],
                       help="模型模式：sim/mock 为模拟；hf/local 为本地小模型")
    p_fit.add_argument("--fraction", type=float, default=None,
                       help="覆盖带水印样本比例 ρ")
    p_fit.add_argument("--out", default="work/fugu/model", help="输出目录")
    p_fit.set_defaults(func=cmd_fit)

    p_audit = sub.add_parser("audit", help="多方法审计并输出报告")
    p_audit.add_argument("--config", default=_default_config(), help="配置文件路径")
    p_audit.add_argument("--fpr", type=float, default=None, help="覆盖目标误报率 α")
    p_audit.add_argument("--fraction", type=float, default=None,
                         help="覆盖带水印样本比例 ρ")
    p_audit.add_argument("--methods", default="all",
                         help="要审计的方法（逗号分隔，默认全部已启用方法）")
    p_audit.add_argument("--out", default="work/fugu/report", help="输出目录")
    p_audit.set_defaults(func=cmd_audit)

    p_sweep = sub.add_parser("sweep", help="比例 × 清洗攻击扫描")
    p_sweep.add_argument("--config", default=_default_config(), help="配置文件路径")
    p_sweep.add_argument("--fractions", default="0.05,0.1,0.25,0.5,1.0",
                         help="带水印比例列表（逗号分隔）")
    p_sweep.add_argument("--attacks", default="none,dedup,paraphrase",
                         help="清洗攻击列表（逗号分隔）")
    p_sweep.add_argument("--trials", type=int, default=None,
                         help="每个组合的重复次数（默认读取配置）")
    p_sweep.add_argument("--out", default="work/fugu/sweep.csv", help="输出 CSV 路径")
    p_sweep.set_defaults(func=cmd_sweep)

    p_demo = sub.add_parser("demo", help="全流程演示（30 秒内）")
    p_demo.add_argument("--config", default=_default_config(), help="配置文件路径")
    p_demo.add_argument("--out", default="work/fugu_demo", help="演示输出目录")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主函数：解析参数并执行；异常统一转为中文错误信息。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (ConfigError, KeyFileError, PipelineError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"错误：找不到文件 {exc.filename or exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
