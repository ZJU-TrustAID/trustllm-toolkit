#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 DataMark 教学用的合成凤尾鱼语料（可复现，无网络、无第三方依赖）。

背景：教材 2.1 节以「水产研究机构的凤尾鱼观测数据」作为贯穿案例。真实观测
数据涉及机构版权，不能随教学工具包分发，因此这里用模板 + 固定随机种子合成
一份**领域风格**的观测记录数据集：

* ``data/fugu_corpus.jsonl``     —— 受保护语料，96 条「凤尾鱼观察」短文档；
* ``data/fugu_public_mix.jsonl`` —— 公共风格混合语料，50 条非水产领域文档，
  用作混合训练语料与对照（负样本）文本。

所有数字均为合成，不对应任何真实水域、机构或个人；生成过程只依赖标准库，
同一随机种子逐字节复现。

用法::

    python scripts/make_synthetic_corpus.py [--out-dir data] [--seed 20260801]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# 只在断言里用到了正则统计汉字数，保持中文语料 30–80 字（汉字计）。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def cjk_len(text: str) -> int:
    """返回字符串中的汉字个数（用于校验语料长度约束）。"""
    return len(_CJK_RE.findall(text))


# ---------------------------------------------------------------------------
# 凤尾鱼观测语料
# ---------------------------------------------------------------------------

SITES = [
    "东湾观测站", "南浦码头", "青屿近岸", "白砂湾", "芦潮港", "外沙渔场",
    "竹岔岛", "月牙湾", "小洋山", "嵊泗东侧", "三沙湾", "胶州湾北口",
]
WEATHER = ["晴", "多云", "阴", "小雨", "东北风三级", "西南风二级", "海雾"]
DIRECTIONS = ["北", "南", "东", "西", "东北", "西南"]
TRENDS = ["上升", "下降", "基本持平"]

# 主句模板：每条文档先取一个主句，再按需追加补充句，直到长度达标。
MAIN_TEMPLATES = [
    "{year}年{month}月{day}日，{site}记录到凤尾鱼{count}尾，平均体长{length}毫米。",
    "{site}完成一次拖网调查，共捕获凤尾鱼{count}尾，水温{temp}摄氏度，盐度{sal}。",
    "在{site}近岸设网{hours}小时，凤尾鱼渔获{count}尾，幼鱼比例约{ratio}%。",
    "夜间灯光诱集试验中，{site}的凤尾鱼在{hour}时前后集群，估计密度{density}尾每立方米。",
    "胃含物分析显示，{site}采集的{count}尾凤尾鱼主要摄食桡足类与小型甲壳类。",
    "标记放流试验在{site}完成，共标记凤尾鱼{count}尾，平均体长{length}毫米。",
    "{site}水温连续{days}天高于{temp}摄氏度，凤尾鱼产卵场向{direction}移动约{distance}公里。",
    "对{site}的凤尾鱼样本测量鳞片年轮，优势年龄组为{age}龄，平均体重{weight}克。",
    "声学调查覆盖{site}水域{area}平方公里，凤尾鱼目标强度集中在{db}分贝。",
    "本周{site}完成{count}尾凤尾鱼的生物学测定，成鱼比例约{ratio}%。",
    "受{weather}影响，{site}的凤尾鱼群在{depth}米水层聚集，围网作业{count}尾。",
    "连续观测表明，{site}凤尾鱼资源量约{stock}吨，较上月{trend}。",
    "{site}的定点观测显示，凤尾鱼群在{depth}米水层活动，水温{temp}摄氏度。",
    "研究人员在{site}采集凤尾鱼{count}尾，测量体长、体重与性腺成熟度。",
]

# 补充句模板：自然记录口吻，不含额外随机细节，便于控制长度。
EXTRA_SENTENCES = [
    "{site}的观测员同步记录了水色与透明度。",
    "现场测得溶解氧{dod}毫克每升，海面能见度良好。",
    "样本已按编号冷冻保存，等待实验室复测。",
    "声学仪器同时记录了鱼群的回声轨迹。",
    "近岸水温较上周上升约{rising}摄氏度。",
    "幼鱼与成鱼的比例接近一比三。",
    "该批数据已录入水产观测台账供后续分析。",
    "网具规格与作业时长按渔业规范记录在案。",
    "海面风浪较小，调查船按计划完成全部站位。",
    "研究组计划在下个航次继续跟踪该鱼群。",
    "水样与生物样本已分别编号存放。",
    "当日的潮汐与风向信息一并归档。",
]

# 公共风格语料：与水产无关的通用文本，作为混合语料/对照。
PUBLIC_SUBJECTS = [
    "社区图书馆", "城市公交", "地铁早高峰", "天气预报", "茶园管理",
    "山地徒步", "小学科普", "博物馆展览", "街道绿化", "社区食堂",
    "快递驿站", "湿地公园", "雨水回收", "屋顶光伏", "旧书交换",
    "夜间市集", "城市骑行", "河道清理", "社区门诊", "老年大学",
]
PUBLIC_TEMPLATES = [
    "{subject}在周末延长开放时间，居民可以通过线上平台预约。",
    "{subject}本周完成例行维护，工作人员提醒市民留意公告。",
    "关于{subject}的问卷调查已经结束，结果将在下月公布。",
    "{subject}的志愿者团队招募新成员，报名截止到本月底。",
    "记者走访了{subject}，发现高峰时段人流明显增加。",
    "{subject}附近新增两处便民设施，受到周边居民欢迎。",
    "为提升服务质量，{subject}计划开展一次意见征集活动。",
    "{subject}的运营时间在节假日期间有所调整，请提前查询。",
    "社区围绕{subject}组织了一场讲座，参与者讨论热烈。",
    "{subject}的年度报告显示，使用人数比去年有所增长。",
]


def _make_value(rng: random.Random, kind: str) -> str:
    """按字段类型生成字符串化的合成观测值。"""
    if kind == "year":
        return str(rng.choice([2021, 2022, 2023, 2024, 2025]))
    if kind == "month":
        return str(rng.randint(1, 12))
    if kind == "day":
        return str(rng.randint(1, 28))
    if kind == "site":
        return rng.choice(SITES)
    if kind == "count":
        return str(rng.randrange(12, 241, 4))
    if kind == "length":
        return str(rng.randint(78, 165))
    if kind == "temp":
        return f"{rng.uniform(8.0, 26.5):.1f}"
    if kind == "sal":
        return f"{rng.uniform(22.0, 34.5):.1f}"
    if kind == "hours":
        return str(rng.randint(2, 6))
    if kind == "ratio":
        return str(rng.randint(5, 45))
    if kind == "hour":
        return str(rng.randint(18, 23))
    if kind == "density":
        return f"{rng.uniform(0.3, 4.2):.1f}"
    if kind == "days":
        return str(rng.randint(3, 12))
    if kind == "direction":
        return rng.choice(DIRECTIONS)
    if kind == "distance":
        return str(rng.randint(3, 40))
    if kind == "age":
        return str(rng.choice([0, 1, 2, 3]))
    if kind == "weight":
        return str(rng.randint(6, 46))
    if kind == "area":
        return str(rng.randint(12, 260))
    if kind == "db":
        return f"-{rng.randint(38, 62)}"
    if kind == "depth":
        return str(rng.randint(2, 35))
    if kind == "stock":
        return str(rng.randrange(40, 880, 10))
    if kind == "trend":
        return rng.choice(TRENDS)
    if kind == "weather":
        return rng.choice(WEATHER)
    if kind == "dod":
        return f"{rng.uniform(4.5, 9.8):.1f}"
    if kind == "rising":
        return f"{rng.uniform(0.2, 1.8):.1f}"
    if kind == "subject":
        return rng.choice(PUBLIC_SUBJECTS)
    raise ValueError(f"未知字段类型：{kind}")


_SLOT_RE = re.compile(r"\{(\w+)\}")


def render(template: str, rng: random.Random) -> str:
    """把模板中的 {slot} 用合成值替换。"""
    return _SLOT_RE.sub(lambda m: _make_value(rng, m.group(1)), template)


def make_fugu_docs(rng: random.Random, n: int) -> list[dict]:
    """生成 n 条凤尾鱼观测短文档，每条 30–80 个汉字。"""
    docs: list[dict] = []
    for i in range(n):
        main = render(rng.choice(MAIN_TEMPLATES), rng)
        text = main
        # 依次追加补充句，直到汉字数达到 34；最长不超过 78 字。
        extras = EXTRA_SENTENCES[:]
        rng.shuffle(extras)
        for extra in extras:
            if cjk_len(text) >= 34:
                break
            cand = render(extra, rng)
            if cjk_len(text) + cjk_len(cand) <= 78:
                text += cand
        assert 30 <= cjk_len(text) <= 80, (i, cjk_len(text), text)
        docs.append({"doc_id": f"fugu-{i + 1:04d}", "text": text})
    return docs


def make_public_docs(rng: random.Random, n: int) -> list[dict]:
    """生成 n 条公共风格短文档（非水产领域），作为混合语料与负样本文本。"""
    docs: list[dict] = []
    seen: set[str] = set()
    i = 0
    guard = 0
    while len(docs) < n:
        guard += 1
        if guard > 20 * n:  # 防御性上限，正常不会触发
            break
        text = render(rng.choice(PUBLIC_TEMPLATES), rng)
        # 偶发追加一句，增加长度与多样性。
        if rng.random() < 0.5:
            text += render(rng.choice(PUBLIC_TEMPLATES), rng)
        if text in seen:
            continue
        seen.add(text)
        i += 1
        docs.append({"doc_id": f"public-{i:04d}", "text": text})
    return docs


def write_jsonl(path: Path, docs: list[dict]) -> None:
    """按「一行一条 JSON」写入，UTF-8、不转义中文（便于阅读与 diff）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 DataMark 合成凤尾鱼教学语料")
    parser.add_argument("--out-dir", default="data", help="输出目录（默认 data）")
    parser.add_argument("--seed", type=int, default=20260801, help="生成随机种子")
    parser.add_argument("--n-fugu", type=int, default=96, help="凤尾鱼文档数（80–120）")
    parser.add_argument("--n-public", type=int, default=50, help="公共混合文档数（40–60）")
    args = parser.parse_args(argv)

    if not 80 <= args.n_fugu <= 120:
        parser.error("--n-fugu 必须在 80–120 之间（书稿要求）")
    if not 40 <= args.n_public <= 60:
        parser.error("--n-public 必须在 40–60 之间（书稿要求）")

    rng = random.Random(args.seed)
    fugu = make_fugu_docs(rng, args.n_fugu)
    public = make_public_docs(rng, args.n_public)

    out = Path(args.out_dir)
    write_jsonl(out / "fugu_corpus.jsonl", fugu)
    write_jsonl(out / "fugu_public_mix.jsonl", public)

    lengths = [cjk_len(d["text"]) for d in fugu]
    print(f"已生成 {len(fugu)} 条凤尾鱼文档 -> {out / 'fugu_corpus.jsonl'}")
    print(f"  汉字数范围：{min(lengths)}–{max(lengths)}（要求 30–80）")
    print(f"已生成 {len(public)} 条公共混合文档 -> {out / 'fugu_public_mix.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
