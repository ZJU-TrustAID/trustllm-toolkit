# -*- coding: utf-8 -*-
"""语料读写、切分与词元化（只依赖标准库）。

数据集格式：JSON Lines，每行一个对象 ``{"doc_id": "fugu-0001", "text": "..."}``。
词元化采用「简单正则」策略：

* 每个汉字单独成词元（中文按字切分，便于统计相邻词元对）；
* 连续数字（含小数）整体成词元，如 ``23.5``；
* 连续英文字母整体成词元，如 ``STAMP``；
* 其余非空白字符（标点、符号）各自成词元。

这一策略足够支撑 bigram 语言模型、相邻词元对哈希、BM25 检索等教学实验，
不需要分词词典，也不会引入第三方依赖。
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

#: 词元正则：优先级依次为 数字 → 英文 → 单字/单符号。
TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?|[A-Za-z]+|[\u4e00-\u9fff]|[^\s\u4e00-\u9fff\dA-Za-z]"
)

#: CJK 字符正则（用于统计汉字数）。
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class CorpusError(ValueError):
    """语料文件格式错误。"""


def tokenize(text: str) -> list[str]:
    """把文本切成词元列表（见模块文档的切分规则）。"""
    return TOKEN_RE.findall(text or "")


def detokenize(tokens: Iterable[str]) -> str:
    """把词元列表拼回文本。

    汉字与标点直接相连；两个「字母/数字」词元之间补一个空格，
    避免 ``23.5`` 与 ``N`` 这类词元粘连。
    """
    out: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        if (out and tok[0].isascii() and tok[0].isalnum()
                and out[-1][-1].isascii() and out[-1][-1].isalnum()):
            out.append(" ")
        out.append(tok)
    return "".join(out)


def cjk_length(text: str) -> int:
    """返回文本中的汉字个数。"""
    return len(CJK_RE.findall(text or ""))


# ---------------------------------------------------------------------------
# JSONL 读写
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    """读取 JSONL 语料文件，校验每行含 ``doc_id`` 与 ``text``。"""
    p = Path(path)
    if not p.exists():
        raise CorpusError(f"找不到语料文件：{p}")
    docs: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{p}:{lineno} 不是合法 JSON（{exc}）") from exc
            if not isinstance(obj, dict) or "doc_id" not in obj or "text" not in obj:
                raise CorpusError(f"{p}:{lineno} 缺少 doc_id 或 text 字段")
            docs.append({"doc_id": str(obj["doc_id"]), "text": str(obj["text"])})
    if not docs:
        raise CorpusError(f"语料文件为空：{p}")
    _check_unique_ids(docs, p)
    return docs


def write_jsonl(path: str | Path, docs: Iterable[dict]) -> Path:
    """写出 JSONL 语料文件（UTF-8，不转义中文）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return p


def _check_unique_ids(docs: list[dict], source: object = "") -> None:
    seen: set[str] = set()
    for doc in docs:
        did = doc["doc_id"]
        if did in seen:
            raise CorpusError(f"{source}: doc_id 重复：{did}")
        seen.add(did)


# ---------------------------------------------------------------------------
# 切分与采样
# ---------------------------------------------------------------------------
def split_docs(docs: list[dict], n_train: int | None = None,
               n_heldout: int | None = None) -> tuple[list[dict], list[dict]]:
    """按顺序切分训练 / 留出文档（不随机，保证可复现）。

    :param docs: 文档列表。
    :param n_train: 训练文档数，默认取前 80%。
    :param n_heldout: 留出文档数，默认取剩余部分。
    """
    total = len(docs)
    train_n = int(n_train) if n_train is not None else int(total * 0.8)
    train_n = max(1, min(train_n, total))
    held = docs[train_n:]
    if n_heldout is not None:
        held = held[: int(n_heldout)]
    return docs[:train_n], held


def sample_docs(docs: list[dict], k: int, rng: random.Random) -> list[dict]:
    """从文档列表中无放回抽取 k 条（不足时全取）。"""
    k = max(0, min(int(k), len(docs)))
    return rng.sample(docs, k)


def token_counts(docs: Iterable[dict]) -> int:
    """统计一组文档的总词元数。"""
    return sum(len(tokenize(d["text"])) for d in docs)


def find_doc(docs: Iterable[dict], doc_id: str) -> dict | None:
    """按 doc_id 查找文档。"""
    for doc in docs:
        if doc["doc_id"] == doc_id:
            return doc
    return None


def replace_text(docs: list[dict], doc_id: str, new_text: str) -> None:
    """就地替换某文档的文本（找不到时静默忽略）。"""
    for doc in docs:
        if doc["doc_id"] == doc_id:
            doc["text"] = new_text
            return
