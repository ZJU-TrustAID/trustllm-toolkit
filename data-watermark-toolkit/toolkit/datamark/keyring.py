# -*- coding: utf-8 -*-
"""水印密钥的生成、保存与加载。

为什么密钥与随机种子要分开？
--------------------------
* **随机种子**只决定实验过程中的抽样顺序（哪几条文档做金丝雀、噪声怎么取），
  公开种子不会泄露水印规则；
* **水印密钥**决定「哪些词元对算 bit-1」「金丝雀编号是什么」「隐藏指令标记是什么」
  等嵌入/检测规则。检测方必须持有同一把密钥才能复现规则，因此密钥属于秘密材料，
  应与数据集、水印清单一起单独保管。

教学复现的取舍
--------------
真实系统应使用 ``os.urandom`` 生成密钥。本工具包为了让「同一配置 + 同一 seed
⇒ 报告逐字节可复现」，默认由 seed 经 SHA-256 确定性派生密钥，并把派生种子写进
密钥文件；密钥文件一旦存在就优先沿用文件的密钥（即密钥材料可以固定下来）。
这是教学级简化，README「已知限制」中已注明。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

KEY_FORMAT = "datamark-key-v1"


class KeyFileError(ValueError):
    """密钥文件损坏或缺失时抛出（避免与内建 KeyError 混淆）。"""


def derive_key(seed: int, label: str = "datamark") -> str:
    """由 seed 确定性派生 256-bit 水印密钥（返回 64 位十六进制字符串）。

    注意这只用于**可复现的教学实验**；真实部署应改用 CSPRNG。
    """
    payload = f"{KEY_FORMAT}|{label}|seed={seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_subkey(key: str, purpose: str, index: int = 0) -> int:
    """从主密钥派生某方法/某一项使用的子密钥（整数）。

    :param key: 64 位十六进制主密钥。
    :param purpose: 用途标签，例如 ``"canary"`` / ``"token_bit"``。
    :param index: 同一用途内的序号。
    """
    payload = f"{key}|{purpose}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def key_file_content(seed: int, key: str | None = None) -> dict:
    """构造密钥文件内容（纯 dict，便于测试与确定性比较）。"""
    material = key if key is not None else derive_key(seed)
    return {
        "format": KEY_FORMAT,
        "seed": int(seed),
        "key": material,
        "note": "教学用确定性派生密钥；真实部署请改用 os.urandom 并单独保管",
    }


def save_key(path: str | Path, content: dict) -> Path:
    """保存密钥文件（JSON，权限尽量收紧为 0600）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
        f.write("\n")
    try:  # 仅做努力尝试：某些文件系统不支持 chmod
        os.chmod(p, 0o600)
    except OSError:  # pragma: no cover - 平台相关
        pass
    return p


def load_key(path: str | Path) -> dict:
    """加载并校验密钥文件。"""
    p = Path(path)
    if not p.exists():
        raise KeyFileError(f"密钥文件不存在：{p}")
    with p.open("r", encoding="utf-8") as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise KeyFileError(f"密钥文件不是合法 JSON：{p}（{exc}）") from exc
    if not isinstance(content, dict) or "key" not in content:
        raise KeyFileError(f"密钥文件缺少 key 字段：{p}")
    key = str(content["key"])
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise KeyFileError(f"密钥格式非法（应为 64 位十六进制）：{p}")
    return content


def load_or_create_key(path: str | Path, seed: int,
                       create_if_missing: bool = True) -> dict:
    """加载密钥；不存在且允许创建时，由 seed 确定性派生并写盘。

    :returns: 密钥文件内容 dict（至少含 ``key`` / ``seed`` / ``format``）。
    """
    p = Path(path)
    if p.exists():
        return load_key(p)
    if not create_if_missing:
        raise KeyFileError(f"密钥文件不存在且未开启自动创建：{p}")
    content = key_file_content(seed)
    save_key(p, content)
    return content
