# -*- coding: utf-8 -*-
"""水印密钥生成、保存与加载的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datamark.keyring import (
    KEY_FORMAT,
    KeyFileError,
    derive_key,
    derive_subkey,
    key_file_content,
    load_key,
    load_or_create_key,
    save_key,
)


class TestKeyring(unittest.TestCase):
    """密钥必须可复现（教学）、与 seed 分离（概念）且能被加载校验。"""

    def test_derive_key_deterministic(self) -> None:
        key_a = derive_key(42)
        key_b = derive_key(42)
        self.assertEqual(key_a, key_b)
        self.assertEqual(len(key_a), 64)
        int(key_a, 16)  # 合法十六进制
        self.assertNotEqual(key_a, derive_key(43))

    def test_subkey_purpose_and_index(self) -> None:
        key = derive_key(42)
        self.assertNotEqual(derive_subkey(key, "canary", 0),
                            derive_subkey(key, "canary", 1))
        self.assertNotEqual(derive_subkey(key, "canary"),
                            derive_subkey(key, "token_bit"))
        self.assertEqual(derive_subkey(key, "canary"),
                         derive_subkey(key, "canary"))

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secret" / "fugu.key"
            content = key_file_content(42)
            save_key(path, content)
            loaded = load_key(path)
            self.assertEqual(loaded["key"], content["key"])
            self.assertEqual(loaded["format"], KEY_FORMAT)

    def test_load_or_create_reuses_existing_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fugu.key"
            first = load_or_create_key(path, seed=42)
            self.assertTrue(path.exists())
            # 手工改掉文件里的密钥，再次加载应沿用文件内容（密钥优先于 seed）
            material = key_file_content(42, "a" * 64)
            save_key(path, material)
            second = load_or_create_key(path, seed=42)
            self.assertEqual(second["key"], "a" * 64)
            self.assertEqual(first["seed"], 42)

    def test_missing_key_without_create_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.key"
            with self.assertRaises(KeyFileError):
                load_or_create_key(path, seed=1, create_if_missing=False)

    def test_corrupt_key_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.key"
            path.write_text(json.dumps({"nokey": "x"}), encoding="utf-8")
            with self.assertRaises(KeyFileError):
                load_key(path)


if __name__ == "__main__":
    unittest.main()
