import base64
import json
import tempfile
import unittest
from pathlib import Path

from textbook_toolkit import jailbreak as jb


class ManualTests(unittest.TestCase):
    def test_template_keeps_query_braces_literal(self):
        query = '解释 JSON：{"key": "{query}"}'
        result = jb.Template("模板 {{example}}：{query}").build(query)
        self.assertEqual(result.text, "模板 {example}：" + query)
        self.assertEqual(result.query, query)

    def test_templates_reject_missing_or_executable_looking_fields(self):
        for template in ("no slot", "{query.__class__}", "{query[0]}", "{query!r}", "{x}", "{"):
            with self.subTest(template=template), self.assertRaises(ValueError):
                jb.Template(template)

    def test_utf8_base64_round_trip(self):
        query = "解释光合作用 🌱\n保留换行"
        result = jb.Cipher().build(query)
        encoded = result.text.split("\n\n", 1)[1]
        self.assertEqual(base64.b64decode(encoded).decode("utf-8"), query)

    def test_caesar_rotates_only_ascii_and_wraps(self):
        self.assertEqual(jb.Cipher("caesar", 3).encode("XyZ 中文 123!"), "AbC 中文 123!")
        self.assertEqual(jb.Cipher("caesar", -3).encode("AbC"), "XyZ")

    def test_rules_stay_in_user_role(self):
        result = jb.RuleRewrite(("使用中文", "分两段说明")).build("解释光合作用")
        self.assertEqual([m.role for m in result.messages], ["user"])
        self.assertIn("1. 使用中文\n2. 分两段说明", result.text)

    def test_context_preserves_roles_and_defensively_copies_history(self):
        history = [jb.Message("system", "课堂助手"), jb.Message("user", "你好")]
        method = jb.Context(history)
        history.clear()
        result = method.build("解释光合作用")
        self.assertEqual(len(result.messages), 3)
        with self.assertRaisesRegex(ValueError, "conversation"):
            _ = result.text
        exported = result.to_messages()
        exported[0]["content"] = "changed"
        self.assertEqual(result.messages[0].content, "课堂助手")

    def test_composition_preserves_original_query_trace_and_final_history(self):
        method = jb.Template("题目：{query}").then(jb.Suffix("请简明回答。"))
        method = method.then(jb.Context((jb.Message("system", "课堂助手"),)))
        result = method.build("解释光合作用")
        self.assertEqual(result.query, "解释光合作用")
        self.assertEqual(result.messages[-1].content, "题目：解释光合作用 请简明回答。")
        self.assertEqual([s.method for s in result.steps], ["template", "suffix", "context"])
        self.assertEqual(len(method.stages), 3)

    def test_pipeline_cannot_drop_history(self):
        method = jb.Context((jb.Message("system", "课堂助手"),)).then(jb.Suffix("!"))
        with self.assertRaisesRegex(ValueError, "conversation"):
            method.build("题目")

    def test_empty_or_invalid_input_fails(self):
        for query in ("", "  ", None, 42):
            with self.subTest(query=query), self.assertRaises(ValueError):
                jb.RolePlay().build(query)
        with self.assertRaises(TypeError):
            jb.RolePlay().build_many("one query")
        with self.assertRaises(ValueError):
            jb.Pipeline(())

    def test_batch_retains_input_order(self):
        results = jb.Template("[{query}]").build_many(iter(["甲", "乙"]))
        self.assertEqual([r.text for r in results], ["[甲]", "[乙]"])


class SerializationTests(unittest.TestCase):
    def test_unicode_conversation_round_trip_and_no_overwrite(self):
        history = (jb.Message("user", "你好"), jb.Message("assistant", "你好！"))
        prompt = jb.Context(history).build("解释光合作用 🌱")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.json"
            prompt.save(path)
            self.assertEqual(jb.Prompt.load(path), prompt)
            with self.assertRaises(FileExistsError):
                jb.Template("{query}").build("new").save(path)
            self.assertEqual(jb.Prompt.load(path), prompt)
            prompt.save(path, overwrite=True)

    def test_invalid_artifact_shapes_are_rejected(self):
        valid = jb.Template("{query}").build("题目").to_dict()
        for changes in (
            {"schema_version": 2},
            {"schema_version": True},
            {"messages": []},
            {"messages": [{"role": "tool", "content": "a"}]},
            {"messages": [{"role": "assistant", "content": "a"}]},
            {"messages": [42]},
            {"steps": "wrong"},
            {"query": None},
            {"unknown": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                jb.Prompt.from_json(json.dumps({**valid, **changes}))

    def test_system_role_is_only_allowed_first(self):
        with self.assertRaises(ValueError):
            jb.Context((jb.Message("user", "a"), jb.Message("system", "b")))


if __name__ == "__main__":
    unittest.main()
