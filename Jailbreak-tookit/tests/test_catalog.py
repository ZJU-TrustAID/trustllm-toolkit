import unittest

from textbook_toolkit import jailbreak as jb
from textbook_toolkit.core import Method, Registry


class CatalogTests(unittest.TestCase):
    def test_each_registered_builtin_has_an_executable_assembly_path(self):
        def generator(messages):
            return "候选：{query}"

        options = {
            "template": {"template": "[{query}]"},
            "rule_rewrite": {"rules": ("使用中文",)},
            "context": {"history": ()},
            "autodan": {"generator": generator, "prototype": "{query}"},
            "red_teaming": {"generator": generator},
        }
        for name in jb.list_methods():
            if name in ("gcg", "cold_attack"):
                continue  # Real optimizer tests live in test_optimization.py.
            with self.subTest(name=name):
                method = jb.get(name, **options.get(name, {}))
                self.assertIsInstance(method, Method)
                prompt = method.build("解释光合作用")
                self.assertEqual(prompt.query, "解释光合作用")
                self.assertEqual(prompt.messages[-1].role, "user")
                self.assertTrue(jb.describe(name))

    def test_method_names_are_normalized_but_unknowns_fail(self):
        self.assertIsInstance(jb.get(" Role-Play "), jb.RolePlay)
        with self.assertRaisesRegex(KeyError, "Available"):
            jb.get("does-not-exist")
        self.assertEqual(len(jb.list_methods()), 11)
        self.assertNotIn("pair", jb.list_methods())

    def test_sibling_registry_uses_independent_types_and_names(self):
        class Double(Method[int, int]):
            def build(self, value):
                return value * 2

        chapter = Registry()
        chapter.register("demo", Double, description="another chapter")
        self.assertEqual(chapter.get("demo")(4), 8)
        self.assertNotIn("demo", jb.list_methods())
        with self.assertRaises(ValueError):
            chapter.register("DEMO", Double)

    def test_registry_creates_fresh_instances(self):
        registry = Registry()
        registry.register("method", jb.Template)
        first = registry.get("method", template="A {query}")
        second = registry.get("method", template="B {query}")
        self.assertEqual(first("x").text, "A x")
        self.assertEqual(second("x").text, "B x")


if __name__ == "__main__":
    unittest.main()
