import unittest

from textbook_toolkit import jailbreak as jb


class ArtifactTests(unittest.TestCase):
    def test_suffix_whitespace_is_preserved_exactly(self):
        fragment = jb.Fragment("  后缀\n", source="local experiment")
        prompt = jb.OptimizedPrompt(fragment=fragment, separator="").build("问题 ")
        self.assertEqual(prompt.text, "问题   后缀\n")
        self.assertEqual(prompt.steps[0].source, "local experiment")
        self.assertEqual(prompt.steps[0].operation, "assemble_suffix")

    def test_artifact_supports_all_text_placements(self):
        for placement, expected in (
            ("prefix", "片段|问题"),
            ("suffix", "问题|片段"),
            ("replacement", "片段"),
        ):
            with self.subTest(placement=placement):
                method = jb.OptimizedPrompt(
                    fragment=jb.Fragment("片段", placement), separator="|"
                )
                self.assertEqual(method("问题").text, expected)

    def test_artifact_rejects_wrong_placement(self):
        method = jb.OptimizedPrompt(
            fragment=jb.Fragment("片段", "prefix"), allowed_placements=("suffix",)
        )
        with self.assertRaisesRegex(ValueError, "placements"):
            method.build("问题")

    def test_query_specific_fragment_cannot_be_reused_accidentally(self):
        method = jb.OptimizedPrompt(fragment=jb.Fragment("片段", for_query="原问题"))
        self.assertEqual(method.build("原问题").query, "原问题")
        with self.assertRaisesRegex(ValueError, "different query"):
            method.build("另一个问题")

    def test_provider_receives_query_once_per_build(self):
        calls = []

        def provider(query):
            calls.append(query)
            return jb.Fragment("附加说明", for_query=query, source="local provider")

        method = jb.OptimizedPrompt(fragment=provider)
        self.assertEqual(calls, [])  # Registration and construction have no inference effects.
        result = method.build("题目")
        self.assertEqual(calls, ["题目"])
        self.assertEqual(result.text, "题目 附加说明")

    def test_raw_arrays_and_wrong_provider_results_are_rejected(self):
        with self.assertRaises(ValueError):
            jb.Fragment([1, 2, 3])
        with self.assertRaises(TypeError):
            jb.OptimizedPrompt(lambda query: "not a Fragment").build("题目")

    def test_provider_errors_are_not_masked(self):
        def provider(query):
            raise RuntimeError("external optimizer failed")

        with self.assertRaisesRegex(RuntimeError, "external optimizer"):
            jb.OptimizedPrompt(provider).build("题目")


if __name__ == "__main__":
    unittest.main()
