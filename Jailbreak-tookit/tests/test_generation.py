import json
import unittest

from textbook_toolkit import jailbreak as jb


class RecordingGenerator:
    def __init__(self, reply="改写：解释光合作用"):
        self.reply, self.calls = reply, []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.reply

    @property
    def payloads(self):
        return [json.loads(messages[-1].content) for messages in self.calls]


class GenerationTests(unittest.TestCase):
    def test_red_teaming_examples_only_go_to_generator(self):
        examples = ["示例一", "示例二"]
        generator = RecordingGenerator()
        method = jb.RedTeaming(generator, examples)
        examples.clear()
        prompt = method.build("解释光合作用")
        self.assertEqual(generator.payloads[0]["examples"], ["示例一", "示例二"])
        self.assertEqual(prompt.text, generator.reply)
        self.assertEqual(prompt.steps[0].operation, "few_shot_generation")

    def test_autodan_mutates_before_substituting_query(self):
        generator = RecordingGenerator("新模板：{query}")
        results = jb.AutoDAN(generator, "原模板：{query}").population('JSON {"a":1}', size=2)
        self.assertEqual(results[0].text, '新模板：JSON {"a":1}')
        self.assertEqual([p["candidate_index"] for p in generator.payloads], [0, 1])
        self.assertNotIn("query", generator.payloads[0])

    def test_model_cannot_silently_drop_query_placeholder(self):
        method = jb.AutoDAN(RecordingGenerator("missing placeholder"), "{query}")
        with self.assertRaisesRegex(ValueError, "query"):
            method.build("题目")

    def test_provider_errors_propagate_and_invalid_responses_fail(self):
        error = RuntimeError("provider unavailable")

        def failing_generator(messages):
            raise error

        with self.assertRaises(RuntimeError) as caught:
            jb.RedTeaming(failing_generator).build("题目")
        self.assertIs(caught.exception, error)
        for reply in (None, {"content": "x"}, "", "  "):
            with self.subTest(reply=reply), self.assertRaises((TypeError, ValueError)):
                jb.RedTeaming(RecordingGenerator(reply)).build("题目")

    def test_invalid_counts_do_not_call_generator(self):
        generator = RecordingGenerator()
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                jb.AutoDAN(generator, "{query}").population("题目", size=count)
        self.assertEqual(generator.calls, [])

    def test_crossover_preserves_sentences_and_complementary_parents(self):
        left = ("L1.", "L2。", "L3!", "L4?")
        right = ("R1.", "R2。", "R3!", "R4?")
        first, second = jb.sentence_crossover(left, right, cuts=3, seed=42)
        self.assertEqual((first, second), jb.sentence_crossover(left, right, cuts=3, seed=42))
        for index in range(4):
            self.assertEqual({first[index], second[index]}, {left[index], right[index]})
        with self.assertRaises(ValueError):
            jb.sentence_crossover(left, right[:2])
