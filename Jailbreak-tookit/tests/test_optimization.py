import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F

    from textbook_toolkit.jailbreak import get
    from textbook_toolkit.jailbreak.optimization import (
        GCG,
        AutoDANConfig,
        AutoDANSearch,
        COLDAttack,
        COLDConfig,
        GCGConfig,
        OptimizationRequest,
    )
    from textbook_toolkit.jailbreak.optimization.cold import ngram_energy
    from textbook_toolkit.jailbreak.optimization.demo import tiny_backend
    from textbook_toolkit.jailbreak.optimization.gcg import sample_candidates, token_gradients
except ImportError:
    torch = None


class OptionalImportTests(unittest.TestCase):
    def test_manual_import_does_not_load_heavy_dependencies(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import textbook_toolkit.jailbreak; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules",
            ],
            check=True,
        )


@unittest.skipIf(torch is None, "Install .[optimization] to run real model tests")
class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.backend = tiny_backend(chat=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def request(self, **kwargs):
        return OptimizationRequest(
            "Explain plants.", "Plants grow", initial_control="! !", **kwargs
        )

    def test_target_loss_matches_hf_masked_labels(self):
        backend = self.backend
        with backend.session():
            layout = backend.prepare(self.request(), "! !")
            ids = torch.cat((layout.input_ids, layout.target_ids))[None]
            labels = ids.clone()
            labels[:, : len(layout.input_ids)] = -100
            expected = backend.model(ids, labels=labels).loss
            measured = backend.losses([layout])[0]
            self.assertAlmostEqual(float(expected), float(measured), places=6)

    def test_one_hot_gradient_agrees_with_finite_difference(self):
        backend = self.backend
        with backend.session():
            layout = backend.prepare(self.request(), "! !")
            gradient = token_gradients(backend, layout)
            hot = F.one_hot(layout.control_ids, backend.vocab_size).float()
            indices = gradient[0].abs().topk(3).indices
            derivatives = []
            for index in indices:
                direction = torch.zeros_like(hot)
                direction[0, index] = 0.05
                with torch.no_grad():
                    plus = backend.target_loss(
                        backend.soft_forward(
                            layout, ((hot + direction) @ backend.embedding)[None]
                        ),
                        layout,
                    )
                    minus = backend.target_loss(
                        backend.soft_forward(
                            layout, ((hot - direction) @ backend.embedding)[None]
                        ),
                        layout,
                    )
                derivatives.append(float((plus - minus) / 0.1))
            numerical = torch.tensor(derivatives)
            analytic = gradient[0, indices].cpu()
            self.assertGreater(float(F.cosine_similarity(numerical, analytic, dim=0)), 0.999)

    def test_sampling_changes_exactly_one_position_and_avoids_specials(self):
        backend = self.backend
        with backend.session():
            layout = backend.prepare(self.request(), "! !")
            samples = sample_candidates(
                layout.control_ids,
                token_gradients(backend, layout),
                backend.allowed_tokens(),
                count=48,
                top_k=20,
                generator=torch.Generator().manual_seed(1),
            )
            self.assertTrue(((samples != layout.control_ids).sum(-1) == 1).all())
            self.assertTrue(backend.allowed_tokens()[samples].all())

    def test_gcg_end_to_end_repeatability_and_model_unchanged(self):
        backend = self.backend
        weights = [p.detach().clone() for p in backend.model.parameters()]
        flags = [p.requires_grad for p in backend.model.parameters()]
        rng = torch.get_rng_state().clone()
        method = get(
            "gcg",
            backend=backend,
            config=GCGConfig(
                steps=3, candidates=32, top_k=16, eval_batch_size=8, max_new_tokens=4
            ),
        )
        result = method(self.request())
        repeated = method(self.request())
        self.assertEqual(result.to_json(), repeated.to_json())
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertLessEqual(result.final_loss, result.initial_loss)
        self.assertGreater(len(result.trace), 1)
        self.assertEqual(
            result.input_ids, tuple(backend.encode_prompt(result.prompt.text).tolist())
        )
        self.assertTrue(result.response_ids)
        self.assertEqual([p.requires_grad for p in backend.model.parameters()], flags)
        for before, after in zip(weights, backend.model.parameters()):
            self.assertTrue(torch.equal(before, after))
            self.assertIsNone(after.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result.save(path)
            self.assertEqual(json.loads(path.read_text())["response"], result.response)
            with self.assertRaises(FileExistsError):
                result.save(path)

    def test_invalid_candidate_pool_returns_actual_seed_result(self):
        with patch.object(self.backend, "candidate", return_value=None):
            result = GCG(
                self.backend, GCGConfig(steps=2, candidates=4, top_k=4, max_new_tokens=2)
            )(self.request())
        self.assertEqual(result.stop_reason, "no_valid_candidates")
        self.assertEqual(result.control, "! !")
        self.assertTrue(result.response_ids)

    def test_unicode_and_boundaries_roundtrip(self):
        request = OptimizationRequest("解释植物 🌱", "Plants", initial_control="! !")
        layout = self.backend.prepare(request, "! !")
        candidate = self.backend.candidate(layout, layout.control_ids)
        self.assertEqual(candidate.request.compose(candidate.control), request.compose("! !"))
        self.assertEqual(candidate.input_ids.tolist(), layout.input_ids.tolist())

    def test_lm_top_k_controls_cold_decode(self):
        backend = self.backend
        method = COLDAttack(backend, COLDConfig(top_k=1, steps=1, max_new_tokens=2))
        with backend.session():
            layout = backend.prepare(self.request(), "! !")
            prefix = layout.input_ids[: layout.start][None]
            expected = []
            for _ in range(len(layout.control_ids)):
                next_logits = backend.logits(ids=prefix).logits[:, -1]
                next_logits[:, ~backend.allowed_tokens()] = -torch.inf
                chosen = next_logits.argmax(-1, keepdim=True)
                expected.append(int(chosen[0, 0]))
                prefix = torch.cat((prefix, chosen), dim=1)
            arbitrary = torch.zeros(1, len(layout.control_ids), backend.vocab_size)
            arbitrary[:, :, 15] = 10000
            self.assertEqual(method.decode(layout, arbitrary)[0].tolist(), expected)

    def test_cold_gradient_and_real_update(self):
        backend = self.backend
        config = COLDConfig(
            steps=3,
            batch_size=2,
            init="seed",
            top_k=12,
            eval_every=1,
            embedding_temperature=1,
            noise_std=0,
            noise_schedule=(),
            include=("plant",),
            exclude=("stone",),
            max_new_tokens=3,
        )
        method = get("cold_attack", backend=backend, config=config)
        with backend.session():
            layout = backend.prepare(self.request(), "! !")
            logits = method.initialize(
                layout, torch.Generator().manual_seed(3)
            ).requires_grad_()
            loss, _ = method.energies(layout, logits)
            (gradient,) = torch.autograd.grad(loss.mean(), logits)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        result = method(self.request())
        self.assertEqual(len(result.trace), 4)
        self.assertNotEqual(
            result.trace[1].terms["soft_objective"], result.trace[-1].terms["soft_objective"]
        )
        self.assertEqual(
            result.input_ids, tuple(backend.encode_prompt(result.prompt.text).tolist())
        )
        self.assertTrue(result.response_ids)
        self.assertEqual(result.to_json(), method(self.request()).to_json())

    def test_cold_control_context_in_output_and_generation(self):
        result = COLDAttack(
            self.backend,
            COLDConfig(
                mode="control",
                steps=1,
                batch_size=1,
                eval_every=1,
                embedding_temperature=1,
                max_new_tokens=2,
            ),
        )(self.request(right_context="Use one sentence."))
        self.assertTrue(result.prompt.text.endswith("Use one sentence."))
        self.assertIn("hard_coherence", result.trace[-1].terms)
        self.assertEqual(
            result.input_ids, tuple(self.backend.encode_prompt(result.prompt.text).tolist())
        )

    def test_paraphrase_reference_and_hidden_state_similarity(self):
        result = COLDAttack(
            self.backend,
            COLDConfig(
                mode="paraphrase",
                steps=1,
                batch_size=1,
                init="seed",
                eval_every=1,
                embedding_temperature=1,
                max_new_tokens=2,
            ),
        ).run("Plants grow.", "Plants")
        self.assertEqual(result.request.placement, "replacement")
        self.assertIn("soft_reference", result.trace[-1].terms)
        self.assertIn("soft_similarity", result.trace[-1].terms)
        self.assertTrue(result.response_ids)

    def test_ngram_alignment_known_optimum_and_gradient(self):
        logits = torch.tensor(
            [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]], requires_grad=True
        )
        ordered = ngram_energy(logits, torch.tensor([0, 1, 2]), 3)
        reversed_order = ngram_energy(logits, torch.tensor([2, 1, 0]), 3)
        self.assertLess(float(ordered.detach()), float(reversed_order.detach()))
        ordered.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_plain_mode_and_length_validation(self):
        backend = tiny_backend(chat=False)
        result = GCG(backend, GCGConfig(steps=1, candidates=4, max_new_tokens=2))(
            self.request()
        )
        self.assertEqual(result.input_ids[0], backend.tokenizer.bos_token_id)
        with self.assertRaisesRegex(ValueError, "context"):
            backend.prepare(OptimizationRequest("x" * 1100, "y"), "!")

    def test_model_flags_restore_after_exception(self):
        model = self.backend.model
        model.train()
        with self.assertRaisesRegex(RuntimeError, "test error"), self.backend.session():
            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            raise RuntimeError("test error")
        self.assertTrue(model.training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_invalid_configs_fail_before_model_execution(self):
        for factory in (
            lambda: GCGConfig(steps=0),
            lambda: COLDConfig(batch_size=-1),
            lambda: COLDConfig(noise_std=-1),
            lambda: COLDConfig(include="x"),
            lambda: COLDConfig(noise_schedule=((10, 1), (5, 1))),
            lambda: COLDConfig(embedding_temperature=float("nan")),
        ):
            with self.assertRaises((ValueError, TypeError)):
                factory()

    def test_autodan_hga_evaluates_mutation_and_synonym_offspring(self):
        calls = []

        def mutator(messages):
            calls.append(messages)
            prefix = json.loads(messages[-1].content)["prefix"]
            return prefix.replace("simple", "clear")

        config = AutoDANConfig(
            steps=4,
            population_size=4,
            paragraph_interval=3,
            word_replacement_probability=1,
            mutation_probability=1,
            max_new_tokens=2,
        )
        method = AutoDANSearch(
            self.backend,
            mutator,
            config,
            synonyms=lambda word: ("plain",) if word == "clear" else (),
        )
        result = method.run("Explain plants.", "Plants", prototype="Use simple words. Explain:")
        self.assertEqual(len(result.trace), 5)
        self.assertTrue(result.prompt.text.endswith("Explain plants."))
        self.assertGreater(len(calls), 3)
        self.assertLessEqual(result.final_loss, result.initial_loss)
        self.assertTrue(result.response_ids)

    def test_autodan_momentum_and_uneven_crossover(self):
        import random

        from textbook_toolkit.jailbreak.optimization.autodan import crossover

        method = AutoDANSearch(
            self.backend, lambda messages: "prefix", AutoDANConfig(momentum=0.5)
        )
        scores = {}
        method.update_momentum(["clear words", "clear idea"], [2, 4], scores)
        self.assertEqual(scores["clear"], -3)
        method.update_momentum(["clear words"], [1], scores)
        self.assertEqual(scores["clear"], -2)
        first, second = crossover("A. B.\n\nKeep tail.", "C. D.", 5, random.Random(0))
        self.assertIn("Keep tail.", first)
        self.assertTrue(second)

    def test_local_model_adapts_to_text_generator(self):
        from textbook_toolkit.jailbreak import Message

        generated = self.backend.text_generator(max_new_tokens=2)(
            (Message("user", "Explain plants."),)
        )
        self.assertIsInstance(generated, str)
