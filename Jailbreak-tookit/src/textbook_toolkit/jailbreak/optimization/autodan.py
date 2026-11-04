"""AutoDAN HGA integrated with the same target-loss backend as GCG and COLD."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch

from textbook_toolkit.core import Method

from ..generation import generate_text
from ..types import require_count, require_text
from .backend import CausalLM
from .types import OptimizationRequest, OptimizationResult, TraceStep

SOURCE = {
    "repository": "https://github.com/SheltonLiu-N/AutoDAN",
    "commit": "34062e964185693e81a6775b4f0d00bfd7507612",
    "paper": "https://arxiv.org/abs/2310.04451",
}


@lru_cache(maxsize=4096)
def wordnet_synonyms(word: str) -> tuple[str, ...]:
    """English defaults; inject a synonym function for another language/domain."""
    try:
        from nltk.corpus import wordnet

        return tuple(
            sorted(
                {
                    lemma.name().replace("_", " ")
                    for sense in wordnet.synsets(word.lower())
                    for lemma in sense.lemmas()
                }
                - {word.lower()}
            )
        )
    except (ImportError, LookupError) as exc:
        raise RuntimeError(
            "Install .[evolution] and run python -m nltk.downloader wordnet, "
            "or provide synonyms(word) explicitly."
        ) from exc


@dataclass(frozen=True)
class AutoDANConfig:
    steps: int = 100
    population_size: int = 20
    elite_fraction: float = 0.1
    paragraph_interval: int = 5
    crossover_probability: float = 0.5
    crossover_points: int = 5
    mutation_probability: float = 0.01
    word_replacement_probability: float = 0.5
    momentum: float = 0.5
    max_word_replacements: int = 5
    eval_batch_size: int = 8
    loss_threshold: float | None = None
    seed: int = 0
    max_new_tokens: int = 64
    stop_words: tuple[str, ...] = (
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "as",
        "at",
        "by",
        "it",
        "this",
        "that",
        "you",
        "your",
        "with",
    )

    def __post_init__(self):
        for name in (
            "steps",
            "population_size",
            "paragraph_interval",
            "crossover_points",
            "max_word_replacements",
            "eval_batch_size",
            "max_new_tokens",
        ):
            require_count(getattr(self, name), name)
        if self.population_size < 2:
            raise ValueError("population_size must be at least two.")
        for name in (
            "elite_fraction",
            "crossover_probability",
            "mutation_probability",
            "word_replacement_probability",
            "momentum",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one.")
        if not 0 < self.elite_fraction < 1:
            raise ValueError("elite_fraction must be strictly between zero and one.")
        if self.loss_threshold is not None and (
            not math.isfinite(self.loss_threshold) or self.loss_threshold < 0
        ):
            raise ValueError("loss_threshold must be finite and nonnegative.")
        object.__setattr__(self, "stop_words", tuple(self.stop_words))


def crossover(left: str, right: str, points: int, rng: random.Random):
    """Complementary sentence blocks; preserve unmatched paragraph tails."""
    parents = [left.split("\n\n"), right.split("\n\n")]
    children = [[], []]
    common = min(map(len, parents))
    for paragraph in range(common):
        sentences = [re.split(r"(?<=[,.!?。！？])\s+", p[paragraph]) for p in parents]
        count = min(map(len, sentences)) - 1
        boundaries = [0, *sorted(rng.sample(range(1, count + 1), min(points, count))), None]
        blocks = [[], []]
        for start, end in zip(boundaries, boundaries[1:]):
            source = rng.randrange(2)
            blocks[0].extend(sentences[source][start:end])
            blocks[1].extend(sentences[1 - source][start:end])
        for child, block in zip(children, blocks):
            child.append(" ".join(block))
    for child, parent in zip(children, parents):
        child.extend(parent[common:])
    return tuple("\n\n".join(child) for child in children)


class AutoDANSearch(Method[OptimizationRequest, OptimizationResult]):
    """Complete fitness/selection/HGA loop; mutation and synonyms are injectable."""

    def __init__(
        self,
        backend: CausalLM,
        generator,
        config: AutoDANConfig | None = None,
        synonyms=wordnet_synonyms,
    ):
        if not callable(generator) or not callable(synonyms):
            raise TypeError("generator and synonyms must be callable.")
        self.backend, self.generator = backend, generator
        self.config, self.synonyms = config or AutoDANConfig(), synonyms

    def mutate(self, prefix: str) -> str:
        return generate_text(
            self.generator,
            "Rewrite the supplied instruction prefix, preserving its meaning and role. "
            "Return only the revised prefix. Do not answer a question or add commentary.",
            {"prefix": prefix},
        )

    def update_momentum(self, population, losses, scores):
        current = defaultdict(list)
        for prefix, loss in zip(population, losses):
            words = set(re.findall(r"\b[A-Za-z]+\b", prefix))
            for word in words:
                if word.lower() not in self.config.stop_words and not word.istitle():
                    current[word.lower()].append(-loss)
        for word, values in current.items():
            average = sum(values) / len(values)
            beta = self.config.momentum
            scores[word] = beta * average + (1 - beta) * scores.get(word, average)

    def replace_words(self, text, scores, rng):
        replaced = 0
        minimum = min(scores.values(), default=0)

        def replace(match):
            nonlocal replaced
            word = match.group()
            if (
                replaced >= self.config.max_word_replacements
                or word.istitle()
                or word.lower() in self.config.stop_words
                or rng.random() >= self.config.word_replacement_probability
            ):
                return word
            choices = sorted(set(self.synonyms(word.lower())) - {word.lower()})
            for choice in choices:
                require_text(choice, "synonym")
            if not choices:
                return word
            weights = [
                max(0, scores.get(choice.lower(), minimum) - minimum) for choice in choices
            ]
            selected = rng.choices(choices, weights=weights if sum(weights) else None, k=1)[0]
            replaced += 1
            return selected

        return re.sub(r"\b[A-Za-z]+\b", replace, text)

    def run(self, query: str, target: str, *, prototype: str):
        return self.build(
            OptimizationRequest(query, target, initial_control=prototype, placement="prefix")
        )

    def build(self, value: OptimizationRequest) -> OptimizationResult:
        if not isinstance(value, OptimizationRequest) or value.placement != "prefix":
            raise ValueError("AutoDANSearch needs a prefix OptimizationRequest.")
        backend, cfg = self.backend, self.config
        rng = random.Random(cfg.seed)
        with backend.session():
            initial_layout = backend.prepare(value, value.initial_control)
            initial = best_loss = float(backend.losses([initial_layout])[0])
            best_layout = initial_layout
            population = [value.initial_control] + [
                self.mutate(value.initial_control) for _ in range(cfg.population_size - 1)
            ]
            trace = [TraceStep(0, initial, initial, value.initial_control, {"target": initial})]
            scores: dict[str, float] = {}
            reason = "steps_exhausted"
            elites = max(1, int(cfg.population_size * cfg.elite_fraction))
            for iteration in range(1, cfg.steps + 1):
                layouts = [backend.prepare(value, prefix) for prefix in population]
                losses = backend.losses(layouts, cfg.eval_batch_size).tolist()
                order = sorted(range(len(population)), key=lambda index: losses[index])
                winner = order[0]
                if losses[winner] < best_loss:
                    best_layout, best_loss = layouts[winner], losses[winner]
                trace.append(
                    TraceStep(
                        iteration,
                        best_loss,
                        best_loss,
                        best_layout.control,
                        {"target": best_loss},
                        len(population),
                    )
                )
                if cfg.loss_threshold is not None and best_loss <= cfg.loss_threshold:
                    reason = "loss_threshold"
                    break
                if iteration == cfg.steps:
                    break
                next_population = [population[i] for i in order[:elites]]
                if (iteration - 1) % cfg.paragraph_interval == 0:
                    fitness = torch.tensor([-x for x in losses]).softmax(0).tolist()
                    while len(next_population) < cfg.population_size:
                        parents = rng.choices(population, weights=fitness, k=2)
                        children = (
                            crossover(*parents, cfg.crossover_points, rng)
                            if rng.random() < cfg.crossover_probability
                            else parents
                        )
                        next_population.extend(children)
                    next_population = next_population[: cfg.population_size]
                else:
                    self.update_momentum(population, losses, scores)
                    next_population.extend(
                        self.replace_words(population[i], scores, rng) for i in order[elites:]
                    )
                population = next_population[:elites] + [
                    self.mutate(prefix) if rng.random() < cfg.mutation_probability else prefix
                    for prefix in next_population[elites:]
                ]
            return backend.finish(
                method="autodan",
                layout=best_layout,
                initial_loss=initial,
                trace=trace,
                stop_reason=reason,
                config=asdict(cfg),
                source=SOURCE,
                max_new_tokens=cfg.max_new_tokens,
            )
