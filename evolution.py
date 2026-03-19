"""
meta-prompt-evolution-hub
=========================
Genetic algorithm-style evolutionary search for prompt optimization.
Uses heuristic fitness evaluation — no external API calls required.

Core components:
  Prompt            — dataclass representing a prompt candidate
  PromptMutator     — 5 mutation strategies for generating variations
  FitnessEvaluator  — heuristic scoring for prompt quality
  EvolutionEngine   — tournament selection, crossover, mutation, elitism
  PromptLeaderboard — tracks best prompts across generations with genealogy
"""

from __future__ import annotations

import itertools
import random
import re
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_ID_COUNTER = itertools.count(1)

# ─────────────────────────── Prompt dataclass ────────────────────────────────

@dataclass
class Prompt:
    """A single prompt candidate in the evolutionary population."""

    text: str
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    mutation_type: Optional[str] = None
    id: int = field(default_factory=lambda: next(_ID_COUNTER))

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"Prompt(gen={self.generation}, fitness={self.fitness_score:.3f}, "
            f"mutation={self.mutation_type!r}, text={self.text[:60]!r}...)"
        )


# ──────────────────────────── Mutation strategies ────────────────────────────

class PromptMutator:
    """Generates prompt variations using five mutation strategies."""

    # Word substitution pairs — domain-agnostic rewording
    _SYNONYMS: dict[str, list[str]] = {
        "you": ["the model", "the assistant", "the AI"],
        "help": ["assist", "support", "aid", "guide"],
        "provide": ["give", "supply", "offer", "deliver"],
        "good": ["high-quality", "excellent", "accurate", "thorough"],
        "answer": ["response", "reply", "output", "result"],
        "user": ["the person", "the developer", "the caller", "them"],
        "always": ["consistently", "reliably", "without exception"],
        "clear": ["unambiguous", "explicit", "precise", "direct"],
        "explain": ["describe", "detail", "walk through", "clarify"],
        "make": ["ensure", "produce", "generate", "create"],
        "code": ["implementation", "solution", "snippet", "program"],
        "write": ["produce", "generate", "craft", "compose"],
        "use": ["apply", "employ", "utilize", "leverage"],
        "following": ["listed", "specified", "given", "below"],
        "must": ["should", "are required to", "need to"],
        "format": ["structure", "layout", "organization", "presentation"],
    }

    # Contextual expansion phrases
    _EXPANSIONS = [
        "When in doubt, ask for clarification before proceeding.",
        "Always consider edge cases and potential failure modes.",
        "Prioritize correctness over brevity.",
        "Think step by step before giving your final answer.",
        "If the request is ambiguous, state your assumptions explicitly.",
        "Prefer well-established patterns over clever one-liners.",
        "Include relevant caveats when appropriate.",
        "Validate your reasoning before presenting conclusions.",
    ]

    # Filler/redundant phrases to strip on compression
    _REDUNDANT = [
        r"\bplease\b",
        r"\bkindly\b",
        r"\bfeel free to\b",
        r"\bdo not hesitate to\b",
        r"\bas an AI language model,?\s*",
        r"\bas an AI,?\s*",
        r"\bit is important to note that\s*",
        r"\bit should be noted that\s*",
        r"\bplease note that\s*",
        r"\bI would like you to\b",
        r"\bI want you to\b",
    ]

    # Domain-specific terminology clusters
    _DOMAIN_TERMS: dict[str, list[str]] = {
        "coding": [
            "DRY principles", "SOLID design", "type safety",
            "error handling", "unit tests", "idiomatic", "readable",
            "maintainable", "documented", "version control"
        ],
        "writing": [
            "active voice", "concise prose", "clear structure",
            "audience-appropriate tone", "logical flow"
        ],
        "analysis": [
            "evidence-based", "quantitative reasoning", "systematic",
            "unbiased", "comprehensive", "reproducible"
        ],
    }

    def __init__(self, domain: str = "coding", seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.domain = domain

    # ── Strategy 1: Rephrase ──────────────────────────────────────────────────

    def rephrase(self, prompt: Prompt) -> Prompt:
        """Word-level substitution using synonym map."""
        text = prompt.text
        for word, synonyms in self._SYNONYMS.items():
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, text, re.IGNORECASE) and self._rng.random() < 0.5:
                replacement = self._rng.choice(synonyms)
                text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        return Prompt(
            text=text,
            generation=prompt.generation,
            parent_ids=[prompt.id],
            mutation_type="rephrase",
        )

    # ── Strategy 2: Expand ────────────────────────────────────────────────────

    def expand(self, prompt: Prompt) -> Prompt:
        """Add a clarifying instruction sentence."""
        addition = self._rng.choice(self._EXPANSIONS)
        # Insert after the first sentence or append
        text = prompt.text.rstrip()
        if ". " in text:
            idx = text.index(". ") + 2
            text = text[:idx] + addition + " " + text[idx:]
        else:
            text = text + " " + addition
        return Prompt(
            text=text,
            generation=prompt.generation,
            parent_ids=[prompt.id],
            mutation_type="expand",
        )

    # ── Strategy 3: Compress ─────────────────────────────────────────────────

    def compress(self, prompt: Prompt) -> Prompt:
        """Remove redundant filler phrases."""
        text = prompt.text
        for pattern in self._REDUNDANT:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        # Collapse multiple spaces/newlines
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if not text:
            text = prompt.text  # fallback — don't produce empty prompt
        return Prompt(
            text=text,
            generation=prompt.generation,
            parent_ids=[prompt.id],
            mutation_type="compress",
        )

    # ── Strategy 4: Specialize ────────────────────────────────────────────────

    def specialize(self, prompt: Prompt) -> Prompt:
        """Inject domain-specific terminology."""
        terms = self._DOMAIN_TERMS.get(self.domain, self._DOMAIN_TERMS["coding"])
        chosen = self._rng.sample(terms, min(2, len(terms)))
        qualifier = f" Emphasize {chosen[0]} and {chosen[1]}." if len(chosen) >= 2 else f" Emphasize {chosen[0]}."
        text = prompt.text.rstrip() + qualifier
        return Prompt(
            text=text,
            generation=prompt.generation,
            parent_ids=[prompt.id],
            mutation_type="specialize",
        )

    # ── Strategy 5: Restructure ───────────────────────────────────────────────

    def restructure(self, prompt: Prompt) -> Prompt:
        """Shuffle sentence order (excluding first and last for coherence)."""
        sentences = re.split(r"(?<=[.!?])\s+", prompt.text.strip())
        if len(sentences) > 3:
            head = sentences[:1]
            tail = sentences[-1:]
            middle = sentences[1:-1]
            self._rng.shuffle(middle)
            text = " ".join(head + middle + tail)
        else:
            # Too short to restructure meaningfully — fall back to rephrase
            return self.rephrase(prompt)
        return Prompt(
            text=text,
            generation=prompt.generation,
            parent_ids=[prompt.id],
            mutation_type="restructure",
        )

    def mutate(self, prompt: Prompt, strategy: Optional[str] = None) -> Prompt:
        """Apply a random (or specified) mutation strategy."""
        strategies = {
            "rephrase": self.rephrase,
            "expand": self.expand,
            "compress": self.compress,
            "specialize": self.specialize,
            "restructure": self.restructure,
        }
        if strategy is None:
            strategy = self._rng.choice(list(strategies.keys()))
        return strategies[strategy](prompt)


# ────────────────────────── Fitness evaluator ────────────────────────────────

class FitnessEvaluator:
    """
    Scores prompt quality using heuristic signals.

    Sub-scores (each 0–1), combined with configurable weights:
      length_score       — penalizes very short or very long prompts
      specificity_score  — rewards concrete/specific language
      clarity_score      — rewards clear, direct language patterns
      instruction_score  — rewards explicit instruction verbs
    """

    # Specificity indicators: concrete nouns, numbers, constraints
    _SPECIFICITY_SIGNALS = [
        r"\b\d+\b",                          # numbers
        r"\b(example|instance|e\.g\.|i\.e\.)\b",
        r"\b(specific|concrete|explicit|precise|exact)\b",
        r"\b(must|required|always|never|ensure)\b",
        r"\b(format|structure|output|return|produce)\b",
    ]

    # Clarity indicators: logical connectors, direct address
    _CLARITY_SIGNALS = [
        r"\b(step\s+by\s+step|first|then|finally|therefore)\b",
        r"\b(you (are|will|should|must))\b",
        r"\b(do not|don\'t|avoid|never)\b",
        r":\s*\n",                           # list-like structure
        r"\b(if|when|unless|provided that)\b",
    ]

    # Strong instruction verbs
    _INSTRUCTION_VERBS = [
        r"\b(analyze|evaluate|explain|describe|compare|contrast)\b",
        r"\b(generate|write|create|produce|output|return)\b",
        r"\b(ensure|verify|validate|check|confirm)\b",
        r"\b(follow|adhere|comply|respect|apply)\b",
        r"\b(prioritize|focus|emphasize|highlight)\b",
    ]

    def __init__(
        self,
        ideal_word_count: int = 80,
        length_tolerance: float = 0.5,
        weights: Optional[dict[str, float]] = None,
    ):
        self.ideal_word_count = ideal_word_count
        self.length_tolerance = length_tolerance
        self.weights = weights or {
            "length": 0.20,
            "specificity": 0.30,
            "clarity": 0.25,
            "instruction": 0.25,
        }

    def length_score(self, text: str) -> float:
        """Gaussian-like penalty around ideal word count."""
        words = len(text.split())
        if words == 0:
            return 0.0
        ratio = words / self.ideal_word_count
        # Penalise if outside [0.4 × ideal, 2.5 × ideal]
        if ratio < 0.1:
            return 0.05
        if ratio < 0.4:
            return 0.4 + (ratio - 0.1) / 0.3 * 0.4
        if ratio <= 1.5:
            # Smooth peak at 1.0
            return 1.0 - 0.3 * abs(ratio - 1.0)
        if ratio <= 2.5:
            return 0.7 - (ratio - 1.5) * 0.4
        return 0.3

    def specificity_score(self, text: str) -> float:
        """Fraction of specificity signals present."""
        hits = sum(
            1 for pat in self._SPECIFICITY_SIGNALS
            if re.search(pat, text, re.IGNORECASE)
        )
        return hits / len(self._SPECIFICITY_SIGNALS)

    def clarity_score(self, text: str) -> float:
        """Fraction of clarity signals present."""
        hits = sum(
            1 for pat in self._CLARITY_SIGNALS
            if re.search(pat, text, re.IGNORECASE)
        )
        return hits / len(self._CLARITY_SIGNALS)

    def instruction_score(self, text: str) -> float:
        """Density of instruction verbs (normalised)."""
        hits = sum(
            1 for pat in self._INSTRUCTION_VERBS
            if re.search(pat, text, re.IGNORECASE)
        )
        # Normalise: 3+ hits = full score
        return min(hits / 3, 1.0)

    def evaluate(self, prompt: Prompt) -> Prompt:
        """Compute composite fitness and attach it to the prompt in-place."""
        text = prompt.text
        sub = {
            "length": self.length_score(text),
            "specificity": self.specificity_score(text),
            "clarity": self.clarity_score(text),
            "instruction": self.instruction_score(text),
        }
        prompt.fitness_score = sum(
            self.weights[k] * v for k, v in sub.items()
        )
        return prompt


# ─────────────────────────── Evolution engine ────────────────────────────────

class EvolutionEngine:
    """
    Runs genetic-algorithm-style evolutionary search over a prompt population.

    Selection  — tournament (k=3)
    Crossover  — sentence-level exchange between two parents
    Mutation   — random strategy from PromptMutator
    Elitism    — top `elite_count` survive unchanged each generation
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_count: int = 2,
        mutation_rate: float = 0.7,
        crossover_rate: float = 0.5,
        tournament_k: int = 3,
        domain: str = "coding",
        seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k
        self._rng = random.Random(seed)
        self.mutator = PromptMutator(domain=domain, seed=seed)
        self.evaluator = FitnessEvaluator()
        self.population: List[Prompt] = []
        self.generation: int = 0
        self.history: List[dict] = []  # per-generation stats

    # ── Initialise ────────────────────────────────────────────────────────────

    def seed_population(self, seed_prompts: List[str]) -> None:
        """Bootstrap population from seed strings, padded via mutation if needed."""
        self.population = []
        for text in seed_prompts:
            p = self.evaluator.evaluate(Prompt(text=text, generation=0))
            self.population.append(p)
        # Fill remaining slots with mutations of the seeds
        while len(self.population) < self.population_size:
            parent = self._rng.choice(self.population[:len(seed_prompts)])
            child = self.mutator.mutate(deepcopy(parent))
            child = self.evaluator.evaluate(child)
            self.population.append(child)
        self._record_stats()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _tournament_select(self) -> Prompt:
        contestants = self._rng.sample(self.population, min(self.tournament_k, len(self.population)))
        return max(contestants, key=lambda p: p.fitness_score)

    # ── Crossover ─────────────────────────────────────────────────────────────

    def _crossover(self, p1: Prompt, p2: Prompt) -> Prompt:
        """Sentence-level crossover: first half from p1, second half from p2."""
        s1 = re.split(r"(?<=[.!?])\s+", p1.text.strip())
        s2 = re.split(r"(?<=[.!?])\s+", p2.text.strip())
        split1 = max(1, len(s1) // 2)
        split2 = max(1, len(s2) // 2)
        combined = s1[:split1] + s2[split2:]
        text = " ".join(combined).strip() or p1.text
        return Prompt(
            text=text,
            generation=self.generation,
            parent_ids=[p1.id, p2.id],
            mutation_type="crossover",
        )

    # ── One generation ────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the population by one generation."""
        self.generation += 1

        # Elitism: preserve top N unchanged
        sorted_pop = sorted(self.population, key=lambda p: p.fitness_score, reverse=True)
        elites = [deepcopy(p) for p in sorted_pop[: self.elite_count]]

        new_pop: List[Prompt] = list(elites)

        while len(new_pop) < self.population_size:
            if self._rng.random() < self.crossover_rate:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child = self._crossover(p1, p2)
            else:
                child = deepcopy(self._tournament_select())

            if self._rng.random() < self.mutation_rate:
                child = self.mutator.mutate(child)

            child.generation = self.generation
            child = self.evaluator.evaluate(child)
            new_pop.append(child)

        self.population = new_pop
        self._record_stats()

    def run(self, generations: int = 10) -> None:
        """Run the engine for N generations."""
        for _ in range(generations):
            self.step()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _record_stats(self) -> None:
        scores = [p.fitness_score for p in self.population]
        self.history.append(
            {
                "generation": self.generation,
                "best": max(scores),
                "mean": statistics.mean(scores),
                "worst": min(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            }
        )

    @property
    def best(self) -> Prompt:
        return max(self.population, key=lambda p: p.fitness_score)


# ─────────────────────────── Leaderboard ─────────────────────────────────────

class PromptLeaderboard:
    """
    Tracks the top-K prompts seen across all generations, with genealogy info.
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._entries: List[Prompt] = []

    def update(self, candidates: List[Prompt]) -> None:
        """Add new candidates and keep only the top-K by fitness."""
        self._entries.extend(candidates)
        # Deduplicate by text
        seen: dict[str, Prompt] = {}
        for p in self._entries:
            key = p.text.strip()
            if key not in seen or p.fitness_score > seen[key].fitness_score:
                seen[key] = p
        self._entries = sorted(seen.values(), key=lambda p: p.fitness_score, reverse=True)[
            : self.capacity
        ]

    def show(self, n: int = 5) -> str:
        """Return a human-readable summary of the top-N prompts."""
        lines = [f"{'Rank':<5} {'Score':<8} {'Gen':<5} {'Mutation':<14} Text"]
        lines.append("-" * 80)
        for rank, p in enumerate(self._entries[:n], 1):
            preview = p.text[:55].replace("\n", " ")
            lines.append(
                f"{rank:<5} {p.fitness_score:<8.3f} {p.generation:<5} "
                f"{(p.mutation_type or 'seed'):<14} {preview}..."
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)


# ──────────────────────────────── Demo ───────────────────────────────────────

SEED_PROMPTS = [
    # Minimal seed
    "You are a helpful coding assistant.",
    # Slightly richer seed
    (
        "You are a coding assistant. Help the user write code. "
        "Please provide good answers."
    ),
    # More detailed seed
    (
        "You are an expert software engineer. "
        "Help users solve programming problems by writing clear, correct, and efficient code. "
        "Always explain your reasoning."
    ),
]


def run_demo(generations: int = 10, population_size: int = 16, seed: int = 42) -> None:
    print("=" * 70)
    print("  Meta-Prompt Evolution Hub — Genetic Search Demo")
    print("  Task: evolve a system prompt for a coding assistant")
    print("=" * 70)
    print(f"\nConfig: {generations} generations | population={population_size} | seed={seed}\n")

    engine = EvolutionEngine(
        population_size=population_size,
        elite_count=2,
        mutation_rate=0.75,
        crossover_rate=0.5,
        tournament_k=3,
        domain="coding",
        seed=seed,
    )
    leaderboard = PromptLeaderboard(capacity=10)

    engine.seed_population(SEED_PROMPTS)
    leaderboard.update(engine.population)

    print(f"{'Gen':>4}  {'Best':>7}  {'Mean':>7}  {'Worst':>7}  {'StdDev':>7}")
    print("-" * 40)

    def _row(stats: dict) -> str:
        return (
            f"{stats['generation']:>4}  "
            f"{stats['best']:>7.4f}  "
            f"{stats['mean']:>7.4f}  "
            f"{stats['worst']:>7.4f}  "
            f"{stats['stdev']:>7.4f}"
        )

    print(_row(engine.history[0]))

    for _ in range(generations):
        engine.step()
        leaderboard.update(engine.population)
        print(_row(engine.history[-1]))

    print("\n" + "=" * 70)
    print("  Fitness Progression (best per generation)")
    print("=" * 70)
    bests = [h["best"] for h in engine.history]
    bar_width = 40
    for i, h in enumerate(engine.history):
        score = h["best"]
        max_score = max(bests) if max(bests) > 0 else 1.0
        bar = "█" * int(score / max_score * bar_width)
        label = "SEED" if i == 0 else f"GEN {i:>2}"
        print(f"  {label}  {score:.4f}  |{bar}")

    improvement = bests[-1] - bests[0]
    print(f"\n  Δ fitness over {generations} generations: {improvement:+.4f}")
    print(f"  Best overall fitness: {max(bests):.4f}")

    print("\n" + "=" * 70)
    print("  Top 5 Prompts (Leaderboard)")
    print("=" * 70)
    print(leaderboard.show(5))

    print("\n" + "=" * 70)
    print("  #1 Evolved Prompt (full text)")
    print("=" * 70)
    champion = leaderboard._entries[0]
    print(f"\n{champion.text}\n")
    print(
        f"  fitness={champion.fitness_score:.4f} | "
        f"gen={champion.generation} | "
        f"mutation={champion.mutation_type or 'seed'} | "
        f"parents={champion.parent_ids}"
    )


if __name__ == "__main__":
    run_demo(generations=10, population_size=16, seed=42)
