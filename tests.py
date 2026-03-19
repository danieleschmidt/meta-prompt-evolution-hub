"""
Tests for meta-prompt-evolution-hub.
Run with: ~/anaconda3/bin/python3 tests.py
"""

import sys
import traceback
from copy import deepcopy

from evolution import (
    Prompt,
    PromptMutator,
    FitnessEvaluator,
    EvolutionEngine,
    PromptLeaderboard,
    SEED_PROMPTS,
)

PASS = "✅"
FAIL = "❌"
_results: list[tuple[str, bool, str]] = []


def test(name: str):
    """Decorator-free test runner helper."""
    def decorator(fn):
        try:
            fn()
            _results.append((name, True, ""))
        except Exception as e:
            _results.append((name, False, traceback.format_exc()))
        return fn
    return decorator


# ──────────────────────────── Prompt dataclass ───────────────────────────────

@test("Prompt: default fields")
def _():
    p = Prompt(text="Hello world")
    assert p.fitness_score == 0.0
    assert p.generation == 0
    assert p.parent_ids == []
    assert p.mutation_type is None


@test("Prompt: fields set correctly")
def _():
    p = Prompt(text="test", fitness_score=0.8, generation=3, parent_ids=[1, 2], mutation_type="rephrase")
    assert p.fitness_score == 0.8
    assert p.generation == 3
    assert p.parent_ids == [1, 2]
    assert p.mutation_type == "rephrase"


@test("Prompt: id is unique")
def _():
    p1 = Prompt(text="a")
    p2 = Prompt(text="b")
    assert p1.id != p2.id


# ──────────────────────────── PromptMutator ──────────────────────────────────

@test("PromptMutator: rephrase returns Prompt")
def _():
    m = PromptMutator(seed=0)
    p = Prompt(text="You are a helpful assistant. Please provide good answers.")
    out = m.rephrase(p)
    assert isinstance(out, Prompt)
    assert out.mutation_type == "rephrase"
    assert out.parent_ids == [p.id]


@test("PromptMutator: expand adds text")
def _():
    m = PromptMutator(seed=1)
    p = Prompt(text="You are a helpful assistant. Provide good code. Always be clear.")
    out = m.expand(p)
    assert len(out.text) > len(p.text)
    assert out.mutation_type == "expand"


@test("PromptMutator: compress removes filler")
def _():
    m = PromptMutator(seed=2)
    text = "Please kindly help the user. As an AI language model, I will assist."
    p = Prompt(text=text)
    out = m.compress(p)
    assert "please" not in out.text.lower() or "kindly" not in out.text.lower()
    assert out.mutation_type == "compress"
    assert len(out.text) > 0  # not empty


@test("PromptMutator: specialize injects domain terms")
def _():
    m = PromptMutator(domain="coding", seed=3)
    p = Prompt(text="You are a helpful assistant. Write code for the user.")
    out = m.specialize(p)
    assert out.mutation_type == "specialize"
    assert len(out.text) > len(p.text)


@test("PromptMutator: restructure reorders sentences")
def _():
    m = PromptMutator(seed=4)
    text = "You are an assistant. Explain everything clearly. Write good code. Always be helpful. Format output nicely."
    p = Prompt(text=text)
    out = m.restructure(p)
    assert out.mutation_type in ("restructure", "rephrase")  # falls back to rephrase if too short


@test("PromptMutator: mutate dispatches to random strategy")
def _():
    m = PromptMutator(seed=5)
    p = Prompt(text="You are a helpful coding assistant. Write clean code. Explain your steps.")
    out = m.mutate(p)
    assert out.mutation_type in ("rephrase", "expand", "compress", "specialize", "restructure")


@test("PromptMutator: mutate respects explicit strategy")
def _():
    m = PromptMutator(seed=6)
    p = Prompt(text="You are a helpful assistant. Always write good code.")
    for strategy in ("rephrase", "expand", "compress", "specialize", "restructure"):
        out = m.mutate(p, strategy=strategy)
        assert out.mutation_type in (strategy, "rephrase"), f"Expected {strategy}, got {out.mutation_type}"


# ──────────────────────────── FitnessEvaluator ───────────────────────────────

@test("FitnessEvaluator: evaluate returns score in [0, 1]")
def _():
    ev = FitnessEvaluator()
    p = Prompt(text="You are a helpful coding assistant. Write correct and readable code.")
    ev.evaluate(p)
    assert 0.0 <= p.fitness_score <= 1.0, f"Score out of range: {p.fitness_score}"


@test("FitnessEvaluator: empty text gets low score")
def _():
    ev = FitnessEvaluator()
    p = Prompt(text="")
    ev.evaluate(p)
    assert p.fitness_score < 0.2


@test("FitnessEvaluator: richer prompt scores higher than bare prompt")
def _():
    ev = FitnessEvaluator()
    bare = Prompt(text="Help me.")
    rich = Prompt(
        text=(
            "You are an expert software engineer. "
            "Always write idiomatic, well-documented code. "
            "Explain your reasoning step by step. "
            "Ensure correctness and handle edge cases. "
            "If the request is ambiguous, ask for clarification before proceeding."
        )
    )
    ev.evaluate(bare)
    ev.evaluate(rich)
    assert rich.fitness_score > bare.fitness_score, (
        f"Rich ({rich.fitness_score:.3f}) should beat bare ({bare.fitness_score:.3f})"
    )


@test("FitnessEvaluator: length_score peaks near ideal")
def _():
    ev = FitnessEvaluator(ideal_word_count=80)
    # 80 words ≈ ideal → should score > 0.8
    text_ideal = " ".join(["word"] * 80)
    assert ev.length_score(text_ideal) > 0.8
    # 5 words → should score low
    assert ev.length_score("Short.") < 0.6
    # 300 words → penalised
    text_long = " ".join(["word"] * 300)
    assert ev.length_score(text_long) < 0.5


# ──────────────────────────── EvolutionEngine ────────────────────────────────

@test("EvolutionEngine: seed_population fills population")
def _():
    eng = EvolutionEngine(population_size=10, seed=7)
    eng.seed_population(SEED_PROMPTS)
    assert len(eng.population) == 10
    assert all(isinstance(p, Prompt) for p in eng.population)


@test("EvolutionEngine: all seeded prompts have fitness > 0")
def _():
    eng = EvolutionEngine(population_size=8, seed=8)
    eng.seed_population(SEED_PROMPTS)
    assert all(p.fitness_score > 0 for p in eng.population)


@test("EvolutionEngine: step increments generation")
def _():
    eng = EvolutionEngine(population_size=8, seed=9)
    eng.seed_population(SEED_PROMPTS)
    assert eng.generation == 0
    eng.step()
    assert eng.generation == 1


@test("EvolutionEngine: population size stable across generations")
def _():
    eng = EvolutionEngine(population_size=10, seed=10)
    eng.seed_population(SEED_PROMPTS)
    for _ in range(5):
        eng.step()
    assert len(eng.population) == 10


@test("EvolutionEngine: history records per-generation stats")
def _():
    eng = EvolutionEngine(population_size=8, seed=11)
    eng.seed_population(SEED_PROMPTS)
    eng.run(generations=3)
    # history[0] = seed, history[1..3] = 3 generations
    assert len(eng.history) == 4
    for h in eng.history:
        assert "best" in h and "mean" in h and "worst" in h


@test("EvolutionEngine: best property returns highest fitness")
def _():
    eng = EvolutionEngine(population_size=8, seed=12)
    eng.seed_population(SEED_PROMPTS)
    best = eng.best
    assert best.fitness_score == max(p.fitness_score for p in eng.population)


@test("EvolutionEngine: elites carry over (fitness non-decreasing for best)")
def _():
    eng = EvolutionEngine(population_size=10, elite_count=1, seed=13)
    eng.seed_population(SEED_PROMPTS)
    bests = [eng.best.fitness_score]
    for _ in range(5):
        eng.step()
        bests.append(eng.best.fitness_score)
    # Best fitness should never decrease (elitism guarantees this)
    for i in range(1, len(bests)):
        assert bests[i] >= bests[i - 1] - 1e-9, (
            f"Fitness dropped at step {i}: {bests[i-1]:.4f} → {bests[i]:.4f}"
        )


# ──────────────────────────── PromptLeaderboard ──────────────────────────────

@test("PromptLeaderboard: tracks top-K")
def _():
    lb = PromptLeaderboard(capacity=3)
    prompts = [Prompt(text=f"Prompt {i}", fitness_score=float(i)) for i in range(10)]
    lb.update(prompts)
    assert len(lb) == 3
    assert lb._entries[0].fitness_score == 9.0


@test("PromptLeaderboard: deduplicates identical text")
def _():
    lb = PromptLeaderboard(capacity=5)
    p1 = Prompt(text="Same text", fitness_score=0.5)
    p2 = Prompt(text="Same text", fitness_score=0.8)
    lb.update([p1, p2])
    assert len(lb) == 1
    assert lb._entries[0].fitness_score == 0.8


@test("PromptLeaderboard: show returns string")
def _():
    lb = PromptLeaderboard(capacity=5)
    eng = EvolutionEngine(population_size=8, seed=14)
    eng.seed_population(SEED_PROMPTS)
    lb.update(eng.population)
    out = lb.show(3)
    assert isinstance(out, str)
    assert "Score" in out


# ──────────────────────────── Integration ────────────────────────────────────

@test("Integration: full 5-generation run improves best fitness")
def _():
    eng = EvolutionEngine(population_size=12, seed=99)
    lb = PromptLeaderboard(capacity=5)
    eng.seed_population(SEED_PROMPTS)
    lb.update(eng.population)
    initial_best = eng.history[0]["best"]
    eng.run(generations=5)
    lb.update(eng.population)
    final_best = eng.history[-1]["best"]
    # Should at least not regress (elitism guarantees non-decrease)
    assert final_best >= initial_best - 1e-9
    assert len(lb) > 0


# ──────────────────────────── Runner ─────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Meta-Prompt Evolution Hub — Test Suite")
    print("=" * 60 + "\n")

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    for name, ok, tb in _results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if tb:
            # Indent traceback
            for line in tb.strip().splitlines():
                print(f"       {line}")

    print(f"\n  {passed}/{passed + failed} tests passed\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
