# meta-prompt-evolution-hub

Genetic algorithm-style evolutionary search for prompt optimization.  
No external APIs. No dependencies beyond Python stdlib.

---

## The Idea

Prompt engineering is largely manual trial-and-error. This project treats it as a search problem: define a population of prompt candidates, score them with heuristic fitness functions, then apply selection → crossover → mutation to evolve better prompts over generations.

The same principles that make genetic algorithms effective for optimization problems apply here:

- **Variation** — multiple mutation strategies diversify the search space
- **Selection pressure** — tournament selection favors better-scoring candidates
- **Elitism** — the best prompts always survive to the next generation
- **Genealogy** — every prompt tracks which parents and mutation produced it

---

## Architecture

```
evolution.py          ← single-file implementation + demo
tests.py              ← 25 unit/integration tests
```

### Core classes

| Class | Role |
|---|---|
| `Prompt` | Dataclass: `text`, `fitness_score`, `generation`, `parent_ids`, `mutation_type` |
| `PromptMutator` | 5 mutation strategies (rephrase, expand, compress, specialize, restructure) |
| `FitnessEvaluator` | Heuristic scoring: length, specificity, clarity, instruction density |
| `EvolutionEngine` | Tournament selection, sentence-level crossover, elitism |
| `PromptLeaderboard` | Tracks top-K prompts across all generations with genealogy |

---

## Mutation Strategies

| Strategy | What it does |
|---|---|
| `rephrase` | Synonym substitution — different words, same intent |
| `expand` | Injects a clarifying instruction sentence |
| `compress` | Strips filler phrases ("please", "as an AI", etc.) |
| `specialize` | Adds domain-specific terminology (coding, writing, analysis) |
| `restructure` | Shuffles sentence order while preserving head/tail |

---

## Fitness Heuristics

Fitness is a weighted composite of four signals:

| Signal | Weight | What it measures |
|---|---|---|
| `length_score` | 0.20 | Proximity to ideal word count (default: 80 words) |
| `specificity_score` | 0.30 | Concrete language: numbers, constraints, explicit markers |
| `clarity_score` | 0.25 | Logical connectors, direct address, structured formatting |
| `instruction_score` | 0.25 | Density of strong instruction verbs |

No LLM calls required — all scoring is regex-based and deterministic.

---

## Quick Start

```bash
# Run the demo: evolve a coding assistant system prompt over 10 generations
python3 evolution.py

# Run tests
python3 tests.py
```

Requires Python 3.9+ (stdlib only).

---

## Demo Output

```
Config: 10 generations | population=16 | seed=42

 Gen     Best     Mean    Worst   StdDev
----------------------------------------
   0   0.3800   0.2010   0.0100   0.1260
   1   0.4800   0.3606   0.2333   0.0666
   ...
  10   0.7748   0.6537   0.5259   0.0647

Δ fitness over 10 generations: +0.3948
Best overall fitness: 0.7748
```

The best evolved prompt for a coding assistant after 10 generations:

> *You are an expert software engineer. Think step by step before giving your final output. When in doubt, ask for clarification before proceeding. support users solve programming problems by writing explicit, correct, and efficient program. Always explain your reasoning. Emphasize unit tests and maintainable.*

---

## Using the Engine in Your Own Code

```python
from evolution import EvolutionEngine, PromptLeaderboard

engine = EvolutionEngine(
    population_size=20,
    elite_count=2,
    mutation_rate=0.75,
    domain="coding",   # "coding" | "writing" | "analysis"
    seed=42,
)

engine.seed_population([
    "You are a helpful assistant.",
    "You are an expert software engineer. Help users write correct code.",
])

leaderboard = PromptLeaderboard(capacity=10)
engine.run(generations=20)
leaderboard.update(engine.population)

print(leaderboard.show(5))
print(engine.best.text)
```

### Plugging in a real fitness function

Replace `EvolutionEngine.evaluator` with your own `FitnessEvaluator` subclass or patch `FitnessEvaluator.evaluate()` to call an LLM, run A/B test metrics, or measure task-specific performance:

```python
class LLMFitnessEvaluator(FitnessEvaluator):
    def evaluate(self, prompt: Prompt) -> Prompt:
        # Call your LLM, run benchmark, compute score
        prompt.fitness_score = your_score_fn(prompt.text)
        return prompt

engine.evaluator = LLMFitnessEvaluator()
```

---

## Extending

- **New mutation strategy** — add a method to `PromptMutator` and register it in `mutate()`
- **Custom domain terms** — extend `PromptMutator._DOMAIN_TERMS`
- **Different selection** — replace `_tournament_select()` in `EvolutionEngine`
- **Multi-objective** — store a vector of fitness scores and implement Pareto ranking

---

## License

MIT
