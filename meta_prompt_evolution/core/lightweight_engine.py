"""
Lightweight Evolution Engine - No Heavy Dependencies
Optimized for environments where full ML stack isn't available.
"""

import random
import math
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid


@dataclass
class LightweightPrompt:
    """Minimal prompt representation for lightweight environments."""
    id: str
    text: str
    fitness: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


class MinimalEvolutionEngine:
    """Ultra-lightweight evolution engine with zero external dependencies."""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.history = []
        
    def evolve_prompts(
        self, 
        seed_prompts: List[str], 
        fitness_evaluator, 
        generations: int = 10
    ) -> List[LightweightPrompt]:
        """
        Evolve prompts using genetic algorithm principles.
        
        Args:
            seed_prompts: Initial prompt texts
            fitness_evaluator: Function that takes prompt text and returns fitness score
            generations: Number of evolution generations
            
        Returns:
            List of evolved prompts sorted by fitness
        """
        print(f"🧬 Starting lightweight evolution: {len(seed_prompts)} seeds → {generations} generations")
        
        # Initialize population
        population = []
        for i, seed in enumerate(seed_prompts):
            prompt = LightweightPrompt(
                id=f"seed_{i}",
                text=seed,
                generation=0
            )
            prompt.fitness = fitness_evaluator(prompt.text)
            population.append(prompt)
        
        # Fill population to target size
        while len(population) < self.population_size:
            parent = random.choice(population)
            mutant = self._mutate_prompt(parent)
            mutant.fitness = fitness_evaluator(mutant.text)
            population.append(mutant)
        
        # Evolution loop
        for gen in range(generations):
            start_time = time.time()
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: Keep top 20%
            population.sort(key=lambda p: p.fitness, reverse=True)
            elite_count = max(1, int(self.population_size * 0.2))
            new_population.extend(population[:elite_count])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                # Crossover
                if random.random() < 0.7:
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate_prompt(child)
                
                child.generation = gen + 1
                child.fitness = fitness_evaluator(child.text)
                new_population.append(child)
            
            population = new_population
            self.generation = gen + 1
            
            # Track progress
            best_fitness = max(p.fitness for p in population)
            avg_fitness = sum(p.fitness for p in population) / len(population)
            diversity = self._calculate_diversity(population)
            
            gen_time = time.time() - start_time
            
            gen_stats = {
                "generation": gen + 1,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "diversity": diversity,
                "time_seconds": gen_time
            }
            self.history.append(gen_stats)
            
            print(f"Gen {gen+1:2d}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, "
                  f"Div={diversity:.3f}, Time={gen_time:.2f}s")
        
        # Return sorted by fitness
        population.sort(key=lambda p: p.fitness, reverse=True)
        return population
    
    def _tournament_select(self, population: List[LightweightPrompt]) -> LightweightPrompt:
        """Tournament selection with tournament size 3."""
        tournament = random.sample(population, min(3, len(population)))
        return max(tournament, key=lambda p: p.fitness)
    
    def _crossover(self, parent1: LightweightPrompt, parent2: LightweightPrompt) -> LightweightPrompt:
        """Simple word-level crossover between two prompts."""
        words1 = parent1.text.split()
        words2 = parent2.text.split()
        
        if not words1 and not words2:
            return parent1
        
        # Random crossover point
        if words1 and words2:
            split1 = random.randint(0, len(words1))
            split2 = random.randint(0, len(words2))
            
            new_words = words1[:split1] + words2[split2:]
        else:
            new_words = words1 if words1 else words2
        
        return LightweightPrompt(
            id=str(uuid.uuid4()),
            text=" ".join(new_words),
            generation=parent1.generation
        )
    
    def _mutate_prompt(self, prompt: LightweightPrompt) -> LightweightPrompt:
        """Apply various mutation operations to a prompt."""
        words = prompt.text.split()
        if not words:
            return prompt
        
        mutation_type = random.choice([
            "word_substitute",
            "word_insert", 
            "word_delete",
            "word_swap",
            "phrase_modify"
        ])
        
        if mutation_type == "word_substitute" and words:
            # Replace a word with a synonym-like variation
            idx = random.randint(0, len(words) - 1)
            words[idx] = self._generate_word_variant(words[idx])
            
        elif mutation_type == "word_insert":
            # Insert a new word
            insert_words = ["please", "help", "assist", "explain", "describe", "analyze"]
            idx = random.randint(0, len(words))
            words.insert(idx, random.choice(insert_words))
            
        elif mutation_type == "word_delete" and len(words) > 1:
            # Delete a word
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
            
        elif mutation_type == "word_swap" and len(words) > 1:
            # Swap two words
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        elif mutation_type == "phrase_modify":
            # Add instruction prefixes/suffixes
            prefixes = ["Please", "Could you", "Can you", "I need you to"]
            suffixes = ["clearly", "in detail", "step by step", "concisely"]
            
            if random.random() < 0.5 and words:
                words.insert(0, random.choice(prefixes))
            if random.random() < 0.5:
                words.append(random.choice(suffixes))
        
        mutated_text = " ".join(words)
        
        return LightweightPrompt(
            id=str(uuid.uuid4()),
            text=mutated_text,
            generation=prompt.generation,
            metadata={"parent_id": prompt.id, "mutation": mutation_type}
        )
    
    def _generate_word_variant(self, word: str) -> str:
        """Generate simple word variants."""
        variants = {
            "help": ["assist", "aid", "support"],
            "explain": ["describe", "clarify", "detail"],
            "analyze": ["examine", "evaluate", "assess"],
            "create": ["generate", "build", "make"],
            "understand": ["grasp", "comprehend", "learn"]
        }
        
        return random.choice(variants.get(word.lower(), [word]))
    
    def _calculate_diversity(self, population: List[LightweightPrompt]) -> float:
        """Calculate simple diversity metric based on text similarity."""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                similarity = self._text_similarity(population[i].text, population[j].text)
                total_distance += (1.0 - similarity)
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def export_results(self, population: List[LightweightPrompt], filename: str):
        """Export evolution results to JSON."""
        results = {
            "evolution_config": {
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "generations": self.generation
            },
            "final_population": [asdict(prompt) for prompt in population],
            "evolution_history": self.history,
            "best_prompts": [asdict(p) for p in population[:10]]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results exported to {filename}")


def simple_fitness_evaluator(prompt_text: str) -> float:
    """
    Simple fitness evaluator for demonstration.
    In practice, this would call an LLM or evaluation service.
    """
    score = 0.0
    
    # Length component (prefer moderate length)
    words = prompt_text.split()
    length_score = 1.0 - abs(len(words) - 10) / 20.0
    score += max(0, length_score) * 0.3
    
    # Quality indicators
    quality_words = ["please", "help", "explain", "describe", "analyze", "step", "detail"]
    quality_score = sum(1 for word in quality_words if word in prompt_text.lower()) / len(quality_words)
    score += quality_score * 0.4
    
    # Structure component
    has_structure = any(marker in prompt_text.lower() 
                       for marker in [":", "?", "step", "first", "then"])
    score += 0.3 if has_structure else 0.0
    
    # Add some randomness for evolutionary diversity
    score += random.uniform(-0.1, 0.1)
    
    return max(0.0, min(1.0, score))


if __name__ == "__main__":
    # Demonstration
    engine = MinimalEvolutionEngine(population_size=15, mutation_rate=0.15)
    
    seed_prompts = [
        "Help me understand this topic",
        "Please explain the concept",
        "I need assistance with analysis",
        "Can you describe the process"
    ]
    
    evolved = engine.evolve_prompts(
        seed_prompts=seed_prompts,
        fitness_evaluator=simple_fitness_evaluator,
        generations=8
    )
    
    print("\n🏆 TOP EVOLVED PROMPTS:")
    for i, prompt in enumerate(evolved[:5], 1):
        print(f"{i}. [{prompt.fitness:.3f}] {prompt.text}")
    
    engine.export_results(evolved, "lightweight_evolution_results.json")