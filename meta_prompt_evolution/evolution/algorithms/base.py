"""Base classes for evolutionary algorithms."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from ..population import PromptPopulation, Prompt


@dataclass
class AlgorithmConfig:
    """Base configuration for evolutionary algorithms."""
    population_size: int = 100
    max_generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    tournament_size: int = 3
    diversity_threshold: float = 0.3


class EvolutionAlgorithm(ABC):
    """Abstract base class for all evolutionary algorithms."""
    
    def __init__(self, config: AlgorithmConfig):
        """Initialize the algorithm with configuration."""
        self.config = config
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        
    @abstractmethod
    def evolve_generation(
        self, 
        population: PromptPopulation, 
        fitness_fn: Callable[[Prompt], Dict[str, float]]
    ) -> PromptPopulation:
        """Evolve one generation of the population."""
        pass
    
    @abstractmethod
    def selection(self, population: PromptPopulation, k: int) -> List[Prompt]:
        """Select k prompts from population for reproduction."""
        pass
    
    def evolve(
        self, 
        initial_population: PromptPopulation,
        fitness_fn: Callable[[Prompt], Dict[str, float]],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ) -> PromptPopulation:
        """Run complete evolution process."""
        population = initial_population
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate fitness if not already done
            for prompt in population:
                if prompt.fitness_scores is None:
                    prompt.fitness_scores = fitness_fn(prompt)
            
            # Check termination criteria
            if termination_criteria and termination_criteria(population):
                break
                
            # Evolve next generation
            population = self.evolve_generation(population, fitness_fn)
            population.generation = generation + 1
            
            # Track progress
            best_prompt = max(population, key=lambda p: p.fitness_scores.get('fitness', 0))
            self.best_fitness_history.append(best_prompt.fitness_scores.get('fitness', 0))
            
            diversity = self._calculate_diversity(population)
            self.diversity_history.append(diversity)
        
        return population
    
    def _calculate_diversity(self, population: PromptPopulation) -> float:
        """Calculate population diversity based on prompt text similarity."""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i, prompt1 in enumerate(population):
            for j, prompt2 in enumerate(population):
                if i < j:
                    # Simple character-based distance as proxy for diversity
                    distance = self._text_distance(prompt1.text, prompt2.text)
                    total_distance += distance
                    comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _text_distance(self, text1: str, text2: str) -> float:
        """Calculate normalized edit distance between two texts."""
        if not text1 and not text2:
            return 0.0
        if not text1 or not text2:
            return 1.0
        
        # Simple Levenshtein distance implementation
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        max_len = max(m, n)
        return dp[m][n] / max_len if max_len > 0 else 0.0