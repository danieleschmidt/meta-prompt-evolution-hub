"""Core evolution hub for orchestrating prompt evolution."""

from typing import List, Dict, Any, Callable, Optional, Union, Type
import asyncio
import logging
import time
from dataclasses import dataclass

from .population import PromptPopulation, Prompt
from .algorithms.base import EvolutionAlgorithm, AlgorithmConfig
from .algorithms.nsga2 import NSGA2, NSGA2Config
from .algorithms.map_elites import MAPElites, MAPElitesConfig
from .algorithms.cma_es import CMAES, CMAESConfig
from ..evaluation.base import FitnessFunction, TestCase
from ..evaluation.evaluator import (
    DistributedEvaluator, ComprehensiveFitnessFunction, 
    EvaluationConfig, MockLLMProvider
)


@dataclass
class EvolutionConfig:
    """Configuration for evolution parameters."""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    selection_method: str = "tournament"
    algorithm: str = "nsga2"  # "nsga2", "map_elites", "cma_es"
    evaluation_parallel: bool = True
    checkpoint_frequency: int = 10
    diversity_threshold: float = 0.3


class EvolutionHub:
    """Central hub for evolutionary prompt optimization."""
    
    def __init__(
        self, 
        config: Optional[EvolutionConfig] = None,
        fitness_function: Optional[FitnessFunction] = None,
        evaluator: Optional[DistributedEvaluator] = None
    ):
        """Initialize evolution hub with configuration."""
        self.config = config or EvolutionConfig()
        self.fitness_function = fitness_function or ComprehensiveFitnessFunction()
        self.evaluator = evaluator or DistributedEvaluator(
            config=EvaluationConfig(parallel_workers=4),
            fitness_function=self.fitness_function
        )
        
        # Evolution history
        self.evolution_history = []
        self.best_prompts_history = []
        self.diversity_history = []
        
        # Initialize algorithm
        self.algorithm = self._create_algorithm()
        
        # Mutation and crossover operators
        self.mutation_operators = []
        self.crossover_operators = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_algorithm(self) -> EvolutionAlgorithm:
        """Create the specified evolutionary algorithm."""
        if self.config.algorithm == "nsga2":
            algo_config = NSGA2Config(
                population_size=self.config.population_size,
                max_generations=self.config.generations,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
                elitism_rate=self.config.elitism_rate
            )
            return NSGA2(algo_config)
        
        elif self.config.algorithm == "map_elites":
            algo_config = MAPElitesConfig(
                population_size=self.config.population_size,
                max_generations=self.config.generations,
                mutation_rate=self.config.mutation_rate
            )
            return MAPElites(algo_config)
        
        elif self.config.algorithm == "cma_es":
            algo_config = CMAESConfig(
                population_size=self.config.population_size,
                max_generations=self.config.generations,
                dimension=50  # Parameter space dimensionality
            )
            return CMAES(algo_config)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def evolve(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ) -> PromptPopulation:
        """Evolve a population using the configured algorithm."""
        self.logger.info(f"Starting evolution with {self.config.algorithm} algorithm")
        self.logger.info(f"Population size: {len(population)}, Generations: {self.config.generations}")
        
        start_time = time.time()
        current_population = population
        
        # Initialize fitness evaluation
        self._evaluate_population(current_population, test_cases)
        
        for generation in range(self.config.generations):
            self.logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # Track generation start time
            gen_start = time.time()
            
            # Evolve using the selected algorithm
            try:
                next_population = self.algorithm.evolve_generation(
                    current_population, 
                    lambda prompt: self.fitness_function.evaluate(prompt, test_cases)
                )
                
                # Ensure fitness evaluation for new prompts
                self._evaluate_population(next_population, test_cases)
                
                current_population = next_population
                current_population.generation = generation + 1
                
            except Exception as e:
                self.logger.error(f"Evolution error in generation {generation}: {e}")
                break
            
            # Track evolution progress
            best_prompt = self._get_best_prompt(current_population)
            diversity = self._calculate_diversity(current_population)
            
            self.best_prompts_history.append(best_prompt)
            self.diversity_history.append(diversity)
            
            gen_time = time.time() - gen_start
            self.evolution_history.append({
                "generation": generation + 1,
                "best_fitness": best_prompt.fitness_scores.get("fitness", 0.0),
                "diversity": diversity,
                "population_size": len(current_population),
                "execution_time": gen_time
            })
            
            self.logger.info(
                f"Generation {generation + 1} completed: "
                f"Best fitness: {best_prompt.fitness_scores.get('fitness', 0.0):.3f}, "
                f"Diversity: {diversity:.3f}, Time: {gen_time:.2f}s"
            )
            
            # Check termination criteria
            if termination_criteria and termination_criteria(current_population):
                self.logger.info("Termination criteria met, stopping evolution")
                break
            
            # Checkpoint
            if (generation + 1) % self.config.checkpoint_frequency == 0:
                self.checkpoint(current_population.get_top_k(10))
        
        total_time = time.time() - start_time
        self.logger.info(f"Evolution completed in {total_time:.2f} seconds")
        
        return current_population
    
    async def evolve_async(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ):
        """Asynchronous evolution generator yielding generations as they complete."""
        self.logger.info(f"Starting async evolution with {self.config.algorithm}")
        
        current_population = population
        
        # Initialize fitness evaluation
        await self._evaluate_population_async(current_population, test_cases)
        
        for generation in range(self.config.generations):
            gen_start = time.time()
            
            try:
                # Evolve generation
                next_population = self.algorithm.evolve_generation(
                    current_population,
                    lambda prompt: self.fitness_function.evaluate(prompt, test_cases)
                )
                
                # Async fitness evaluation
                await self._evaluate_population_async(next_population, test_cases)
                
                current_population = next_population
                current_population.generation = generation + 1
                
                # Track progress
                best_prompt = self._get_best_prompt(current_population)
                diversity = self._calculate_diversity(current_population)
                
                gen_time = time.time() - gen_start
                generation_info = {
                    "generation": generation + 1,
                    "population": current_population,
                    "best_fitness": best_prompt.fitness_scores.get("fitness", 0.0),
                    "diversity": diversity,
                    "execution_time": gen_time
                }
                
                yield generation_info
                
                # Check termination
                if termination_criteria and termination_criteria(current_population):
                    break
                    
            except Exception as e:
                self.logger.error(f"Async evolution error in generation {generation}: {e}")
                break
    
    def _evaluate_population(self, population: PromptPopulation, test_cases: List[TestCase]):
        """Evaluate fitness for all prompts in population."""
        if self.config.evaluation_parallel:
            results = self.evaluator.evaluate_population(population, test_cases)
            for prompt in population:
                if prompt.id in results:
                    prompt.fitness_scores = results[prompt.id].metrics
        else:
            # Sequential evaluation
            for prompt in population:
                if prompt.fitness_scores is None:
                    prompt.fitness_scores = self.fitness_function.evaluate(prompt, test_cases)
    
    async def _evaluate_population_async(self, population: PromptPopulation, test_cases: List[TestCase]):
        """Asynchronously evaluate fitness for all prompts in population."""
        results = await self.evaluator.evaluate_population_async(population, test_cases)
        for prompt in population:
            if prompt.id in results:
                prompt.fitness_scores = results[prompt.id].metrics
    
    def _get_best_prompt(self, population: PromptPopulation) -> Prompt:
        """Get the best prompt from the population."""
        return max(
            population.prompts,
            key=lambda p: p.fitness_scores.get("fitness", 0.0) if p.fitness_scores else 0.0
        )
    
    def _calculate_diversity(self, population: PromptPopulation) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i, prompt1 in enumerate(population):
            for j, prompt2 in enumerate(population):
                if i < j:
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
        
        # Simple word-based Jaccard distance
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        union = words1.union(words2)
        intersection = words1.intersection(words2)
        
        if not union:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        return 1.0 - jaccard_similarity
    
    def add_mutation_operator(self, operator, probability: float = 0.1):
        """Add a custom mutation operator."""
        self.mutation_operators.append((operator, probability))
    
    def checkpoint(self, prompts: List[Prompt]):
        """Save checkpoint of best prompts."""
        checkpoint_data = {
            "timestamp": time.time(),
            "generation": self.algorithm.generation,
            "prompts": [
                {
                    "id": prompt.id,
                    "text": prompt.text,
                    "fitness_scores": prompt.fitness_scores,
                    "generation": prompt.generation
                }
                for prompt in prompts
            ],
            "config": {
                "algorithm": self.config.algorithm,
                "population_size": self.config.population_size,
                "generation": self.algorithm.generation
            }
        }
        
        self.logger.info(f"Checkpoint saved for generation {self.algorithm.generation}")
        # In a real implementation, this would save to persistent storage
        return checkpoint_data
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        if not self.evolution_history:
            return {}
        
        fitness_scores = [gen["best_fitness"] for gen in self.evolution_history]
        diversity_scores = [gen["diversity"] for gen in self.evolution_history]
        
        return {
            "total_generations": len(self.evolution_history),
            "final_best_fitness": fitness_scores[-1] if fitness_scores else 0.0,
            "fitness_improvement": fitness_scores[-1] - fitness_scores[0] if len(fitness_scores) > 1 else 0.0,
            "average_diversity": sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0,
            "algorithm_used": self.config.algorithm,
            "population_size": self.config.population_size,
            "evolution_history": self.evolution_history,
            "diversity_history": diversity_scores
        }
    
    def create_test_cases(self, test_data: List[Dict[str, Any]]) -> List[TestCase]:
        """Helper method to create test cases from data."""
        test_cases = []
        for item in test_data:
            test_case = TestCase(
                input_data=item.get("input", ""),
                expected_output=item.get("expected", ""),
                metadata=item.get("metadata", {}),
                weight=item.get("weight", 1.0)
            )
            test_cases.append(test_case)
        return test_cases