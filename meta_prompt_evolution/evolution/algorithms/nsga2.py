"""NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation."""

import random
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import math

from .base import EvolutionAlgorithm, AlgorithmConfig
from ..population import PromptPopulation, Prompt


@dataclass
class NSGA2Config(AlgorithmConfig):
    """Configuration specific to NSGA-II algorithm."""
    objectives: List[str] = None  # List of objective names to optimize
    maximize_objectives: List[bool] = None  # True to maximize, False to minimize
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["accuracy", "efficiency", "safety"]
        if self.maximize_objectives is None:
            self.maximize_objectives = [True, True, True]  # Default: maximize all


class NSGA2(EvolutionAlgorithm):
    """NSGA-II algorithm for multi-objective prompt optimization."""
    
    def __init__(self, config: NSGA2Config):
        """Initialize NSGA-II with multi-objective configuration."""
        super().__init__(config)
        self.config: NSGA2Config = config
        
    def evolve_generation(
        self, 
        population: PromptPopulation, 
        fitness_fn: Callable[[Prompt], Dict[str, float]]
    ) -> PromptPopulation:
        """Evolve one generation using NSGA-II procedure."""
        # Create offspring through selection, crossover, and mutation
        offspring = self._create_offspring(population, fitness_fn)
        
        # Combine parent and offspring populations
        combined_population = PromptPopulation(population.prompts + offspring.prompts)
        
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined_population)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Select next generation
        next_generation = []
        current_size = 0
        
        for front in fronts:
            if current_size + len(front) <= self.config.population_size:
                next_generation.extend(front)
                current_size += len(front)
            else:
                # Sort by crowding distance and select remaining
                remaining_slots = self.config.population_size - current_size
                front.sort(key=lambda p: getattr(p, 'crowding_distance', 0), reverse=True)
                next_generation.extend(front[:remaining_slots])
                break
        
        return PromptPopulation(next_generation)
    
    def _create_offspring(
        self, 
        population: PromptPopulation, 
        fitness_fn: Callable[[Prompt], Dict[str, float]]
    ) -> PromptPopulation:
        """Create offspring through selection, crossover, and mutation."""
        offspring = []
        
        while len(offspring) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, 2)
            parent2 = self._tournament_selection(population, 2)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        # Mutation
        for prompt in offspring:
            if random.random() < self.config.mutation_rate:
                prompt = self._mutate(prompt)
        
        # Trim to exact size and evaluate fitness
        offspring = offspring[:self.config.population_size]
        for prompt in offspring:
            if prompt.fitness_scores is None:
                prompt.fitness_scores = fitness_fn(prompt)
        
        return PromptPopulation(offspring)
    
    def _fast_non_dominated_sort(self, population: PromptPopulation) -> List[List[Prompt]]:
        """Perform fast non-dominated sorting."""
        fronts = [[]]
        
        for prompt in population:
            prompt.domination_count = 0
            prompt.dominated_solutions = []
            
            for other in population:
                if prompt is not other:
                    if self._dominates(prompt, other):
                        prompt.dominated_solutions.append(other)
                    elif self._dominates(other, prompt):
                        prompt.domination_count += 1
            
            if prompt.domination_count == 0:
                prompt.rank = 0
                fronts[0].append(prompt)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for prompt in fronts[i]:
                for dominated in prompt.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = i + 1
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, prompt1: Prompt, prompt2: Prompt) -> bool:
        """Check if prompt1 dominates prompt2."""
        if not prompt1.fitness_scores or not prompt2.fitness_scores:
            return False
        
        better_in_at_least_one = False
        
        for i, objective in enumerate(self.config.objectives):
            value1 = prompt1.fitness_scores.get(objective, 0)
            value2 = prompt2.fitness_scores.get(objective, 0)
            
            if self.config.maximize_objectives[i]:
                if value1 < value2:
                    return False  # prompt1 is worse in this objective
                elif value1 > value2:
                    better_in_at_least_one = True
            else:  # minimize objective
                if value1 > value2:
                    return False  # prompt1 is worse in this objective
                elif value1 < value2:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _calculate_crowding_distance(self, front: List[Prompt]):
        """Calculate crowding distance for prompts in a front."""
        if len(front) <= 2:
            for prompt in front:
                prompt.crowding_distance = float('inf')
            return
        
        for prompt in front:
            prompt.crowding_distance = 0
        
        for objective in self.config.objectives:
            # Sort by objective value
            front.sort(key=lambda p: p.fitness_scores.get(objective, 0))
            
            # Set boundary points to infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_min = front[0].fitness_scores.get(objective, 0)
            obj_max = front[-1].fitness_scores.get(objective, 0)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distance for middle points
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float('inf'):
                    distance = (front[i+1].fitness_scores.get(objective, 0) - 
                              front[i-1].fitness_scores.get(objective, 0)) / obj_range
                    front[i].crowding_distance += distance
    
    def _tournament_selection(self, population: PromptPopulation, tournament_size: int) -> Prompt:
        """Select prompt using tournament selection with NSGA-II criteria."""
        tournament = random.sample(population.prompts, min(tournament_size, len(population)))
        
        # Sort by rank first, then by crowding distance
        tournament.sort(key=lambda p: (
            getattr(p, 'rank', float('inf')),
            -getattr(p, 'crowding_distance', 0)
        ))
        
        return tournament[0]
    
    def _crossover(self, parent1: Prompt, parent2: Prompt) -> Tuple[Prompt, Prompt]:
        """Perform crossover between two prompts."""
        # Simple word-level crossover
        words1 = parent1.text.split()
        words2 = parent2.text.split()
        
        if len(words1) == 0 or len(words2) == 0:
            return parent1, parent2
        
        # Single-point crossover
        crossover_point1 = random.randint(0, len(words1))
        crossover_point2 = random.randint(0, len(words2))
        
        child1_text = " ".join(words1[:crossover_point1] + words2[crossover_point2:])
        child2_text = " ".join(words2[:crossover_point2] + words1[crossover_point1:])
        
        child1 = Prompt(
            text=child1_text,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        child2 = Prompt(
            text=child2_text,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child1, child2
    
    def _mutate(self, prompt: Prompt) -> Prompt:
        """Perform mutation on a prompt."""
        words = prompt.text.split()
        if not words:
            return prompt
        
        mutation_type = random.choice(['substitute', 'insert', 'delete', 'reorder'])
        
        if mutation_type == 'substitute' and words:
            # Replace a random word
            idx = random.randint(0, len(words) - 1)
            words[idx] = self._get_similar_word(words[idx])
        
        elif mutation_type == 'insert':
            # Insert a random word
            idx = random.randint(0, len(words))
            new_word = random.choice(['please', 'carefully', 'specifically', 'exactly', 'clearly'])
            words.insert(idx, new_word)
        
        elif mutation_type == 'delete' and len(words) > 1:
            # Delete a random word
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        
        elif mutation_type == 'reorder' and len(words) > 1:
            # Swap two random words
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        mutated_prompt = Prompt(
            text=" ".join(words),
            generation=self.generation + 1,
            parent_ids=[prompt.id]
        )
        
        return mutated_prompt
    
    def _get_similar_word(self, word: str) -> str:
        """Get a semantically similar word (simplified implementation)."""
        synonyms = {
            'help': ['assist', 'aid', 'support'],
            'explain': ['describe', 'clarify', 'elaborate'],
            'create': ['generate', 'produce', 'make'],
            'analyze': ['examine', 'evaluate', 'assess'],
            'write': ['compose', 'draft', 'author'],
            'solve': ['resolve', 'address', 'fix']
        }
        
        return random.choice(synonyms.get(word.lower(), [word]))
    
    def selection(self, population: PromptPopulation, k: int) -> List[Prompt]:
        """Select k prompts using NSGA-II selection criteria."""
        selected = []
        for _ in range(k):
            selected.append(self._tournament_selection(population, self.config.tournament_size))
        return selected