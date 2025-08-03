"""MAP-Elites algorithm for quality-diversity optimization."""

import random
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass

from .base import EvolutionAlgorithm, AlgorithmConfig
from ..population import PromptPopulation, Prompt


@dataclass
class MAPElitesConfig(AlgorithmConfig):
    """Configuration for MAP-Elites algorithm."""
    behavior_dimensions: List[str] = None  # Behavior characteristic names
    grid_resolution: int = 20  # Resolution for each dimension
    initial_random_samples: int = 1000  # Random samples for initialization
    
    def __post_init__(self):
        if self.behavior_dimensions is None:
            self.behavior_dimensions = ["formality", "specificity", "length"]


class MAPElites(EvolutionAlgorithm):
    """MAP-Elites algorithm for maintaining diverse, high-quality prompts."""
    
    def __init__(self, config: MAPElitesConfig):
        """Initialize MAP-Elites with quality-diversity configuration."""
        super().__init__(config)
        self.config: MAPElitesConfig = config
        self.archive = {}  # Grid cell -> best prompt mapping
        self.behavior_ranges = {}  # Dimension -> (min, max) mapping
        
    def evolve_generation(
        self, 
        population: PromptPopulation, 
        fitness_fn: Callable[[Prompt], Dict[str, float]]
    ) -> PromptPopulation:
        """Evolve one generation using MAP-Elites procedure."""
        # Initialize archive if empty
        if not self.archive:
            self._initialize_archive(population, fitness_fn)
        
        # Generate new solutions
        for _ in range(self.config.population_size):
            # Select random prompt from archive
            if self.archive:
                parent = random.choice(list(self.archive.values()))
                
                # Mutate parent
                offspring = self._mutate(parent)
                offspring.fitness_scores = fitness_fn(offspring)
                
                # Calculate behavior characteristics
                behavior_vector = self._calculate_behavior_vector(offspring)
                grid_cell = self._behavior_to_grid_cell(behavior_vector)
                
                # Add to archive if cell is empty or offspring is better
                if (grid_cell not in self.archive or 
                    offspring.fitness_scores.get('fitness', 0) > 
                    self.archive[grid_cell].fitness_scores.get('fitness', 0)):
                    self.archive[grid_cell] = offspring
        
        # Return current archive as population
        archive_prompts = list(self.archive.values())
        return PromptPopulation(archive_prompts)
    
    def _initialize_archive(
        self, 
        population: PromptPopulation, 
        fitness_fn: Callable[[Prompt], Dict[str, float]]
    ):
        """Initialize the archive with random solutions and population."""
        all_prompts = list(population.prompts)
        
        # Generate additional random prompts for initialization
        random_prompts = self._generate_random_prompts(
            self.config.initial_random_samples, 
            population.prompts[0] if population.prompts else None
        )
        all_prompts.extend(random_prompts)
        
        # Calculate behavior characteristics for all prompts
        behavior_vectors = []
        for prompt in all_prompts:
            if prompt.fitness_scores is None:
                prompt.fitness_scores = fitness_fn(prompt)
            behavior_vector = self._calculate_behavior_vector(prompt)
            behavior_vectors.append(behavior_vector)
        
        # Determine behavior ranges
        if behavior_vectors:
            behavior_array = np.array(behavior_vectors)
            for i, dimension in enumerate(self.config.behavior_dimensions):
                self.behavior_ranges[dimension] = (
                    behavior_array[:, i].min(),
                    behavior_array[:, i].max()
                )
        
        # Populate archive
        for prompt, behavior_vector in zip(all_prompts, behavior_vectors):
            grid_cell = self._behavior_to_grid_cell(behavior_vector)
            if (grid_cell not in self.archive or 
                prompt.fitness_scores.get('fitness', 0) > 
                self.archive[grid_cell].fitness_scores.get('fitness', 0)):
                self.archive[grid_cell] = prompt
    
    def _generate_random_prompts(self, count: int, template: Optional[Prompt]) -> List[Prompt]:
        """Generate random prompt variations."""
        if not template:
            base_prompts = [
                "You are a helpful assistant. Please help with: {task}",
                "As an AI, I will carefully: {task}",
                "Let me assist you with: {task}",
                "I'll help you solve: {task}",
                "Please allow me to: {task}"
            ]
            return [Prompt(text=text) for text in base_prompts[:count]]
        
        random_prompts = []
        for _ in range(count):
            # Generate variations of the template
            words = template.text.split()
            if words:
                # Random word substitution
                new_words = words.copy()
                for i in range(random.randint(1, min(3, len(words)))):
                    idx = random.randint(0, len(new_words) - 1)
                    new_words[idx] = self._get_random_word()
                
                random_prompts.append(Prompt(text=" ".join(new_words)))
        
        return random_prompts
    
    def _calculate_behavior_vector(self, prompt: Prompt) -> List[float]:
        """Calculate behavior characteristics for a prompt."""
        behavior_vector = []
        
        for dimension in self.config.behavior_dimensions:
            if dimension == "formality":
                value = self._calculate_formality(prompt.text)
            elif dimension == "specificity":
                value = self._calculate_specificity(prompt.text)
            elif dimension == "length":
                value = len(prompt.text.split())
            elif dimension == "complexity":
                value = self._calculate_complexity(prompt.text)
            elif dimension == "politeness":
                value = self._calculate_politeness(prompt.text)
            else:
                value = random.random()  # Fallback for unknown dimensions
            
            behavior_vector.append(value)
        
        return behavior_vector
    
    def _calculate_formality(self, text: str) -> float:
        """Calculate formality score (0-1, higher = more formal)."""
        formal_words = {'please', 'kindly', 'respectfully', 'formally', 'officially'}
        informal_words = {'hey', 'yeah', 'cool', 'awesome', 'stuff'}
        
        words = set(text.lower().split())
        formal_count = len(words.intersection(formal_words))
        informal_count = len(words.intersection(informal_words))
        
        if formal_count + informal_count == 0:
            return 0.5  # Neutral
        
        return formal_count / (formal_count + informal_count)
    
    def _calculate_specificity(self, text: str) -> float:
        """Calculate specificity score (0-1, higher = more specific)."""
        specific_words = {'exactly', 'precisely', 'specifically', 'detailed', 'step-by-step'}
        general_words = {'generally', 'roughly', 'approximately', 'broadly', 'overall'}
        
        words = set(text.lower().split())
        specific_count = len(words.intersection(specific_words))
        general_count = len(words.intersection(general_words))
        
        # Also consider presence of numbers, proper nouns (uppercase)
        word_list = text.split()
        numbers = sum(1 for word in word_list if any(c.isdigit() for c in word))
        proper_nouns = sum(1 for word in word_list if word[0].isupper() and word.lower() not in {'the', 'a', 'an'})
        
        specificity_score = (specific_count * 2 + numbers + proper_nouns) / max(len(word_list), 1)
        return min(specificity_score, 1.0)
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate complexity score based on sentence structure."""
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Count complex punctuation
        complex_punct = text.count(',') + text.count(';') + text.count(':')
        
        complexity = (avg_sentence_length / 20.0) + (complex_punct / 10.0)
        return min(complexity, 1.0)
    
    def _calculate_politeness(self, text: str) -> float:
        """Calculate politeness score."""
        polite_words = {'please', 'thank', 'kindly', 'appreciate', 'grateful', 'sorry'}
        words = set(text.lower().split())
        polite_count = len(words.intersection(polite_words))
        
        # Check for question marks (polite questioning)
        questions = text.count('?')
        
        politeness = (polite_count * 2 + questions) / max(len(text.split()), 1)
        return min(politeness, 1.0)
    
    def _behavior_to_grid_cell(self, behavior_vector: List[float]) -> Tuple[int, ...]:
        """Convert behavior vector to grid cell coordinates."""
        grid_cell = []
        
        for i, (value, dimension) in enumerate(zip(behavior_vector, self.config.behavior_dimensions)):
            if dimension in self.behavior_ranges:
                min_val, max_val = self.behavior_ranges[dimension]
                if max_val == min_val:
                    cell_coord = 0
                else:
                    normalized = (value - min_val) / (max_val - min_val)
                    cell_coord = min(int(normalized * self.config.grid_resolution), 
                                   self.config.grid_resolution - 1)
            else:
                # Fallback for unknown dimensions
                cell_coord = int(value * self.config.grid_resolution) % self.config.grid_resolution
            
            grid_cell.append(cell_coord)
        
        return tuple(grid_cell)
    
    def _mutate(self, prompt: Prompt) -> Prompt:
        """Mutate a prompt to create offspring."""
        words = prompt.text.split()
        if not words:
            return prompt
        
        mutation_type = random.choice(['substitute', 'insert', 'delete', 'behavioral'])
        
        if mutation_type == 'substitute' and words:
            idx = random.randint(0, len(words) - 1)
            words[idx] = self._get_random_word()
        
        elif mutation_type == 'insert':
            idx = random.randint(0, len(words))
            words.insert(idx, self._get_random_word())
        
        elif mutation_type == 'delete' and len(words) > 1:
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        
        elif mutation_type == 'behavioral':
            # Behavioral mutation: modify to increase diversity
            behavior_words = {
                'formality': ['please', 'kindly', 'hey', 'yo'],
                'specificity': ['exactly', 'specifically', 'roughly', 'generally'],
                'politeness': ['please', 'thank you', 'sorry', 'appreciate']
            }
            
            target_dimension = random.choice(self.config.behavior_dimensions)
            if target_dimension in behavior_words:
                words.append(random.choice(behavior_words[target_dimension]))
        
        mutated_prompt = Prompt(
            text=" ".join(words),
            generation=self.generation + 1,
            parent_ids=[prompt.id]
        )
        
        return mutated_prompt
    
    def _get_random_word(self) -> str:
        """Get a random word for mutations."""
        word_categories = {
            'modifiers': ['carefully', 'quickly', 'thoroughly', 'precisely', 'clearly'],
            'actions': ['analyze', 'create', 'explain', 'solve', 'help'],
            'connectors': ['and', 'but', 'however', 'therefore', 'additionally'],
            'courtesy': ['please', 'kindly', 'thank you', 'appreciate']
        }
        
        category = random.choice(list(word_categories.keys()))
        return random.choice(word_categories[category])
    
    def selection(self, population: PromptPopulation, k: int) -> List[Prompt]:
        """Select k prompts from archive (random selection for diversity)."""
        if not self.archive:
            return random.sample(population.prompts, min(k, len(population)))
        
        archive_prompts = list(self.archive.values())
        return random.sample(archive_prompts, min(k, len(archive_prompts)))
    
    def get_archive_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current archive."""
        if not self.archive:
            return {"cells_filled": 0, "total_cells": 0, "coverage": 0.0}
        
        total_possible_cells = self.config.grid_resolution ** len(self.config.behavior_dimensions)
        cells_filled = len(self.archive)
        coverage = cells_filled / total_possible_cells
        
        return {
            "cells_filled": cells_filled,
            "total_cells": total_possible_cells,
            "coverage": coverage,
            "behavior_dimensions": self.config.behavior_dimensions,
            "grid_resolution": self.config.grid_resolution
        }