"""Unit tests for prompt population management."""

import pytest
from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation


class TestPrompt:
    """Test cases for Prompt class."""
    
    def test_prompt_creation(self):
        """Test basic prompt creation."""
        prompt = Prompt(text="Test prompt")
        assert prompt.text == "Test prompt"
        assert prompt.fitness_scores is None
        assert prompt.generation == 0
        assert prompt.id is not None
        assert prompt.id.startswith("prompt_")
    
    def test_prompt_with_fitness(self):
        """Test prompt creation with fitness scores."""
        fitness = {"accuracy": 0.85, "clarity": 0.90}
        prompt = Prompt(text="Test", fitness_scores=fitness)
        assert prompt.fitness_scores == fitness
    
    def test_prompt_with_parents(self):
        """Test prompt creation with parent information."""
        parents = ["parent_1", "parent_2"]
        prompt = Prompt(text="Test", parent_ids=parents, generation=2)
        assert prompt.parent_ids == parents
        assert prompt.generation == 2


class TestPromptPopulation:
    """Test cases for PromptPopulation class."""
    
    def test_population_creation(self, sample_prompt_objects):
        """Test population creation from prompt objects."""
        population = PromptPopulation(sample_prompt_objects)
        assert len(population) == 4
        assert population.size() == 4
        assert population.generation == 0
    
    def test_from_seeds(self, sample_prompts):
        """Test population creation from seed strings."""
        population = PromptPopulation.from_seeds(sample_prompts)
        assert len(population) == 4
        assert all(isinstance(p, Prompt) for p in population)
        assert population.prompts[0].text == sample_prompts[0]
    
    def test_get_top_k_empty_population(self):
        """Test get_top_k with empty population."""
        population = PromptPopulation([])
        top_prompts = population.get_top_k(5)
        assert top_prompts == []
    
    def test_get_top_k_with_fitness(self):
        """Test get_top_k with fitness scores."""
        prompts = [
            Prompt("Low", fitness_scores={"accuracy": 0.3}),
            Prompt("High", fitness_scores={"accuracy": 0.9}),
            Prompt("Medium", fitness_scores={"accuracy": 0.6})
        ]
        population = PromptPopulation(prompts)
        
        top_2 = population.get_top_k(2, "accuracy")
        assert len(top_2) == 2
        assert top_2[0].text == "High"
        assert top_2[1].text == "Medium"
    
    def test_inject_prompts(self, sample_population):
        """Test injecting new prompts into population."""
        initial_size = sample_population.size()
        new_prompts = [Prompt("New prompt 1"), Prompt("New prompt 2")]
        
        sample_population.inject_prompts(new_prompts)
        assert sample_population.size() == initial_size + 2
    
    def test_iteration(self, sample_population):
        """Test population iteration."""
        prompts_from_iter = list(sample_population)
        assert len(prompts_from_iter) == sample_population.size()
        assert all(isinstance(p, Prompt) for p in prompts_from_iter)