"""Pytest configuration and shared fixtures."""

import pytest
from typing import List
from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
from meta_prompt_evolution.evolution.hub import EvolutionHub, EvolutionConfig


@pytest.fixture
def sample_prompts() -> List[str]:
    """Sample prompt strings for testing."""
    return [
        "You are a helpful assistant. Please {task}",
        "As an AI, I will help you {task}",
        "Let me assist you with {task}",
        "I can help you {task}"
    ]


@pytest.fixture
def sample_prompt_objects(sample_prompts) -> List[Prompt]:
    """Sample Prompt objects for testing."""
    return [Prompt(text=prompt) for prompt in sample_prompts]


@pytest.fixture
def sample_population(sample_prompt_objects) -> PromptPopulation:
    """Sample population for testing."""
    return PromptPopulation(sample_prompt_objects)


@pytest.fixture
def evolution_config() -> EvolutionConfig:
    """Sample evolution configuration for testing."""
    return EvolutionConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.2,
        crossover_rate=0.8
    )


@pytest.fixture
def evolution_hub(evolution_config) -> EvolutionHub:
    """Sample evolution hub for testing."""
    return EvolutionHub(evolution_config)