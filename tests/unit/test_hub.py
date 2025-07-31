"""Unit tests for evolution hub."""

import pytest
from meta_prompt_evolution.evolution.hub import EvolutionHub, EvolutionConfig


class TestEvolutionConfig:
    """Test cases for EvolutionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()
        assert config.population_size == 1000
        assert config.generations == 50
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.7
        assert config.elitism_rate == 0.1
        assert config.selection_method == "tournament"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvolutionConfig(
            population_size=500,
            generations=25,
            mutation_rate=0.2
        )
        assert config.population_size == 500
        assert config.generations == 25
        assert config.mutation_rate == 0.2
        # Defaults should still apply
        assert config.crossover_rate == 0.7


class TestEvolutionHub:
    """Test cases for EvolutionHub."""
    
    def test_hub_creation_default_config(self):
        """Test hub creation with default config."""
        hub = EvolutionHub()
        assert isinstance(hub.config, EvolutionConfig)
        assert hub.config.population_size == 1000
        assert hub.mutation_operators == []
        assert hub.crossover_operators == []
    
    def test_hub_creation_custom_config(self, evolution_config):
        """Test hub creation with custom config."""
        hub = EvolutionHub(evolution_config)
        assert hub.config == evolution_config
        assert hub.config.population_size == 10
    
    def test_add_mutation_operator(self, evolution_hub):
        """Test adding mutation operators."""
        def dummy_mutator():
            pass
        
        evolution_hub.add_mutation_operator(dummy_mutator, 0.3)
        assert len(evolution_hub.mutation_operators) == 1
        assert evolution_hub.mutation_operators[0] == (dummy_mutator, 0.3)
    
    def test_evolve_not_implemented(self, evolution_hub, sample_population):
        """Test that evolve method raises NotImplementedError."""
        def dummy_fitness(prompt, test_cases):
            return 0.5
        
        with pytest.raises(NotImplementedError):
            evolution_hub.evolve(
                population=sample_population,
                fitness_fn=dummy_fitness,
                test_cases=[]
            )
    
    def test_checkpoint(self, evolution_hub):
        """Test checkpoint method (should not raise errors)."""
        evolution_hub.checkpoint([])  # Should not raise
    
    @pytest.mark.asyncio
    async def test_evolve_async(self, evolution_hub, sample_population):
        """Test async evolution generator."""
        generation_count = 0
        async for generation in evolution_hub.evolve_async(sample_population):
            generation_count += 1
            assert generation == sample_population
            if generation_count >= 3:  # Test first few iterations
                break
        
        assert generation_count == 3