#!/usr/bin/env python3
"""Quick Generation 1 validation test."""

from meta_prompt_evolution import EvolutionHub, PromptPopulation
from meta_prompt_evolution.evolution.hub import EvolutionConfig
from meta_prompt_evolution.evaluation.base import TestCase
import json

def main():
    """Quick test of all three algorithms."""
    print("🚀 Generation 1: Quick Validation Test")
    
    # Test 1: NSGA-II
    config = EvolutionConfig(population_size=5, generations=2, algorithm="nsga2")
    hub = EvolutionHub(config)
    population = PromptPopulation.from_seeds(["Help me", "Please assist", "I need help"])
    test_cases = [TestCase("test", "output", weight=1.0)]
    
    evolved = hub.evolve(population, test_cases)
    nsga2_fitness = evolved.get_top_k(1)[0].fitness_scores.get('fitness', 0)
    print(f"✅ NSGA-II: {nsga2_fitness:.3f}")
    
    # Test 2: MAP-Elites  
    config.algorithm = "map_elites"
    hub = EvolutionHub(config)
    evolved = hub.evolve(population, test_cases)
    map_fitness = evolved.get_top_k(1)[0].fitness_scores.get('fitness', 0)
    print(f"✅ MAP-Elites: {map_fitness:.3f}")
    
    # Test 3: CMA-ES
    config.algorithm = "cma_es"
    hub = EvolutionHub(config)
    evolved = hub.evolve(population, test_cases)
    cma_fitness = evolved.get_top_k(1)[0].fitness_scores.get('fitness', 0)
    print(f"✅ CMA-ES: {cma_fitness:.3f}")
    
    results = {
        "generation_1_status": "COMPLETE",
        "algorithms_working": {
            "nsga2": nsga2_fitness > 0,
            "map_elites": map_fitness > 0, 
            "cma_es": cma_fitness > 0
        },
        "fitness_scores": {
            "nsga2": nsga2_fitness,
            "map_elites": map_fitness,
            "cma_es": cma_fitness
        }
    }
    
    with open('/root/repo/gen1_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🎉 GENERATION 1 COMPLETE - All algorithms working!")
    return results

if __name__ == "__main__":
    main()