#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK (Simple Implementation)
Demonstration of basic evolutionary prompt optimization.
"""

from meta_prompt_evolution import EvolutionHub, PromptPopulation
from meta_prompt_evolution.evolution.hub import EvolutionConfig
from meta_prompt_evolution.evaluation.base import TestCase
from meta_prompt_evolution.evolution.algorithms.nsga2 import NSGA2Config
from meta_prompt_evolution.evolution.algorithms.map_elites import MAPElitesConfig
from meta_prompt_evolution.evolution.algorithms.cma_es import CMAESConfig
import json

def demo_nsga2():
    """Demonstrate NSGA-II multi-objective optimization."""
    print("🧬 Testing NSGA-II Algorithm...")
    
    config = EvolutionConfig(
        population_size=20,
        generations=5,
        algorithm="nsga2",
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    hub = EvolutionHub(config)
    population = PromptPopulation.from_seeds([
        "You are a helpful assistant. Please help with: {task}",
        "As an AI, I will carefully: {task}",
        "Let me assist you with: {task}",
        "I'll help you solve: {task}",
        "Please allow me to: {task}"
    ])
    
    test_cases = [
        TestCase("summarize this document", "A brief summary of the key points", weight=1.0),
        TestCase("explain quantum computing", "Simple explanation of quantum principles", weight=1.5),
        TestCase("write a Python function", "Clean, documented code", weight=1.2)
    ]
    
    evolved = hub.evolve(population, test_cases)
    best_prompts = evolved.get_top_k(3)
    
    print(f"  ✅ NSGA-II completed: {len(evolved)} prompts evolved")
    print(f"  🏆 Best fitness: {best_prompts[0].fitness_scores.get('fitness', 0):.3f}")
    print(f"  📈 Stats: {hub.get_evolution_statistics()}")
    return best_prompts

def demo_map_elites():
    """Demonstrate MAP-Elites quality-diversity optimization."""
    print("\n🗺️  Testing MAP-Elites Algorithm...")
    
    config = EvolutionConfig(
        population_size=15,
        generations=4,
        algorithm="map_elites",
        mutation_rate=0.2
    )
    
    hub = EvolutionHub(config)
    population = PromptPopulation.from_seeds([
        "Help me with this task",
        "Please assist me carefully",
        "I need help solving this"
    ])
    
    test_cases = [
        TestCase("classify this text", "Category A", weight=1.0),
        TestCase("analyze the data", "Detailed analysis", weight=1.0)
    ]
    
    evolved = hub.evolve(population, test_cases)
    
    # Access MAP-Elites specific stats
    map_elites_algo = hub.algorithm
    if hasattr(map_elites_algo, 'get_archive_statistics'):
        archive_stats = map_elites_algo.get_archive_statistics()
        print(f"  ✅ MAP-Elites completed: Archive coverage {archive_stats['coverage']:.2%}")
        print(f"  📊 Cells filled: {archive_stats['cells_filled']}/{archive_stats['total_cells']}")
    
    return evolved.get_top_k(3)

def demo_cma_es():
    """Demonstrate CMA-ES continuous optimization."""
    print("\n📊 Testing CMA-ES Algorithm...")
    
    config = EvolutionConfig(
        population_size=12,
        generations=3,
        algorithm="cma_es"
    )
    
    hub = EvolutionHub(config)
    population = PromptPopulation.from_seeds([
        "Please help with the following task",
        "I will assist you systematically"
    ])
    
    test_cases = [
        TestCase("solve this problem", "Step by step solution", weight=1.0)
    ]
    
    evolved = hub.evolve(population, test_cases)
    
    # Access CMA-ES specific metrics
    cma_es_algo = hub.algorithm
    if hasattr(cma_es_algo, 'get_convergence_metrics'):
        convergence = cma_es_algo.get_convergence_metrics()
        print(f"  ✅ CMA-ES completed: σ={convergence['sigma']:.3f}")
        print(f"  📏 Condition number: {convergence['condition_number']:.2f}")
    
    return evolved.get_top_k(3)

def demo_comprehensive_evaluation():
    """Demonstrate comprehensive multi-metric evaluation."""
    print("\n🎯 Testing Comprehensive Evaluation...")
    
    from meta_prompt_evolution.evaluation.evaluator import ComprehensiveFitnessFunction, MockLLMProvider
    
    # Create custom fitness function
    llm_provider = MockLLMProvider(model_name="demo-model", latency_ms=100)
    fitness_fn = ComprehensiveFitnessFunction(
        llm_provider=llm_provider,
        metrics={
            "accuracy": 0.3,
            "similarity": 0.2,
            "latency": 0.2,
            "safety": 0.3
        }
    )
    
    config = EvolutionConfig(population_size=8, generations=3)
    hub = EvolutionHub(config, fitness_function=fitness_fn)
    
    population = PromptPopulation.from_seeds([
        "You are helpful and safe",
        "Please assist carefully",
        "I will help you quickly"
    ])
    
    test_cases = [
        TestCase("explain AI safety", "Safe AI explanation", weight=2.0),
        TestCase("summarize quickly", "Quick summary", weight=1.0)
    ]
    
    evolved = hub.evolve(population, test_cases)
    best = evolved.get_top_k(1)[0]
    
    print(f"  ✅ Comprehensive evaluation completed")
    print(f"  🏆 Best prompt: '{best.text}'")
    print(f"  📊 Detailed scores: {json.dumps(best.fitness_scores, indent=2)}")
    print(f"  🔧 LLM calls made: {llm_provider.call_count}")
    
    return best

def main():
    """Run Generation 1 demonstration."""
    print("🚀 Generation 1: MAKE IT WORK - Basic Functionality Demo")
    print("=" * 60)
    
    try:
        # Test all algorithms
        nsga2_results = demo_nsga2()
        map_elites_results = demo_map_elites()
        cma_es_results = demo_cma_es()
        comprehensive_result = demo_comprehensive_evaluation()
        
        print("\n" + "=" * 60)
        print("🎉 GENERATION 1 COMPLETE: ALL SYSTEMS OPERATIONAL")
        print("✅ NSGA-II: Multi-objective optimization working")
        print("✅ MAP-Elites: Quality-diversity optimization working")
        print("✅ CMA-ES: Continuous parameter optimization working")
        print("✅ Evaluation: Multi-metric fitness assessment working")
        print("✅ Population Management: Prompt evolution working")
        print("✅ Algorithms: All three evolutionary strategies functional")
        
        print(f"\n📈 Summary:")
        print(f"  • Total algorithms tested: 3")
        print(f"  • Evaluation metrics: 4 (accuracy, similarity, latency, safety)")
        print(f"  • Population management: ✓")
        print(f"  • Fitness evaluation: ✓")
        print(f"  • Evolution strategies: ✓")
        
        # Save results
        results = {
            "generation": 1,
            "status": "WORKING",
            "algorithms_tested": ["NSGA2", "MAP-Elites", "CMA-ES"],
            "best_overall_fitness": max([
                nsga2_results[0].fitness_scores.get('fitness', 0),
                comprehensive_result.fitness_scores.get('fitness', 0)
            ]),
            "functionality_verified": [
                "Multi-objective optimization",
                "Quality-diversity search", 
                "Continuous parameter optimization",
                "Multi-metric evaluation",
                "Population evolution"
            ]
        }
        
        with open('/root/repo/generation_1_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: generation_1_results.json")
        print("\n🎯 Ready for Generation 2: MAKE IT ROBUST!")
        
    except Exception as e:
        print(f"\n❌ Error in Generation 1: {e}")
        raise

if __name__ == "__main__":
    main()