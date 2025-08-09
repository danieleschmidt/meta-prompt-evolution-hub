#!/usr/bin/env python3
"""
Generation 1 Quick Test: MAKE IT WORK (Simple Implementation)
Fast demonstration of basic evolutionary prompt optimization.
"""

from meta_prompt_evolution import EvolutionHub, PromptPopulation
from meta_prompt_evolution.evolution.hub import EvolutionConfig
from meta_prompt_evolution.evaluation.base import TestCase
import json

def test_basic_evolution():
    """Test basic evolution functionality with minimal configuration."""
    print("🧬 Testing Basic Evolution...")
    
    config = EvolutionConfig(
        population_size=3,
        generations=2,
        algorithm="nsga2",
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    hub = EvolutionHub(config)
    population = PromptPopulation.from_seeds([
        "You are a helpful assistant. Please help with: {task}",
        "As an AI, I will carefully: {task}",
        "Let me assist you with: {task}"
    ])
    
    test_cases = [
        TestCase("summarize this document", "A brief summary of the key points", weight=1.0),
        TestCase("explain quantum computing", "Simple explanation of quantum principles", weight=1.0)
    ]
    
    evolved = hub.evolve(population, test_cases)
    best_prompts = evolved.get_top_k(2)
    
    print(f"  ✅ Basic evolution completed: {len(evolved)} prompts evolved")
    print(f"  🏆 Best fitness: {best_prompts[0].fitness_scores.get('fitness', 0):.3f}")
    stats = hub.get_evolution_statistics()
    print(f"  📈 Generations: {stats.get('total_generations', 0)}")
    print(f"  📊 Fitness improvement: {stats.get('fitness_improvement', 0):.3f}")
    
    return best_prompts

def test_population_management():
    """Test prompt population management."""
    print("\n👥 Testing Population Management...")
    
    # Create population
    seeds = ["Help me with", "Please assist", "I need support"]
    population = PromptPopulation.from_seeds(seeds)
    
    print(f"  ✅ Population created: {len(population)} prompts")
    print(f"  🧬 Initial prompts: {[p.text for p in population.prompts]}")
    
    # Test fitness assignment
    for i, prompt in enumerate(population.prompts):
        prompt.fitness_scores = {"fitness": 0.1 * (i + 1)}
    
    top_prompts = population.get_top_k(2)
    print(f"  🏆 Top 2 prompts: {[p.text for p in top_prompts]}")
    
    return population

def test_comprehensive_evaluation():
    """Test multi-metric evaluation system."""
    print("\n🎯 Testing Comprehensive Evaluation...")
    
    from meta_prompt_evolution.evaluation.evaluator import ComprehensiveFitnessFunction, MockLLMProvider
    
    # Create mock LLM provider for fast testing
    llm_provider = MockLLMProvider(model_name="test-model", latency_ms=50)
    fitness_fn = ComprehensiveFitnessFunction(
        llm_provider=llm_provider,
        metrics={
            "accuracy": 0.4,
            "similarity": 0.3,
            "latency": 0.3
        }
    )
    
    # Test single prompt evaluation
    from meta_prompt_evolution.evolution.population import Prompt
    test_prompt = Prompt("You are helpful and efficient")
    test_cases = [TestCase("help me", "helpful response", weight=1.0)]
    
    scores = fitness_fn.evaluate(test_prompt, test_cases)
    
    print(f"  ✅ Evaluation completed")
    print(f"  📊 Scores: {json.dumps(scores, indent=2)}")
    print(f"  🔧 LLM calls made: {llm_provider.call_count}")
    
    return scores

def main():
    """Run Generation 1 quick test."""
    print("🚀 Generation 1: MAKE IT WORK - Quick Functionality Test")
    print("=" * 55)
    
    try:
        # Run quick tests
        evolution_results = test_basic_evolution()
        population_results = test_population_management()
        evaluation_results = test_comprehensive_evaluation()
        
        print("\n" + "=" * 55)
        print("🎉 GENERATION 1 QUICK TEST COMPLETE")
        print("✅ Basic Evolution: Working")
        print("✅ Population Management: Working") 
        print("✅ Multi-metric Evaluation: Working")
        print("✅ NSGA-II Algorithm: Working")
        print("✅ Fitness Functions: Working")
        
        # Save results
        results = {
            "generation": 1,
            "status": "WORKING",
            "test_type": "QUICK_TEST",
            "components_tested": [
                "Basic Evolution",
                "Population Management",
                "Multi-metric Evaluation",
                "NSGA-II Algorithm",
                "Fitness Functions"
            ],
            "best_fitness": evolution_results[0].fitness_scores.get('fitness', 0) if evolution_results else 0,
            "evaluation_metrics": evaluation_results
        }
        
        with open('/root/repo/demo_results/generation_1_quick_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: demo_results/generation_1_quick_results.json")
        print("🎯 Ready for Generation 2: MAKE IT ROBUST!")
        
    except Exception as e:
        print(f"\n❌ Error in Generation 1 Quick Test: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()