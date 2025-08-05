#!/usr/bin/env python3
"""Quick Generation 3 validation test."""

import time
import json
from scalable_evolution_hub import create_scalable_hub
from meta_prompt_evolution.evolution.population import PromptPopulation
from meta_prompt_evolution.evaluation.base import TestCase
from caching_system import evaluation_cache

def main():
    """Quick test of scaling features."""
    print("🚀 Generation 3: Quick Scaling Validation")
    
    results = {"generation_3_status": "TESTING"}
    
    try:
        # Test 1: Basic scalable hub
        hub = create_scalable_hub(population_size=20, generations=2)
        population = PromptPopulation.from_seeds([
            "Help me solve problems",
            "Please provide assistance", 
            "I will help you carefully"
        ])
        test_cases = [TestCase("test", "output", weight=1.0)]
        
        start_time = time.time()
        evolved = hub.evolve(population, test_cases)
        duration = time.time() - start_time
        
        throughput = len(population) / duration
        best_fitness = max(p.fitness_scores.get('fitness', 0) for p in evolved.prompts)
        
        print(f"✅ Scalable Evolution: {throughput:.1f} prompts/s, {best_fitness:.3f} fitness")
        
        # Test 2: Caching
        cache_key_test = evaluation_cache.cache_evaluation_result(
            "test prompt", ["input"], {"fitness": 0.8}
        )
        cached_result = evaluation_cache.get_evaluation_result("test prompt", ["input"])
        cache_works = cached_result is not None
        
        print(f"✅ Caching System: {'Working' if cache_works else 'Failed'}")
        
        # Test 3: Optimization metrics
        scaling_metrics = hub.get_scaling_metrics()
        optimizations_applied = scaling_metrics['scaling_metrics']['populations_processed'] > 0
        
        print(f"✅ Optimization Engine: {'Working' if optimizations_applied else 'Failed'}")
        
        hub.shutdown()
        
        # Results
        all_tests_passed = best_fitness > 0 and cache_works and optimizations_applied
        
        results.update({
            "generation_3_status": "COMPLETE" if all_tests_passed else "PARTIAL",
            "throughput": throughput,
            "best_fitness": best_fitness,
            "cache_working": cache_works,
            "optimizations_working": optimizations_applied,
            "features_working": {
                "scalable_evolution": best_fitness > 0,
                "caching_system": cache_works,
                "optimization_engine": optimizations_applied
            }
        })
        
        if all_tests_passed:
            print("🎉 GENERATION 3 COMPLETE - Scaling features working!")
        else:
            print("⚠️ GENERATION 3 PARTIAL - Some scaling issues")
            
    except Exception as e:
        print(f"❌ Generation 3 failed: {e}")
        results["generation_3_status"] = "FAILED"
        results["error"] = str(e)
    
    with open('/root/repo/gen3_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()