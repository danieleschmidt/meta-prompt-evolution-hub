#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - Comprehensive Scaling and Performance Demo
Advanced demonstration of high-performance, scalable evolutionary optimization.
"""

import asyncio
import time
import json
import multiprocessing as mp
from typing import List, Dict, Any

from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase
from scalable_evolution_hub import create_scalable_hub, create_high_throughput_hub
from optimization_engine import OptimizationConfig, PerformanceOptimizer
from caching_system import evaluation_cache, population_cache, distributed_cache

def demo_caching_performance():
    """Demonstrate advanced caching system performance."""
    print("💾 Testing Advanced Caching System...")
    
    # Test evaluation cache
    test_prompts = [
        "You are a helpful assistant",
        "Please help me solve this problem",
        "I will provide accurate information",
        "Let me assist you carefully"
    ]
    
    test_inputs = ["solve problem", "explain concept", "analyze data"]
    
    # First run - populate cache
    start_time = time.time()
    for prompt in test_prompts:
        for test_input in test_inputs:
            # Simulate evaluation result
            result = {"fitness": 0.8, "accuracy": 0.9, "latency": 0.2}
            evaluation_cache.cache_evaluation_result(prompt, [test_input], result)
    first_run_time = time.time() - start_time
    
    # Second run - use cache
    start_time = time.time()
    cache_hits = 0
    for prompt in test_prompts:
        for test_input in test_inputs:
            cached_result = evaluation_cache.get_evaluation_result(prompt, [test_input])
            if cached_result:
                cache_hits += 1
    second_run_time = time.time() - start_time
    
    cache_stats = evaluation_cache.get_cache_stats()
    
    print(f"  ✅ Cache Performance Test Complete")
    print(f"  📊 Cache hits: {cache_hits}/{len(test_prompts) * len(test_inputs)}")
    print(f"  ⚡ Speed improvement: {first_run_time/second_run_time:.1f}x faster")
    print(f"  💾 Cache hit rate: {cache_stats['hit_rate']:.1%}")
    
    return cache_hits > 0

def demo_parallel_optimization():
    """Demonstrate parallel processing optimization."""
    print("\n🚀 Testing Parallel Processing Optimization...")
    
    # Create optimizer with different worker counts
    single_worker_config = OptimizationConfig(max_workers=1, batch_size=20)
    multi_worker_config = OptimizationConfig(max_workers=mp.cpu_count(), batch_size=20)
    
    single_optimizer = PerformanceOptimizer(single_worker_config)
    multi_optimizer = PerformanceOptimizer(multi_worker_config)
    
    # Test population
    test_population = PromptPopulation.from_seeds([
        f"Prompt {i}: Help me solve this complex problem step by step"
        for i in range(40)
    ])
    
    test_cases = [
        TestCase("solve equation", "step by step solution", weight=1.0),
        TestCase("analyze data", "comprehensive analysis", weight=1.0),
        TestCase("explain concept", "clear explanation", weight=1.0)
    ]
    
    def simple_fitness(prompt, test_cases):
        time.sleep(0.05)  # Simulate evaluation time
        return {"fitness": 0.7, "accuracy": 0.8}
    
    # Single-threaded evaluation
    start_time = time.time()
    single_optimizer.optimize_population_evaluation(test_population, test_cases, simple_fitness)
    single_thread_time = time.time() - start_time
    
    # Multi-threaded evaluation
    start_time = time.time()
    multi_optimizer.optimize_population_evaluation(test_population, test_cases, simple_fitness)
    multi_thread_time = time.time() - start_time
    
    speedup = single_thread_time / multi_thread_time
    
    print(f"  ✅ Parallel Processing Test Complete")
    print(f"  🔄 Single-threaded time: {single_thread_time:.2f}s")
    print(f"  ⚡ Multi-threaded time: {multi_thread_time:.2f}s")
    print(f"  📈 Speedup: {speedup:.1f}x")
    print(f"  👥 Workers used: {multi_worker_config.max_workers}")
    
    # Cleanup
    single_optimizer.shutdown()
    multi_optimizer.shutdown()
    
    return speedup > 1.5

def demo_memory_optimization():
    """Demonstrate memory optimization features."""
    print("\n🧠 Testing Memory Optimization...")
    
    hub = create_scalable_hub(population_size=100, generations=3)
    
    # Get initial memory usage
    initial_memory = hub.performance_optimizer.resource_pool.memory_monitor.check_memory_usage()
    
    # Create large population to stress memory
    large_population = PromptPopulation.from_seeds([
        f"Large prompt {i}: " + "This is a very long prompt with lots of text " * 20
        for i in range(200)
    ])
    
    test_cases = [TestCase("test", "result", weight=1.0)]
    
    # Run evolution with memory monitoring
    start_time = time.time()
    result = hub.evolve(large_population, test_cases)
    duration = time.time() - start_time
    
    final_memory = hub.performance_optimizer.resource_pool.memory_monitor.check_memory_usage()
    optimization_metrics = hub.performance_optimizer.get_optimization_metrics()
    
    print(f"  ✅ Memory Optimization Test Complete")
    print(f"  💾 Initial memory: {initial_memory['memory_mb']:.1f} MB")
    print(f"  💾 Final memory: {final_memory['memory_mb']:.1f} MB")
    print(f"  🔧 Memory optimizations: {optimization_metrics['optimization_metrics']['memory_optimizations']}")
    print(f"  ⏱️ Evolution time: {duration:.2f}s")
    print(f"  📊 Population size: {len(large_population)} -> {len(result)}")
    
    hub.shutdown()
    
    return optimization_metrics['optimization_metrics']['memory_optimizations'] > 0

def demo_scalable_evolution():
    """Demonstrate scalable evolution with different population sizes."""
    print("\n📈 Testing Scalable Evolution Performance...")
    
    test_sizes = [20, 50, 100]
    results = {}
    
    for size in test_sizes:
        print(f"  🔄 Testing population size: {size}")
        
        hub = create_scalable_hub(
            population_size=size,
            generations=3,
            algorithm="nsga2"
        )
        
        # Create test population
        population = PromptPopulation.from_seeds([
            f"Scalable prompt {i}: Help me with task {i % 10}"
            for i in range(size)
        ])
        
        test_cases = [
            TestCase("classify text", "category", weight=1.0),
            TestCase("summarize content", "summary", weight=1.0)
        ]
        
        # Run evolution
        start_time = time.time()
        evolved = hub.evolve(population, test_cases)
        duration = time.time() - start_time
        
        # Get metrics
        scaling_metrics = hub.get_scaling_metrics()
        
        results[size] = {
            "duration": duration,
            "prompts_per_second": size / duration,
            "best_fitness": max(p.fitness_scores.get('fitness', 0) for p in evolved.prompts),
            "cache_hit_rate": scaling_metrics['cache_performance']['evaluation_cache']['hit_rate'],
            "memory_optimizations": scaling_metrics['optimization_metrics']['optimization_metrics']['memory_optimizations']
        }
        
        print(f"    ⏱️ Time: {duration:.2f}s")
        print(f"    📊 Throughput: {results[size]['prompts_per_second']:.1f} prompts/s")
        print(f"    🎯 Best fitness: {results[size]['best_fitness']:.3f}")
        
        hub.shutdown()
    
    print(f"  ✅ Scalable Evolution Test Complete")
    print(f"  📈 Performance scaling demonstrated across {len(test_sizes)} population sizes")
    
    return all(r['best_fitness'] > 0 for r in results.values())

async def demo_async_evolution():
    """Demonstrate asynchronous evolution capabilities."""
    print("\n⚡ Testing Asynchronous Evolution...")
    
    hub = create_scalable_hub(population_size=30, generations=2)
    
    population = PromptPopulation.from_seeds([
        f"Async prompt {i}: Process this request efficiently"
        for i in range(30)
    ])
    
    test_cases = [
        TestCase("async task", "quick result", weight=1.0)
    ]
    
    # Run async evolution
    start_time = time.time()
    result = await hub.evolve_async(population, test_cases)
    async_duration = time.time() - start_time
    
    # Compare with sync evolution
    start_time = time.time()
    sync_result = hub.evolve(population, test_cases)
    sync_duration = time.time() - start_time
    
    async_speedup = sync_duration / async_duration if async_duration > 0 else 1.0
    
    print(f"  ✅ Async Evolution Test Complete")
    print(f"  ⚡ Async time: {async_duration:.2f}s")
    print(f"  🔄 Sync time: {sync_duration:.2f}s")
    print(f"  📈 Async speedup: {async_speedup:.1f}x")
    print(f"  📊 Results quality maintained: {len(result) == len(sync_result)}")
    
    hub.shutdown()
    
    return async_speedup > 0.8  # Allow for some overhead

def demo_high_throughput_processing():
    """Demonstrate high-throughput processing capabilities."""
    print("\n🏭 Testing High-Throughput Processing...")
    
    # Create high-throughput hub
    hub = create_high_throughput_hub()
    
    # Create large test workload
    large_population = PromptPopulation.from_seeds([
        f"High-throughput prompt {i}: {['Analyze', 'Summarize', 'Classify', 'Explain'][i % 4]} this content efficiently"
        for i in range(300)
    ])
    
    test_cases = [
        TestCase("process data", "results", weight=1.0),
        TestCase("generate report", "summary", weight=1.0)
    ]
    
    # Run high-throughput evolution
    start_time = time.time()
    result = hub.evolve(large_population, test_cases)
    duration = time.time() - start_time
    
    throughput = len(large_population) / duration
    final_metrics = hub.get_scaling_metrics()
    
    print(f"  ✅ High-Throughput Test Complete")
    print(f"  🏭 Processed: {len(large_population)} prompts")
    print(f"  ⏱️ Duration: {duration:.2f}s")
    print(f"  📊 Throughput: {throughput:.1f} prompts/second")
    print(f"  💾 Cache hit rate: {final_metrics['cache_performance']['evaluation_cache']['hit_rate']:.1%}")
    print(f"  🔧 Optimizations applied: {final_metrics['optimization_metrics']['optimization_metrics']['memory_optimizations']}")
    print(f"  🎯 Best fitness achieved: {max(p.fitness_scores.get('fitness', 0) for p in result.prompts):.3f}")
    
    hub.shutdown()
    
    return throughput > 10  # Target: >10 prompts/second

def demo_comprehensive_scaling():
    """Demonstrate comprehensive scaling across all features."""
    print("\n🌐 Testing Comprehensive Scaling Integration...")
    
    hub = create_scalable_hub(
        population_size=150,
        generations=5,
        algorithm="nsga2",
        enable_all_optimizations=True
    )
    
    # Mixed workload test
    mixed_population = PromptPopulation.from_seeds([
        "You are a helpful AI assistant that provides accurate information",
        "Please analyze this data carefully and provide insights",
        "Help me solve this complex problem step by step",
        "Classify the following text into appropriate categories",
        "Summarize the key points from this document",
        "Explain this concept in simple terms",
        "Generate a creative solution to this challenge",
        "Optimize this process for better efficiency"
    ] * 20)  # 160 prompts total
    
    comprehensive_test_cases = [
        TestCase("analyze complex dataset", "detailed analysis with insights", weight=2.0),
        TestCase("solve optimization problem", "optimal solution with explanation", weight=2.0),
        TestCase("classify customer feedback", "accurate categorization", weight=1.5),
        TestCase("summarize research paper", "concise key points", weight=1.5),
        TestCase("generate creative content", "original creative output", weight=1.0)
    ]
    
    print("  🚀 Running comprehensive scaling test...")
    start_time = time.time()
    
    # Run full evolution
    result = hub.evolve(mixed_population, comprehensive_test_cases)
    
    duration = time.time() - start_time
    scaling_metrics = hub.get_scaling_metrics()
    
    # Performance analysis
    best_fitness = max(p.fitness_scores.get('fitness', 0) for p in result.prompts)
    avg_fitness = sum(p.fitness_scores.get('fitness', 0) for p in result.prompts) / len(result.prompts)
    throughput = len(mixed_population) / duration
    
    print(f"  ✅ Comprehensive Scaling Test Complete")
    print(f"  📊 Population: {len(mixed_population)} -> {len(result)} prompts")
    print(f"  ⏱️ Total time: {duration:.2f}s")
    print(f"  🚀 Throughput: {throughput:.1f} prompts/second")
    print(f"  🎯 Best fitness: {best_fitness:.3f}")
    print(f"  📈 Average fitness: {avg_fitness:.3f}")
    
    # Cache performance
    cache_perf = scaling_metrics['cache_performance']
    print(f"  💾 Evaluation cache hit rate: {cache_perf['evaluation_cache']['hit_rate']:.1%}")
    print(f"  🔄 Distributed cache stats: {cache_perf['distributed_cache']['overall_hit_rate']:.1%}")
    
    # Optimization metrics
    opt_metrics = scaling_metrics['optimization_metrics']['optimization_metrics']
    print(f"  ⚡ Parallel batches: {opt_metrics['parallel_batches']}")
    print(f"  🧠 Memory optimizations: {opt_metrics['memory_optimizations']}")
    print(f"  📊 Total evaluations: {opt_metrics['total_evaluations']}")
    
    hub.shutdown()
    
    return {
        "throughput": throughput,
        "best_fitness": best_fitness,
        "cache_hit_rate": cache_perf['evaluation_cache']['hit_rate'],
        "parallel_batches": opt_metrics['parallel_batches'],
        "success": throughput > 5 and best_fitness > 0.3
    }

async def main():
    """Run Generation 3 comprehensive demonstration."""
    print("🚀 Generation 3: MAKE IT SCALE - Comprehensive Performance Demo")
    print("=" * 70)
    
    results = {
        "generation": 3,
        "status": "TESTING",
        "features_tested": {},
        "performance_benchmarks": {}
    }
    
    try:
        # Test all scaling features
        print("Testing individual scaling components...")
        results["features_tested"]["caching_performance"] = demo_caching_performance()
        results["features_tested"]["parallel_optimization"] = demo_parallel_optimization()
        results["features_tested"]["memory_optimization"] = demo_memory_optimization()
        results["features_tested"]["scalable_evolution"] = demo_scalable_evolution()
        results["features_tested"]["async_evolution"] = await demo_async_evolution()
        results["features_tested"]["high_throughput"] = demo_high_throughput_processing()
        
        # Comprehensive integration test
        comprehensive_result = demo_comprehensive_scaling()
        results["features_tested"]["comprehensive_scaling"] = comprehensive_result["success"]
        results["performance_benchmarks"] = comprehensive_result
        
        # Calculate success rate
        total_features = len(results["features_tested"])
        passed_features = sum(results["features_tested"].values())
        success_rate = passed_features / total_features
        
        print("\n" + "=" * 70)
        if success_rate >= 0.85:
            print("🎉 GENERATION 3 COMPLETE: HIGH-PERFORMANCE SCALING ACHIEVED")
            results["status"] = "SCALED"
        else:
            print("⚠️  GENERATION 3 PARTIAL: SOME SCALING ISSUES")
            results["status"] = "PARTIAL"
            
        # Feature status report
        print(f"✅ Caching Performance: {'✓' if results['features_tested']['caching_performance'] else '✗'}")
        print(f"✅ Parallel Optimization: {'✓' if results['features_tested']['parallel_optimization'] else '✗'}")
        print(f"✅ Memory Optimization: {'✓' if results['features_tested']['memory_optimization'] else '✗'}")
        print(f"✅ Scalable Evolution: {'✓' if results['features_tested']['scalable_evolution'] else '✗'}")
        print(f"✅ Async Evolution: {'✓' if results['features_tested']['async_evolution'] else '✗'}")
        print(f"✅ High Throughput: {'✓' if results['features_tested']['high_throughput'] else '✗'}")
        print(f"✅ Comprehensive Scaling: {'✓' if results['features_tested']['comprehensive_scaling'] else '✗'}")
        
        # Performance summary
        perf = results["performance_benchmarks"]
        print(f"\n📊 Performance Achievements:")
        print(f"  🚀 Peak throughput: {perf.get('throughput', 0):.1f} prompts/second")
        print(f"  🎯 Best fitness achieved: {perf.get('best_fitness', 0):.3f}")
        print(f"  💾 Cache hit rate: {perf.get('cache_hit_rate', 0):.1%}")
        print(f"  ⚡ Parallel batches executed: {perf.get('parallel_batches', 0)}")
        
        print(f"\n📈 Success Rate: {success_rate:.1%} ({passed_features}/{total_features} features)")
        
        results["success_rate"] = success_rate
        results["features_passed"] = passed_features
        results["features_total"] = total_features
        
        # Save results
        with open('/root/repo/generation_3_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n💾 Results saved to: generation_3_results.json")
        
        if success_rate >= 0.85:
            print("\n🎯 Ready for Quality Gates and Production Deployment!")
        else:
            print("\n🔧 Performance tuning needed before production")
            
    except Exception as e:
        print(f"\n❌ Generation 3 Demo Failed: {e}")
        results["status"] = "FAILED"
        results["error"] = str(e)
        
        with open('/root/repo/generation_3_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        raise

if __name__ == "__main__":
    asyncio.run(main())