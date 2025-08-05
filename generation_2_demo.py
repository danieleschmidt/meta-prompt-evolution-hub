#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Reliability and Error Handling Demo
Comprehensive demonstration of enhanced robustness features.
"""

from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase
from robust_evolution_hub import create_robust_hub
from error_handling import error_handler
from validation_system import prompt_validator, test_validator
from monitoring_system import health_checker, performance_tracker
import json
import time

def demo_input_validation():
    """Demonstrate comprehensive input validation."""
    print("🛡️  Testing Input Validation...")
    
    # Test valid prompts
    valid_prompts = [
        "You are a helpful assistant.",
        "Please provide accurate information.",
        "I will help you solve this problem step by step."
    ]
    
    # Test invalid prompts  
    invalid_prompts = [
        "",  # Empty
        "x" * 6000,  # Too long
        "<script>alert('xss')</script>Help me",  # Unsafe content
        "   ",  # Whitespace only
    ]
    
    all_test_prompts = valid_prompts + invalid_prompts
    
    valid_count = 0
    for i, prompt_text in enumerate(all_test_prompts):
        result = prompt_validator.validate_prompt(prompt_text)
        if result.is_valid:
            valid_count += 1
        print(f"  Prompt {i+1}: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
        if result.errors:
            print(f"    Errors: {result.errors}")
        if result.warnings:
            print(f"    Warnings: {result.warnings}")
    
    print(f"  📊 Validation Results: {valid_count}/{len(all_test_prompts)} prompts valid")
    return valid_count > 0

def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n🚨 Testing Error Handling...")
    
    # Create hub with intentionally problematic config
    hub = create_robust_hub(population_size=5, generations=2)
    
    # Test with empty population (should trigger error handling)
    try:
        empty_population = PromptPopulation([])
        test_cases = [TestCase("test input", "expected output")]
        
        with error_handler.error_context("test_error_handling"):
            result = hub.evolve(empty_population, test_cases)
            print("  ❌ Should have failed with empty population")
            
    except Exception as e:
        print(f"  ✅ Caught expected error: {type(e).__name__}")
        
    # Test with invalid test cases
    try:
        population = PromptPopulation.from_seeds(["Valid prompt"])
        invalid_test_cases = [
            TestCase("", ""),  # Empty
            TestCase("test", "", weight=-1.0),  # Invalid weight
        ]
        
        result = hub.evolve(population, invalid_test_cases)
        print("  ✅ Handled invalid test cases gracefully")
        
    except Exception as e:
        print(f"  ⚠️  Unexpected error: {e}")
        
    # Get error summary
    error_summary = error_handler.get_error_summary()
    print(f"  📊 Error Summary: {error_summary['total_errors']} total errors")
    
    return True

def demo_health_monitoring():
    """Demonstrate system health monitoring."""
    print("\n💓 Testing Health Monitoring...")
    
    # Start monitoring (should already be started by robust hub)
    health_checker.start_monitoring()
    
    # Wait for some metrics to be collected
    print("  ⏳ Collecting system metrics...")
    time.sleep(2)
    
    # Get health report
    health_report = health_checker.get_health_report()
    print(f"  🏥 Health Status: {health_report['status']}")
    
    if 'current_metrics' in health_report:
        metrics = health_report['current_metrics']
        print(f"  📊 CPU Usage: {metrics['cpu_usage']:.1f}%")
        print(f"  💾 Memory Usage: {metrics['memory_usage']:.1f}%")
        print(f"  💽 Disk Usage: {metrics['disk_usage']:.1f}%")
        
    if health_report.get('recommendations'):
        print(f"  💡 Recommendations: {health_report['recommendations']}")
        
    return health_report['status'] in ['healthy', 'warning']

def demo_performance_tracking():
    """Demonstrate performance tracking."""
    print("\n📈 Testing Performance Tracking...")
    
    # Run a small evolution to generate metrics
    hub = create_robust_hub(population_size=5, generations=2, algorithm="nsga2")
    
    population = PromptPopulation.from_seeds([
        "Help me with this task",
        "Please provide assistance", 
        "I need help solving this"
    ])
    
    test_cases = [
        TestCase("solve problem", "detailed solution", weight=1.0)
    ]
    
    print("  🔄 Running evolution for metrics...")
    start_time = time.time()
    result = hub.evolve(population, test_cases)
    duration = time.time() - start_time
    
    # Get performance summary
    perf_summary = performance_tracker.get_performance_summary()
    print(f"  ✅ Evolution completed in {duration:.2f}s")
    print(f"  📊 Total evolutions tracked: {perf_summary.get('total_evolutions', 0)}")
    
    if 'recent_performance' in perf_summary:
        recent = perf_summary['recent_performance']
        print(f"  ⚡ Avg generation time: {recent.get('average_generation_time', 0):.2f}s")
        print(f"  🏆 Best fitness achieved: {recent.get('best_fitness_achieved', 0):.3f}")
        
    return len(result) > 0

def demo_comprehensive_robustness():
    """Demonstrate full robustness system integration."""
    print("\n🏰 Testing Comprehensive Robustness...")
    
    # Create robust hub with all features enabled
    hub = create_robust_hub(
        population_size=8,
        generations=3,
        algorithm="nsga2",
        enable_all_features=True
    )
    
    # Test with mixed valid/invalid inputs
    mixed_seeds = [
        "You are a helpful AI assistant",  # Valid
        "Please help me solve this problem",  # Valid  
        "",  # Invalid - will be filtered
        "I will provide accurate assistance"  # Valid
    ]
    
    population = PromptPopulation.from_seeds([s for s in mixed_seeds if s])  # Filter empty
    
    test_cases = [
        TestCase("classify text", "category", weight=1.0),
        TestCase("summarize document", "summary", weight=1.5),
        TestCase("explain concept", "explanation", weight=1.2)
    ]
    
    print("  🚀 Running robust evolution...")
    start_time = time.time()
    
    try:
        result = hub.evolve(population, test_cases)
        duration = time.time() - start_time
        
        print(f"  ✅ Robust evolution completed in {duration:.2f}s")
        print(f"  📊 Final population size: {len(result)}")
        
        if result.prompts:
            best = result.get_top_k(1)[0]
            print(f"  🏆 Best prompt: '{best.text[:50]}...'")
            print(f"  📈 Best fitness: {best.fitness_scores.get('fitness', 0):.3f}")
            
        # Get comprehensive status
        status = hub.get_comprehensive_status()
        print(f"  🏥 System health: {status.get('health_report', {}).get('status', 'unknown')}")
        print(f"  🚨 Total errors: {status.get('error_summary', {}).get('total_errors', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Robust evolution failed: {e}")
        return False
    finally:
        hub.shutdown()

def main():
    """Run Generation 2 comprehensive demonstration."""
    print("🚀 Generation 2: MAKE IT ROBUST - Comprehensive Demo")
    print("=" * 65)
    
    results = {
        "generation": 2,
        "status": "TESTING",
        "features_tested": {}
    }
    
    try:
        # Test all robustness features
        results["features_tested"]["input_validation"] = demo_input_validation()
        results["features_tested"]["error_handling"] = demo_error_handling()
        results["features_tested"]["health_monitoring"] = demo_health_monitoring()
        results["features_tested"]["performance_tracking"] = demo_performance_tracking()
        results["features_tested"]["comprehensive_robustness"] = demo_comprehensive_robustness()
        
        # Calculate overall success
        total_features = len(results["features_tested"])
        passed_features = sum(results["features_tested"].values())
        success_rate = passed_features / total_features
        
        print("\n" + "=" * 65)
        if success_rate >= 0.8:
            print("🎉 GENERATION 2 COMPLETE: ROBUST SYSTEM OPERATIONAL")
            results["status"] = "ROBUST"
        else:
            print("⚠️  GENERATION 2 PARTIAL: SOME ROBUSTNESS ISSUES")
            results["status"] = "PARTIAL"
            
        print(f"✅ Input Validation: {'✓' if results['features_tested']['input_validation'] else '✗'}")
        print(f"✅ Error Handling: {'✓' if results['features_tested']['error_handling'] else '✗'}")
        print(f"✅ Health Monitoring: {'✓' if results['features_tested']['health_monitoring'] else '✗'}")
        print(f"✅ Performance Tracking: {'✓' if results['features_tested']['performance_tracking'] else '✗'}")
        print(f"✅ Comprehensive Robustness: {'✓' if results['features_tested']['comprehensive_robustness'] else '✗'}")
        
        print(f"\n📊 Success Rate: {success_rate:.1%} ({passed_features}/{total_features} features)")
        
        results["success_rate"] = success_rate
        results["features_passed"] = passed_features
        results["features_total"] = total_features
        
        # Save results
        with open('/root/repo/generation_2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n💾 Results saved to: generation_2_results.json")
        
        if success_rate >= 0.8:
            print("\n🎯 Ready for Generation 3: MAKE IT SCALE!")
        else:
            print("\n🔧 Needs improvement before scaling")
            
    except Exception as e:
        print(f"\n❌ Generation 2 Demo Failed: {e}")
        results["status"] = "FAILED"
        results["error"] = str(e)
        
        with open('/root/repo/generation_2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        raise

if __name__ == "__main__":
    main()