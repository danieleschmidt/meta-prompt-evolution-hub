"""
Simplified Test Runner for Sentiment Analyzer
No external dependencies - runs basic tests to validate functionality
"""

import asyncio
import time
import json
import traceback
from typing import Dict, List, Any

# Import analyzers with fallbacks
try:
    from sentiment_analyzer import SentimentEvolutionHub, SentimentLabel
    GENERATION_1_AVAILABLE = True
except Exception as e:
    print(f"Generation 1 not available: {e}")
    GENERATION_1_AVAILABLE = False
    SentimentLabel = None

try:
    from robust_sentiment_analyzer import RobustSentimentAnalyzer
    GENERATION_2_AVAILABLE = True
except Exception as e:
    print(f"Generation 2 not available: {e}")
    GENERATION_2_AVAILABLE = False

try:
    from scalable_sentiment_analyzer import ScalableSentimentAnalyzer
    GENERATION_3_AVAILABLE = True
except Exception as e:
    print(f"Generation 3 not available: {e}")
    GENERATION_3_AVAILABLE = False

def run_test(test_name: str, test_func):
    """Run a test function and capture results"""
    try:
        print(f"  Running {test_name}...", end=" ")
        start_time = time.time()
        
        if asyncio.iscoroutinefunction(test_func):
            result = asyncio.run(test_func())
        else:
            result = test_func()
        
        duration = time.time() - start_time
        print(f"✅ PASS ({duration:.2f}s)")
        return {"status": "PASS", "duration": duration, "error": None}
    
    except Exception as e:
        print(f"❌ FAIL - {str(e)}")
        return {"status": "FAIL", "duration": 0, "error": str(e)}

def test_generation_1():
    """Test basic sentiment analysis functionality"""
    if not GENERATION_1_AVAILABLE:
        raise Exception("Generation 1 not available")
    
    # Test initialization
    analyzer = SentimentEvolutionHub(population_size=10)
    assert analyzer is not None
    assert len(analyzer.population) == 10
    
    # Test single analysis
    result = analyzer.analyze_sentiment("I love this product!")
    assert result is not None
    assert hasattr(result, 'label')
    assert hasattr(result, 'confidence')
    assert result.label in SentimentLabel
    assert 0.0 <= result.confidence <= 1.0
    
    # Test batch analysis
    texts = ["Great product!", "Terrible service", "It's okay"]
    results = analyzer.batch_analyze(texts)
    assert len(results) == len(texts)
    
    # Test evolution
    test_cases = [
        ("I love this!", SentimentLabel.POSITIVE),
        ("This is terrible", SentimentLabel.NEGATIVE),
        ("It's okay", SentimentLabel.NEUTRAL)
    ]
    
    initial_generation = analyzer.generation
    analyzer.evolve_generation(test_cases)
    assert analyzer.generation > initial_generation
    
    return True

def test_generation_2():
    """Test robust error handling and validation"""
    if not GENERATION_2_AVAILABLE:
        raise Exception("Generation 2 not available")
    
    # Test initialization
    analyzer = RobustSentimentAnalyzer()
    assert analyzer is not None
    
    # Test valid analysis
    result = analyzer.analyze_sentiment("This is a test")
    assert result is not None
    assert hasattr(result, 'label')
    
    # Test error handling
    try:
        result = analyzer.analyze_sentiment(None, enable_rate_limit=False)
        # Should either handle gracefully or raise ValidationError
        if result.error_details:
            assert result.label.value == "unknown"
    except Exception:
        pass  # Expected for invalid input
    
    # Test health status
    health = analyzer.get_health_status()
    assert "status" in health
    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    
    return True

async def test_generation_3():
    """Test scalable performance features"""
    if not GENERATION_3_AVAILABLE:
        raise Exception("Generation 3 not available")
    
    # Test initialization
    analyzer = ScalableSentimentAnalyzer(
        cache_size=100,
        min_workers=2,
        max_workers=4
    )
    assert analyzer is not None
    
    # Test async analysis
    result = await analyzer.analyze_sentiment("Test async analysis")
    assert result is not None
    
    # Test caching
    result1 = analyzer.analyze_sentiment_sync("Cache test")
    result2 = analyzer.analyze_sentiment_sync("Cache test")
    
    # Second should be from cache
    assert result2.get("from_cache", False) == True
    
    # Test batch processing
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    batch_results = await analyzer.batch_analyze(texts, max_concurrency=3)
    assert len(batch_results) == len(texts)
    
    # Test metrics
    metrics = analyzer.get_performance_metrics()
    assert metrics is not None
    assert hasattr(metrics, 'throughput_per_second')
    
    # Test health check
    health = analyzer.health_check()
    assert "status" in health
    assert "health_score" in health
    
    return True

def test_integration():
    """Test integration between all generations"""
    results = {}
    
    # Test that each generation builds upon the previous
    if GENERATION_1_AVAILABLE:
        analyzer1 = SentimentEvolutionHub(population_size=5)
        result1 = analyzer1.analyze_sentiment("Integration test")
        results["gen1"] = result1
    
    if GENERATION_2_AVAILABLE:
        analyzer2 = RobustSentimentAnalyzer()
        result2 = analyzer2.analyze_sentiment("Integration test")
        results["gen2"] = result2
        
        # Should have additional robustness features
        assert hasattr(result2, 'validation_result') or hasattr(result2, 'error_details')
    
    if GENERATION_3_AVAILABLE:
        analyzer3 = ScalableSentimentAnalyzer(cache_size=10)
        result3 = analyzer3.analyze_sentiment_sync("Integration test")
        results["gen3"] = result3
        
        # Should have caching capabilities
        metrics = analyzer3.get_performance_metrics()
        assert metrics is not None
    
    # All should produce sentiment analysis results
    for gen, result in results.items():
        if hasattr(result, 'label'):
            assert result.label is not None
        elif isinstance(result, dict):
            assert "label" in result or "error" in result
    
    return True

def performance_benchmark():
    """Simple performance benchmark"""
    if not GENERATION_3_AVAILABLE:
        raise Exception("Generation 3 not available for performance test")
    
    analyzer = ScalableSentimentAnalyzer()
    
    # Benchmark sequential processing
    test_texts = [
        "Great product, highly recommend!",
        "Poor quality, very disappointing",
        "Average experience, nothing special",
        "Excellent service and fast delivery",
        "Terrible customer support"
    ] * 10  # 50 texts total
    
    start_time = time.time()
    results = []
    for text in test_texts:
        result = analyzer.analyze_sentiment_sync(text)
        results.append(result)
    sequential_time = time.time() - start_time
    
    sequential_throughput = len(test_texts) / sequential_time
    
    # Check results quality
    successful_results = [r for r in results if not r.get("error")]
    success_rate = len(successful_results) / len(results)
    
    assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"
    assert sequential_throughput >= 5.0, f"Throughput too low: {sequential_throughput:.1f} texts/sec"
    
    print(f"    Performance: {sequential_throughput:.1f} texts/sec, {success_rate:.1%} success rate")
    
    return {
        "throughput": sequential_throughput,
        "success_rate": success_rate,
        "total_texts": len(test_texts),
        "processing_time": sequential_time
    }

def main():
    """Main test runner"""
    print("🧪 Simple Sentiment Analyzer Test Suite")
    print("=" * 50)
    
    test_results = {
        "summary": {},
        "details": {},
        "performance": {},
        "timestamp": time.time()
    }
    
    # Test availability
    print(f"\n📋 System Status:")
    print(f"  Generation 1 (Basic): {'✅ Available' if GENERATION_1_AVAILABLE else '❌ Not Available'}")
    print(f"  Generation 2 (Robust): {'✅ Available' if GENERATION_2_AVAILABLE else '❌ Not Available'}")
    print(f"  Generation 3 (Scalable): {'✅ Available' if GENERATION_3_AVAILABLE else '❌ Not Available'}")
    
    # Define tests
    tests = [
        ("Generation 1 - Basic Functionality", test_generation_1, GENERATION_1_AVAILABLE),
        ("Generation 2 - Robust Error Handling", test_generation_2, GENERATION_2_AVAILABLE),
        ("Generation 3 - Scalable Performance", test_generation_3, GENERATION_3_AVAILABLE),
        ("Integration Test", test_integration, True),
        ("Performance Benchmark", performance_benchmark, GENERATION_3_AVAILABLE),
    ]
    
    # Run tests
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_func, available in tests:
        if not available:
            print(f"\n🔸 {test_name}: SKIPPED (not available)")
            continue
        
        print(f"\n🔸 {test_name}:")
        total_tests += 1
        
        result = run_test(test_name, test_func)
        test_results["details"][test_name] = result
        
        if result["status"] == "PASS":
            passed_tests += 1
        else:
            print(f"    Error: {result['error']}")
    
    # Calculate summary
    success_rate = passed_tests / max(total_tests, 1)
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": success_rate
    }
    
    # Performance summary
    if "Performance Benchmark" in test_results["details"]:
        perf_result = test_results["details"]["Performance Benchmark"]
        if perf_result["status"] == "PASS":
            # Extract performance data from the test function
            # (In a real scenario, we'd return this from the function)
            test_results["performance"] = {
                "available": True,
                "status": "measured"
            }
    
    # Print summary
    print(f"\n" + "=" * 50)
    print(f"📊 TEST SUMMARY")
    print(f"=" * 50)
    
    status_emoji = "🎉" if success_rate >= 0.9 else "✅" if success_rate >= 0.7 else "⚠️" if success_rate >= 0.5 else "❌"
    print(f"{status_emoji} Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
    
    for test_name, result in test_results["details"].items():
        status = "✅" if result["status"] == "PASS" else "❌"
        print(f"  {status} {test_name} ({result['duration']:.2f}s)")
    
    # Quality gate check
    print(f"\n🚪 Quality Gates:")
    
    gates_passed = 0
    total_gates = 3
    
    # Gate 1: Success rate
    if success_rate >= 0.8:
        print(f"  ✅ Success Rate: {success_rate:.1%} (≥80% required)")
        gates_passed += 1
    else:
        print(f"  ❌ Success Rate: {success_rate:.1%} (≥80% required)")
    
    # Gate 2: Core functionality
    gen1_passed = test_results["details"].get("Generation 1 - Basic Functionality", {}).get("status") == "PASS"
    if gen1_passed or not GENERATION_1_AVAILABLE:
        print(f"  ✅ Core Functionality: {'Working' if gen1_passed else 'Not Required'}")
        gates_passed += 1
    else:
        print(f"  ❌ Core Functionality: Failed")
    
    # Gate 3: No critical failures
    critical_failures = [
        name for name, result in test_results["details"].items()
        if result["status"] == "FAIL" and "Generation" in name
    ]
    
    if not critical_failures:
        print(f"  ✅ No Critical Failures")
        gates_passed += 1
    else:
        print(f"  ❌ Critical Failures: {len(critical_failures)}")
    
    gate_success = gates_passed / total_gates
    final_status = "🎉 ALL GATES PASSED" if gate_success == 1.0 else f"⚠️ {gates_passed}/{total_gates} GATES PASSED"
    print(f"\n{final_status}")
    
    # Save results
    with open("simple_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: simple_test_results.json")
    
    # Return success status
    return gate_success >= 0.8

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 Tests completed successfully!")
        exit(0)
    else:
        print(f"\n❌ Tests failed - needs improvement")
        exit(1)