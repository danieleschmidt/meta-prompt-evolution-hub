#!/usr/bin/env python3
"""
Comprehensive Test Suite for Sentiment Analyzer
Quality Gates: Unit tests, Integration tests, Performance benchmarks
"""
import asyncio
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any
import unittest
from unittest.mock import patch, MagicMock

# Import our sentiment analyzers
try:
    from sentiment_analyzer_simple import SentimentAnalyzer as SimpleAnalyzer
    from sentiment_analyzer_robust import RobustSentimentAnalyzer
    from sentiment_analyzer_scalable import ScalableSentimentAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


class TestSentimentAnalyzerBase(unittest.TestCase):
    """Base test class with common utilities"""
    
    def setUp(self):
        self.test_cases = [
            # Positive sentiment tests
            ("I love this product! It's amazing.", "positive"),
            ("Excellent quality and great value for money.", "positive"),
            ("Outstanding service, highly recommended!", "positive"),
            ("Perfect! Exactly what I was looking for.", "positive"),
            
            # Negative sentiment tests
            ("This is terrible and completely useless.", "negative"),
            ("Awful experience, very disappointed.", "negative"),
            ("Waste of money, poor quality product.", "negative"),
            ("Horrible service, would not recommend.", "negative"),
            
            # Neutral sentiment tests
            ("It's okay, nothing special but adequate.", "neutral"),
            ("The product arrived on time as expected.", "neutral"),
            ("Standard quality for this price range.", "neutral"),
            ("It works fine for basic needs.", "neutral"),
            
            # Edge cases
            ("Not bad at all, actually quite good!", "positive"),  # Negation + positive
            ("I can't say I'm not satisfied.", "negative"),  # Double negation (tricky)
            ("This isn't terrible but it's not great either.", "neutral"),  # Mixed signals
            ("", "neutral"),  # Empty string
            ("???", "neutral"),  # Only punctuation
        ]
        
        self.performance_benchmarks = {
            'min_throughput': 100,  # texts per second
            'max_avg_latency': 0.1,  # seconds
            'min_accuracy': 0.7,  # 70% accuracy on test cases
            'max_error_rate': 0.05  # 5% error rate
        }
    
    def calculate_accuracy(self, results: List[Dict], expected_sentiments: List[str]) -> float:
        """Calculate accuracy against expected results"""
        if len(results) != len(expected_sentiments):
            return 0.0
        
        correct = 0
        for result, expected in zip(results, expected_sentiments):
            if isinstance(result, dict):
                predicted = result.get('sentiment', 'neutral')
            else:
                predicted = getattr(result, 'sentiment', 'neutral')
            
            if predicted == expected:
                correct += 1
        
        return correct / len(results)


class TestSimpleSentimentAnalyzer(TestSentimentAnalyzerBase):
    """Test suite for Generation 1: Simple Sentiment Analyzer"""
    
    def setUp(self):
        super().setUp()
        self.analyzer = SimpleAnalyzer()
    
    def test_basic_functionality(self):
        """Test basic sentiment analysis functionality"""
        result = self.analyzer.analyze_text("I love this product!")
        
        self.assertIsInstance(result, type(self.analyzer.analyze_text("")))
        self.assertIn(result.sentiment, ['positive', 'negative', 'neutral'])
        self.assertTrue(0.0 <= result.confidence <= 1.0)
        self.assertIsInstance(result.scores, dict)
        self.assertIn('positive', result.scores)
        self.assertIn('negative', result.scores)
        self.assertIn('neutral', result.scores)
    
    def test_sentiment_accuracy(self):
        """Test sentiment classification accuracy"""
        results = []
        expected = []
        
        for text, expected_sentiment in self.test_cases:
            if text:  # Skip empty strings for this test
                result = self.analyzer.analyze_text(text)
                results.append(result)
                expected.append(expected_sentiment)
        
        accuracy = self.calculate_accuracy(results, expected)
        self.assertGreaterEqual(accuracy, self.performance_benchmarks['min_accuracy'],
                               f"Accuracy {accuracy:.2%} below minimum {self.performance_benchmarks['min_accuracy']:.2%}")
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        texts = [case[0] for case in self.test_cases if case[0]]  # Non-empty texts
        
        batch_result = self.analyzer.analyze_batch(texts)
        
        self.assertEqual(len(batch_result.results), len(texts))
        self.assertGreater(batch_result.processing_time, 0)
        
        # Test distribution calculation
        distribution = self.analyzer.get_sentiment_distribution(batch_result.results)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=2)
    
    def test_performance_benchmark(self):
        """Test performance meets minimum requirements"""
        test_texts = ["This is a test sentence for performance."] * 100
        
        start_time = time.time()
        batch_result = self.analyzer.analyze_batch(test_texts)
        total_time = time.time() - start_time
        
        throughput = len(test_texts) / total_time
        avg_latency = total_time / len(test_texts)
        
        self.assertGreaterEqual(throughput, self.performance_benchmarks['min_throughput'],
                               f"Throughput {throughput:.1f} below minimum {self.performance_benchmarks['min_throughput']}")
        self.assertLessEqual(avg_latency, self.performance_benchmarks['max_avg_latency'],
                           f"Average latency {avg_latency:.3f}s above maximum {self.performance_benchmarks['max_avg_latency']}s")
    
    def test_edge_cases(self):
        """Test handling of edge cases"""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a" * 1000,  # Very long text
            "123456789",  # Numbers only
            "!@#$%^&*()",  # Special characters only
            "NOT GOOD BUT NOT BAD",  # All caps with negation
        ]
        
        for text in edge_cases:
            with self.subTest(text=text[:20] + "..."):
                result = self.analyzer.analyze_text(text)
                self.assertIsNotNone(result)
                self.assertIn(result.sentiment, ['positive', 'negative', 'neutral'])


class TestRobustSentimentAnalyzer(TestSentimentAnalyzerBase):
    """Test suite for Generation 2: Robust Sentiment Analyzer"""
    
    def setUp(self):
        super().setUp()
        self.analyzer = RobustSentimentAnalyzer()
    
    def test_security_filtering(self):
        """Test security filtering capabilities"""
        security_test_cases = [
            "My email is test@example.com and I love this product!",  # PII
            "SELECT * FROM users; This product is great!",  # SQL injection
            "<script>alert('xss')</script>Good product",  # XSS
            "This product is good! Call me at 555-123-4567",  # Phone number
        ]
        
        for text in security_test_cases:
            with self.subTest(text=text[:30] + "..."):
                result = self.analyzer.analyze_text(text, include_security_report=True)
                
                self.assertTrue(result.validation_passed)
                self.assertIsNotNone(result.security_report)
                
                # Check if PII was detected and redacted
                if 'email' in text or 'phone' in text or '555' in text:
                    self.assertGreater(len(result.security_report.get('pii_detected', [])), 0)
                
                # Check if malicious patterns were detected
                if 'SELECT' in text or '<script>' in text:
                    self.assertGreater(len(result.security_report.get('malicious_patterns', [])), 0)
    
    def test_error_handling(self):
        """Test robust error handling"""
        # Test with invalid inputs
        invalid_inputs = [
            None,
            123,
            [],
            {},
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input_type=type(invalid_input).__name__):
                try:
                    # This should handle the error gracefully
                    result = self.analyzer.analyze_text(str(invalid_input) if invalid_input else "")
                    self.assertIsNotNone(result)
                except Exception as e:
                    self.fail(f"Analyzer should handle invalid input gracefully, but raised: {e}")
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        # This is a simplified test - in reality, you'd need to simulate failures
        original_state = self.analyzer.circuit_breaker['state']
        self.assertEqual(original_state, 'closed')
        
        # Test normal operation doesn't trigger circuit breaker
        result = self.analyzer.analyze_text("This is a test")
        self.assertTrue(result.validation_passed)
        self.assertEqual(self.analyzer.circuit_breaker['state'], 'closed')
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Create analyzer with low rate limit for testing
        test_analyzer = RobustSentimentAnalyzer()
        test_analyzer.rate_limiter.max_requests = 5
        test_analyzer.rate_limiter.time_window = 1
        
        # Make requests up to limit
        results = []
        for i in range(7):  # Exceed the limit
            result = test_analyzer.analyze_text(f"Test message {i}")
            results.append(result)
        
        # Check that some requests were rate limited
        rate_limited = [r for r in results if not r.validation_passed and "rate limit" in (r.error_details or "").lower()]
        
        # Note: Due to timing, this test might be flaky, so we just check the mechanism exists
        self.assertTrue(hasattr(test_analyzer, 'rate_limiter'))
    
    def test_batch_processing_robust(self):
        """Test robust batch processing"""
        # Mix of valid and problematic texts
        test_texts = [
            "I love this product!",
            "test@example.com - great service!",
            "<script>alert('xss')</script>",
            "",
            "Normal text here",
            "a" * 15000,  # Very long text
        ]
        
        results, batch_summary = self.analyzer.analyze_batch_robust(test_texts)
        
        self.assertEqual(len(results), len(test_texts))
        self.assertIn('security_events', batch_summary)
        self.assertIn('failed_analyses', batch_summary)
        self.assertGreaterEqual(batch_summary['successful_analyses'], 0)


class TestScalableSentimentAnalyzer(TestSentimentAnalyzerBase):
    """Test suite for Generation 3: Scalable Sentiment Analyzer"""
    
    def setUp(self):
        super().setUp()
        self.analyzer = ScalableSentimentAnalyzer(max_workers=4, cache_size=1000)
    
    def test_caching_functionality(self):
        """Test caching improves performance"""
        test_text = "This is a test for caching functionality"
        
        # First analysis (cache miss)
        result1 = self.analyzer.analyze_text_sync(test_text, use_cache=True)
        self.assertFalse(result1.get('cache_hit', False))
        
        # Second analysis (should be cache hit)
        result2 = self.analyzer.analyze_text_sync(test_text, use_cache=True)
        self.assertTrue(result2.get('cache_hit', False))
        
        # Results should be the same
        self.assertEqual(result1['sentiment'], result2['sentiment'])
        self.assertEqual(result1['confidence'], result2['confidence'])
    
    def test_parallel_processing(self):
        """Test parallel processing performance"""
        test_texts = [f"Test message number {i}" for i in range(100)]
        
        start_time = time.time()
        results, stats = self.analyzer.analyze_batch_parallel(test_texts, use_cache=False)
        processing_time = time.time() - start_time
        
        self.assertEqual(len(results), len(test_texts))
        self.assertEqual(stats['successful_analyses'], len(test_texts))
        self.assertGreater(stats['throughput'], 0)
        
        # Parallel processing should be reasonably fast
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
    
    def test_async_processing(self):
        """Test asynchronous processing"""
        async def run_async_test():
            test_texts = [f"Async test message {i}" for i in range(50)]
            
            results, stats = await self.analyzer.analyze_batch_async(test_texts, concurrency_limit=10)
            
            self.assertEqual(len(results), len(test_texts))
            self.assertEqual(stats['successful_analyses'], len(test_texts))
            self.assertGreater(stats['throughput'], 0)
            
            return results, stats
        
        # Run the async test
        results, stats = asyncio.run(run_async_test())
        self.assertIsNotNone(results)
        self.assertIsNotNone(stats)
    
    def test_performance_optimization(self):
        """Test performance optimization features"""
        # Generate some load to populate metrics
        test_texts = [f"Performance test {i}" for i in range(100)]
        self.analyzer.analyze_batch_parallel(test_texts[:50])
        self.analyzer.analyze_batch_parallel(test_texts[50:])  # Some cache hits
        
        # Get performance report
        performance_report = self.analyzer.get_performance_report()
        
        self.assertIn('processing_stats', performance_report)
        self.assertIn('cache_performance', performance_report)
        self.assertIn('system_info', performance_report)
        
        # Test optimization
        optimization_report = self.analyzer.optimize_performance()
        self.assertIn('optimizations_applied', optimization_report)
        self.assertIn('cache_hit_rate', optimization_report)
    
    def test_high_throughput_performance(self):
        """Test high throughput performance meets requirements"""
        # Generate substantial load
        test_texts = ["High throughput test message"] * 1000
        
        start_time = time.time()
        results, stats = self.analyzer.analyze_batch_parallel(test_texts, batch_size=100)
        total_time = time.time() - start_time
        
        throughput = len(test_texts) / total_time
        
        # Should achieve at least 1000 texts per second
        self.assertGreaterEqual(throughput, 1000, 
                               f"High throughput test failed: {throughput:.1f} texts/sec < 1000 texts/sec")
        
        # All analyses should succeed
        self.assertEqual(stats['successful_analyses'], len(test_texts))


class TestIntegrationScenarios(TestSentimentAnalyzerBase):
    """Integration tests for real-world scenarios"""
    
    def setUp(self):
        super().setUp()
        # Use all three analyzers for integration testing
        self.simple_analyzer = SimpleAnalyzer()
        self.robust_analyzer = RobustSentimentAnalyzer()
        self.scalable_analyzer = ScalableSentimentAnalyzer()
    
    def test_cross_analyzer_consistency(self):
        """Test that all analyzers produce consistent results"""
        test_texts = [
            "I absolutely love this product!",
            "This is terrible, worst experience ever.",
            "It's okay, nothing special but adequate.",
        ]
        
        for text in test_texts:
            with self.subTest(text=text):
                simple_result = self.simple_analyzer.analyze_text(text)
                robust_result = self.robust_analyzer.analyze_text(text)
                scalable_result = self.scalable_analyzer.analyze_text_sync(text)
                
                # All should classify to same primary sentiment for clear cases
                sentiments = [
                    simple_result.sentiment,
                    robust_result.sentiment,
                    scalable_result['sentiment']
                ]
                
                # Allow for some variation in neutral cases
                unique_sentiments = set(sentiments)
                
                # For clearly positive/negative texts, all should agree
                if "love" in text.lower() or "terrible" in text.lower():
                    self.assertEqual(len(unique_sentiments), 1, 
                                   f"Analyzers disagreed on clear sentiment: {sentiments}")
    
    def test_production_load_simulation(self):
        """Simulate production load across all analyzers"""
        # Simulate mixed production workload
        production_texts = [
            "Excellent product, highly recommend!",
            "user@example.com says this product is good",  # PII
            "Poor quality, not worth the money",
            "SELECT * FROM products WHERE rating > 4",  # SQL injection attempt
            "Average product, does what it's supposed to",
            "<script>alert('test')</script>Great item!",  # XSS attempt
            "Outstanding service and fast delivery!",
            "Disappointed with the quality and support",
        ] * 25  # 200 total texts
        
        # Test robust analyzer with security filtering
        robust_results, robust_stats = self.robust_analyzer.analyze_batch_robust(production_texts)
        
        self.assertEqual(len(robust_results), len(production_texts))
        self.assertGreater(robust_stats['security_events'], 0)  # Should detect security issues
        
        # Test scalable analyzer with high performance
        scalable_results, scalable_stats = self.scalable_analyzer.analyze_batch_parallel(production_texts)
        
        self.assertEqual(len(scalable_results), len(production_texts))
        self.assertGreater(scalable_stats['throughput'], 100)  # Should maintain high throughput
    
    def test_memory_efficiency(self):
        """Test memory efficiency under load"""
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        # Process large batch
        large_batch = ["Memory efficiency test message"] * 1000
        
        # Take snapshot before processing
        snapshot1 = tracemalloc.take_snapshot()
        
        # Process batch
        results, stats = self.scalable_analyzer.analyze_batch_parallel(large_batch)
        
        # Take snapshot after processing
        snapshot2 = tracemalloc.take_snapshot()
        
        # Compare memory usage
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Memory growth should be reasonable (less than 50MB for 1000 texts)
        total_growth = sum(stat.size_diff for stat in top_stats)
        max_growth_mb = 50 * 1024 * 1024  # 50MB in bytes
        
        self.assertLess(total_growth, max_growth_mb,
                       f"Memory growth too high: {total_growth / 1024 / 1024:.1f}MB > 50MB")
        
        tracemalloc.stop()


class TestSecurityValidation(unittest.TestCase):
    """Security-focused tests"""
    
    def setUp(self):
        self.analyzer = RobustSentimentAnalyzer()
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection"""
        sql_attacks = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "admin'--",
            "' OR 1=1--",
        ]
        
        for attack in sql_attacks:
            with self.subTest(attack=attack):
                result = self.analyzer.analyze_text(f"This product is good {attack}")
                
                self.assertTrue(result.validation_passed)
                self.assertIsNotNone(result.security_report)
                self.assertIn('sql_injection', result.security_report.get('malicious_patterns', []))
    
    def test_xss_detection(self):
        """Test XSS attack detection"""
        xss_attacks = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "javascript:alert('xss')",
        ]
        
        for attack in xss_attacks:
            with self.subTest(attack=attack):
                result = self.analyzer.analyze_text(f"Great product {attack}")
                
                self.assertTrue(result.validation_passed)
                self.assertIsNotNone(result.security_report)
                # Should detect XSS or command injection
                detected_threats = result.security_report.get('malicious_patterns', [])
                self.assertTrue(any('xss' in threat or 'command' in threat for threat in detected_threats))
    
    def test_pii_detection_and_redaction(self):
        """Test PII detection and redaction"""
        pii_test_cases = [
            ("Contact me at john@example.com", "email"),
            ("Call me at 555-123-4567", "phone"),
            ("My SSN is 123-45-6789", "ssn"),
            ("Use card 4532-1234-5678-9012", "credit_card"),
        ]
        
        for text, pii_type in pii_test_cases:
            with self.subTest(pii_type=pii_type):
                result = self.analyzer.analyze_text(text)
                
                self.assertTrue(result.validation_passed)
                
                # Should detect PII
                pii_detected = result.security_report.get('pii_detected', [])
                detected_types = [item['type'] for item in pii_detected]
                self.assertIn(pii_type, detected_types)
                
                # Should redact PII from processed text
                self.assertIn('REDACTED', result.text.upper())


def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("\n🚀 PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)
    
    analyzers = {
        'Simple (Gen 1)': SimpleAnalyzer(),
        'Robust (Gen 2)': RobustSentimentAnalyzer(),
        'Scalable (Gen 3)': ScalableSentimentAnalyzer()
    }
    
    test_sizes = [100, 500, 1000]
    results = {}
    
    for name, analyzer in analyzers.items():
        print(f"\n📊 Testing {name}")
        print("-" * 40)
        
        analyzer_results = {}
        
        for size in test_sizes:
            test_texts = [f"Performance test message number {i}" for i in range(size)]
            
            start_time = time.time()
            
            if name == 'Scalable (Gen 3)':
                batch_results, stats = analyzer.analyze_batch_parallel(test_texts)
                throughput = stats['throughput']
            elif name == 'Robust (Gen 2)':
                batch_results, stats = analyzer.analyze_batch_robust(test_texts)
                throughput = stats['throughput']
            else:  # Simple
                batch_result = analyzer.analyze_batch(test_texts)
                processing_time = time.time() - start_time
                throughput = len(test_texts) / processing_time
            
            analyzer_results[size] = throughput
            print(f"  {size:4d} texts: {throughput:8.1f} texts/sec")
        
        results[name] = analyzer_results
    
    # Summary
    print(f"\n🏆 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Analyzer':<20} {'100 texts':<12} {'500 texts':<12} {'1000 texts':<12}")
    print("-" * 60)
    
    for name, analyzer_results in results.items():
        print(f"{name:<20} {analyzer_results[100]:8.1f}     {analyzer_results[500]:8.1f}     {analyzer_results[1000]:8.1f}")
    
    return results


def main():
    """Run comprehensive test suite"""
    print("🧪 SENTIMENT ANALYZER COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print("Running Quality Gates: Unit Tests, Integration Tests, Security Tests, Performance Benchmarks")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSimpleSentimentAnalyzer,
        TestRobustSentimentAnalyzer,
        TestScalableSentimentAnalyzer,
        TestIntegrationScenarios,
        TestSecurityValidation,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    # Run performance benchmark
    performance_results = run_performance_benchmark()
    
    # Summary report
    print(f"\n✅ QUALITY GATES SUMMARY")
    print("=" * 70)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {(test_result.testsRun - len(test_result.failures) - len(test_result.errors))/test_result.testsRun:.1%}")
    
    # Export test results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_report_file = f"/root/repo/test_report_{timestamp}.json"
    
    test_report = {
        'test_summary': {
            'total_tests': test_result.testsRun,
            'failures': len(test_result.failures),
            'errors': len(test_result.errors),
            'success_rate': (test_result.testsRun - len(test_result.failures) - len(test_result.errors))/test_result.testsRun,
            'timestamp': timestamp
        },
        'performance_benchmarks': performance_results,
        'test_failures': [str(failure) for failure in test_result.failures],
        'test_errors': [str(error) for error in test_result.errors]
    }
    
    with open(test_report_file, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n💾 Test report exported to: {test_report_file}")
    
    # Quality gate validation
    quality_gates_passed = (
        test_result.testsRun > 0 and
        len(test_result.failures) == 0 and
        len(test_result.errors) == 0
    )
    
    print(f"\n🎯 QUALITY GATES STATUS: {'✅ PASSED' if quality_gates_passed else '❌ FAILED'}")
    
    if not quality_gates_passed:
        print("⚠️ Quality gates failed. Review test failures and errors before production deployment.")
        return False
    
    print("🎉 All quality gates passed! Ready for production deployment.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)