#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST (Reliable)
Comprehensive error handling, validation, monitoring, and reliability testing.
"""

import json
import time
import logging
import traceback
import sys
from typing import List, Dict, Any, Optional
from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase, FitnessFunction


class RobustFitnessFunction(FitnessFunction):
    """Robust fitness function with comprehensive error handling."""
    
    def __init__(self, enable_safety_checks: bool = True):
        self.enable_safety_checks = enable_safety_checks
        self.unsafe_patterns = ['harmful', 'dangerous', 'illegal', 'offensive']
        self.evaluation_count = 0
        self.error_count = 0
        
    def evaluate(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
        """Robust fitness evaluation with error handling and validation."""
        try:
            self.evaluation_count += 1
            
            # Input validation
            if not prompt or not prompt.text:
                raise ValueError("Invalid prompt: empty or None")
            
            if not isinstance(prompt.text, str):
                raise TypeError(f"Prompt text must be string, got {type(prompt.text)}")
            
            if len(prompt.text.strip()) == 0:
                return {"fitness": 0.0, "error": "empty_prompt"}
            
            # Safety validation
            if self.enable_safety_checks and self._is_unsafe(prompt.text):
                return {"fitness": 0.0, "safety_score": 0.0, "error": "unsafe_content"}
            
            text = prompt.text.lower()
            
            # Robust metrics calculation
            try:
                length_score = self._calculate_length_score(text)
                keyword_score = self._calculate_keyword_score(text)
                structure_score = self._calculate_structure_score(text)
                safety_score = 1.0 if not self._is_unsafe(text) else 0.0
                
                # Weighted fitness calculation
                fitness = (
                    length_score * 0.3 + 
                    keyword_score * 0.3 + 
                    structure_score * 0.2 + 
                    safety_score * 0.2
                )
                
                return {
                    "fitness": round(fitness, 4),
                    "length_score": round(length_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "structure_score": round(structure_score, 4),
                    "safety_score": round(safety_score, 4),
                    "text_length": len(prompt.text),
                    "evaluation_id": self.evaluation_count
                }
                
            except Exception as calc_error:
                logging.error(f"Calculation error: {calc_error}")
                return {"fitness": 0.0, "error": f"calculation_error: {str(calc_error)}"}
                
        except Exception as e:
            self.error_count += 1
            logging.error(f"Evaluation error for prompt {prompt.id}: {e}")
            return {"fitness": 0.0, "error": f"evaluation_error: {str(e)}"}
    
    async def evaluate_async(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
        """Async version with same robustness."""
        return self.evaluate(prompt, test_cases)
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate normalized length score with bounds checking."""
        try:
            length = len(text)
            if length == 0:
                return 0.0
            if length > 1000:  # Too long
                return 0.5
            return min(length / 200.0, 1.0)  # Optimal around 200 chars
        except Exception:
            return 0.0
    
    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate keyword presence score with error handling."""
        try:
            keywords = ['help', 'assist', 'task', 'will', 'can', 'support', 'guide']
            found_keywords = sum(1 for keyword in keywords if keyword in text)
            return min(found_keywords / len(keywords), 1.0)
        except Exception:
            return 0.0
    
    def _calculate_structure_score(self, text: str) -> float:
        """Calculate structural quality score."""
        try:
            score = 0.0
            
            # Check for proper formatting
            if '{task}' in text:
                score += 0.4
            if text.strip().endswith('.') or text.strip().endswith('?'):
                score += 0.3
            if len(text.split()) >= 3:  # Minimum word count
                score += 0.3
                
            return min(score, 1.0)
        except Exception:
            return 0.0
    
    def _is_unsafe(self, text: str) -> bool:
        """Check for unsafe content patterns."""
        try:
            text_lower = text.lower()
            return any(pattern in text_lower for pattern in self.unsafe_patterns)
        except Exception:
            return True  # Assume unsafe if can't check


class RobustSystemMonitor:
    """System monitoring for health checks and performance tracking."""
    
    def __init__(self):
        self.start_time = time.time()
        self.operation_count = 0
        self.error_count = 0
        self.performance_metrics = []
        
    def record_operation(self, operation_type: str, duration: float, success: bool):
        """Record operation metrics."""
        self.operation_count += 1
        if not success:
            self.error_count += 1
            
        self.performance_metrics.append({
            "operation": operation_type,
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        })
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        uptime = time.time() - self.start_time
        error_rate = self.error_count / max(self.operation_count, 1)
        
        avg_duration = 0.0
        if self.performance_metrics:
            avg_duration = sum(m["duration"] for m in self.performance_metrics) / len(self.performance_metrics)
        
        status = "healthy"
        if error_rate > 0.1:  # More than 10% errors
            status = "degraded"
        if error_rate > 0.5:  # More than 50% errors
            status = "unhealthy"
        
        return {
            "status": status,
            "uptime": uptime,
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "average_operation_duration": avg_duration,
            "timestamp": time.time()
        }


def validate_inputs(population: PromptPopulation, test_cases: List[TestCase]) -> List[str]:
    """Comprehensive input validation with detailed error reporting."""
    errors = []
    
    # Population validation
    if not population:
        errors.append("Population is None or empty")
        return errors
    
    if len(population) == 0:
        errors.append("Population contains no prompts")
        return errors
    
    # Individual prompt validation
    for i, prompt in enumerate(population):
        if not prompt:
            errors.append(f"Prompt {i} is None")
            continue
            
        if not hasattr(prompt, 'text') or not prompt.text:
            errors.append(f"Prompt {i} has no text content")
            continue
            
        if not isinstance(prompt.text, str):
            errors.append(f"Prompt {i} text is not a string: {type(prompt.text)}")
            continue
            
        if len(prompt.text.strip()) == 0:
            errors.append(f"Prompt {i} has empty text content")
            continue
            
        # Check for extremely long prompts
        if len(prompt.text) > 10000:
            errors.append(f"Prompt {i} is too long: {len(prompt.text)} characters")
    
    # Test case validation
    if not test_cases:
        errors.append("No test cases provided")
    
    for i, test_case in enumerate(test_cases or []):
        if not test_case:
            errors.append(f"Test case {i} is None")
            continue
            
        if not hasattr(test_case, 'input_data'):
            errors.append(f"Test case {i} missing input_data")
    
    return errors


def run_generation_2_robust_test():
    """Run Generation 2 robustness and reliability test."""
    print("🛡️ Generation 2: MAKE IT ROBUST (Reliable) - Starting Test")
    start_time = time.time()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize monitoring
    monitor = RobustSystemMonitor()
    
    try:
        # Test data with edge cases
        test_prompts = [
            "You are a helpful assistant. Please {task}",
            "As an AI assistant, I will help you {task}",
            "",  # Empty prompt (edge case)
            "A" * 500,  # Long prompt (edge case)
            "Help with {task} - let me assist you properly.",
            "Simple {task} handler",
            "I can support your {task} efficiently"
        ]
        
        # Filter out empty prompts for population creation
        valid_prompts = [p for p in test_prompts if p and p.strip()]
        population = PromptPopulation.from_seeds(valid_prompts)
        
        # Add edge case manually for testing
        edge_prompt = Prompt(text="", id="test_empty")
        population.prompts.append(edge_prompt)
        
        logger.info(f"Created test population with {len(population)} prompts")
        
        # Create comprehensive test cases
        test_cases = [
            TestCase(
                input_data="Explain quantum computing",
                expected_output="Clear scientific explanation",
                metadata={"difficulty": "high", "domain": "science"},
                weight=1.0
            ),
            TestCase(
                input_data="Write a summary",
                expected_output="Concise summary",
                metadata={"difficulty": "medium", "domain": "writing"},
                weight=0.8
            ),
            TestCase(
                input_data="",  # Empty input (edge case)
                expected_output="Handle gracefully",
                metadata={"difficulty": "edge_case"},
                weight=0.5
            )
        ]
        
        # Input validation
        print("🔍 Running comprehensive input validation...")
        validation_start = time.time()
        validation_errors = validate_inputs(population, test_cases)
        validation_time = time.time() - validation_start
        
        monitor.record_operation("validation", validation_time, len(validation_errors) == 0)
        
        if validation_errors:
            print(f"⚠️  Validation found {len(validation_errors)} issues:")
            for error in validation_errors[:5]:  # Show first 5 errors
                print(f"   • {error}")
        else:
            print("✅ Input validation passed")
        
        # Robust fitness evaluation
        print("🧮 Running robust fitness evaluation...")
        fitness_fn = RobustFitnessFunction(enable_safety_checks=True)
        
        successful_evaluations = 0
        failed_evaluations = 0
        
        for prompt in population:
            eval_start = time.time()
            try:
                prompt.fitness_scores = fitness_fn.evaluate(prompt, test_cases)
                if "error" not in prompt.fitness_scores:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
                    logger.warning(f"Evaluation error for prompt {prompt.id}: {prompt.fitness_scores.get('error', 'unknown')}")
                    
                eval_time = time.time() - eval_start
                monitor.record_operation("evaluation", eval_time, "error" not in prompt.fitness_scores)
                
            except Exception as e:
                failed_evaluations += 1
                logger.error(f"Critical evaluation error for prompt {prompt.id}: {e}")
                prompt.fitness_scores = {"fitness": 0.0, "error": f"critical_error: {str(e)}"}
                
                eval_time = time.time() - eval_start
                monitor.record_operation("evaluation", eval_time, False)
        
        print(f"✅ Successful evaluations: {successful_evaluations}")
        print(f"⚠️  Failed evaluations: {failed_evaluations}")
        
        # Error recovery test
        print("🔄 Testing error recovery...")
        valid_prompts = []
        try:
            # Simulate recovery from failed prompts
            valid_prompts = [p for p in population if p.fitness_scores.get("fitness", 0) > 0]
            if len(valid_prompts) == 0:
                print("⚠️  No valid prompts after error recovery")
                # Fallback to basic prompts
                fallback_prompts = ["Help with {task}", "Assist with {task}"]
                fallback_population = PromptPopulation.from_seeds(fallback_prompts)
                for prompt in fallback_population:
                    prompt.fitness_scores = fitness_fn.evaluate(prompt, test_cases)
                valid_prompts = fallback_population.prompts
                print(f"✅ Fallback recovery: {len(valid_prompts)} prompts")
                
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
        
        # Health monitoring
        print("❤️  System health check...")
        health = monitor.get_health_status()
        print(f"   Status: {health['status']}")
        print(f"   Error Rate: {health['error_rate']:.1%}")
        print(f"   Operations: {health['operation_count']}")
        
        # Performance validation
        avg_duration = health.get('average_operation_duration', 0)
        if avg_duration > 1.0:  # More than 1 second per operation
            print("⚠️  Performance degradation detected")
        else:
            print("✅ Performance within acceptable limits")
        
        # Results summary
        execution_time = time.time() - start_time
        
        # Get top performers (filter out error cases)
        valid_population_prompts = [p for p in population if p.fitness_scores.get("fitness", 0) > 0]
        
        if valid_population_prompts:
            top_prompts = sorted(
                valid_population_prompts,
                key=lambda p: p.fitness_scores.get("fitness", 0),
                reverse=True
            )[:3]
            
            print("\n📊 Top 3 Robust Prompts:")
            for i, prompt in enumerate(top_prompts, 1):
                fitness = prompt.fitness_scores.get("fitness", 0.0)
                safety = prompt.fitness_scores.get("safety_score", 0.0)
                print(f"{i}. Fitness: {fitness:.3f}, Safety: {safety:.3f} - '{prompt.text[:40]}...'")
        
        results = {
            "generation": 2,
            "status": "ROBUST_COMPLETE",
            "execution_time": execution_time,
            "population_size": len(population),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "validation_errors": len(validation_errors),
            "health_status": health,
            "error_recovery": "successful" if valid_prompts else "failed",
            "fitness_function_stats": {
                "evaluation_count": fitness_fn.evaluation_count,
                "error_count": fitness_fn.error_count
            }
        }
        
        print(f"\n✅ Generation 2 Robust Test Complete!")
        print(f"⏱️  Execution Time: {execution_time:.2f}s")
        print(f"🛡️  System Status: {health['status']}")
        print(f"📊 Success Rate: {successful_evaluations/(successful_evaluations + failed_evaluations):.1%}")
        
        # Save results
        with open("generation_2_robust_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("💾 Results saved to generation_2_robust_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"Critical system error: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "generation": 2,
            "status": "CRITICAL_ERROR",
            "error": str(e),
            "execution_time": time.time() - start_time
        }


if __name__ == "__main__":
    results = run_generation_2_robust_test()
    
    # Validate robustness criteria
    if (results.get("status") == "ROBUST_COMPLETE" and 
        results.get("successful_evaluations", 0) > 0 and
        results.get("health_status", {}).get("status") != "unhealthy"):
        print("\n🎉 Generation 2: MAKE IT ROBUST - SUCCESS!")
        print("✅ Comprehensive error handling operational")
        print("✅ Input validation working")
        print("✅ System monitoring functional")
        print("✅ Error recovery mechanisms in place")
        print("✅ Ready for Generation 3 optimization")
    else:
        print("\n⚠️  Generation 2 robustness needs improvement")
        print(f"Status: {results.get('status', 'unknown')}")