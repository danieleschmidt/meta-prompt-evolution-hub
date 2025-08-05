"""Distributed and async evaluators for prompt assessment."""

from typing import List, Dict, Any, Optional, Callable
import asyncio
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .base import (
    Evaluator, FitnessFunction, TestCase, EvaluationResult,
    AccuracyMetric, SimilarityMetric, LatencyMetric, CostMetric, SafetyMetric
)
from ..evolution.population import Prompt, PromptPopulation


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    timeout_seconds: float = 30.0
    max_retries: int = 3
    batch_size: int = 10
    parallel_workers: int = 4
    use_caching: bool = True
    cache_ttl_seconds: int = 3600


class MockLLMProvider:
    """Mock LLM provider for testing and demonstration."""
    
    def __init__(self, model_name: str = "mock-gpt-4", latency_ms: float = 200):
        self.model_name = model_name
        self.latency_ms = latency_ms
        self.call_count = 0
    
    async def generate_async(self, prompt: str, input_data: Any) -> str:
        """Generate response asynchronously."""
        self.call_count += 1
        await asyncio.sleep(self.latency_ms / 1000)  # Simulate latency
        
        # Mock response based on prompt content
        if "summarize" in prompt.lower():
            return f"Summary of {input_data}: Key points include main concepts and conclusions."
        elif "classify" in prompt.lower():
            return random.choice(["Category A", "Category B", "Category C"])
        elif "explain" in prompt.lower():
            return f"Explanation: {input_data} can be understood through systematic analysis."
        else:
            return f"Response to '{input_data}' using prompt strategy from {prompt[:30]}..."
    
    def generate(self, prompt: str, input_data: Any) -> str:
        """Generate response synchronously."""
        self.call_count += 1
        time.sleep(self.latency_ms / 1000)  # Simulate latency
        
        # Mock response based on prompt content
        if "summarize" in prompt.lower():
            return f"Summary of {input_data}: Key points include main concepts and conclusions."
        elif "classify" in prompt.lower():
            return random.choice(["Category A", "Category B", "Category C"])
        elif "explain" in prompt.lower():
            return f"Explanation: {input_data} can be understood through systematic analysis."
        else:
            return f"Response to '{input_data}' using prompt strategy from {prompt[:30]}..."


class ComprehensiveFitnessFunction(FitnessFunction):
    """Multi-metric fitness function with real evaluation logic."""
    
    def __init__(
        self,
        llm_provider: Optional[MockLLMProvider] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Initialize with LLM provider and metric weights."""
        self.llm_provider = llm_provider or MockLLMProvider()
        self.metrics = metrics or {
            "accuracy": 0.4,
            "similarity": 0.2,
            "latency": 0.2,
            "safety": 0.2
        }
        
        self.calculators = {
            "accuracy": AccuracyMetric(),
            "similarity": SimilarityMetric(),
            "latency": LatencyMetric(),
            "cost": CostMetric(),
            "safety": SafetyMetric()
        }
    
    def evaluate(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
        """Evaluate prompt against test cases."""
        if not test_cases:
            return {"fitness": 0.0}
        
        start_time = time.time()
        total_scores = {metric: 0.0 for metric in self.metrics}
        total_weight = 0.0
        
        for test_case in test_cases:
            try:
                # Generate response using prompt
                case_start = time.time()
                response = self.llm_provider.generate(prompt.text, test_case.input_data)
                execution_time = time.time() - case_start
                
                # Calculate metrics
                metadata = {
                    "execution_time": execution_time,
                    "cost": execution_time * 0.001,  # Mock cost calculation
                }
                
                for metric_name, weight in self.metrics.items():
                    if metric_name in self.calculators:
                        score = self.calculators[metric_name].calculate(
                            response, test_case.expected_output, metadata
                        )
                        total_scores[metric_name] += score * test_case.weight * weight
                
                total_weight += test_case.weight
                
            except Exception as e:
                logging.warning(f"Evaluation error for prompt {prompt.id}: {e}")
                continue
        
        # Normalize scores
        if total_weight > 0:
            for metric in total_scores:
                total_scores[metric] /= total_weight
        
        # Calculate overall fitness
        fitness = sum(total_scores.values())
        total_scores["fitness"] = fitness
        total_scores["evaluation_time"] = time.time() - start_time
        
        return total_scores
    
    async def evaluate_async(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
        """Asynchronously evaluate prompt against test cases."""
        if not test_cases:
            return {"fitness": 0.0}
        
        start_time = time.time()
        total_scores = {metric: 0.0 for metric in self.metrics}
        total_weight = 0.0
        
        # Process test cases concurrently
        tasks = []
        for test_case in test_cases:
            task = self._evaluate_single_case_async(prompt, test_case)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result, test_case in zip(results, test_cases):
            if isinstance(result, Exception):
                logging.warning(f"Async evaluation error: {result}")
                continue
            
            case_scores, execution_time = result
            for metric_name, weight in self.metrics.items():
                if metric_name in case_scores:
                    total_scores[metric_name] += case_scores[metric_name] * test_case.weight * weight
            
            total_weight += test_case.weight
        
        # Normalize scores
        if total_weight > 0:
            for metric in total_scores:
                total_scores[metric] /= total_weight
        
        # Calculate overall fitness
        fitness = sum(total_scores.values())
        total_scores["fitness"] = fitness
        total_scores["evaluation_time"] = time.time() - start_time
        
        return total_scores
    
    async def _evaluate_single_case_async(self, prompt: Prompt, test_case: TestCase):
        """Evaluate a single test case asynchronously."""
        case_start = time.time()
        response = await self.llm_provider.generate_async(prompt.text, test_case.input_data)
        execution_time = time.time() - case_start
        
        metadata = {
            "execution_time": execution_time,
            "cost": execution_time * 0.001,
        }
        
        case_scores = {}
        for metric_name in self.metrics:
            if metric_name in self.calculators:
                score = self.calculators[metric_name].calculate(
                    response, test_case.expected_output, metadata
                )
                case_scores[metric_name] = score
        
        return case_scores, execution_time


class DistributedEvaluator(Evaluator):
    """Distributed evaluator using Ray or ThreadPoolExecutor."""
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        fitness_function: Optional[FitnessFunction] = None
    ):
        """Initialize distributed evaluator."""
        self.config = config or EvaluationConfig()
        self.fitness_function = fitness_function or ComprehensiveFitnessFunction()
        self.use_ray = RAY_AVAILABLE and ray.is_initialized()
        
        if not self.use_ray:
            self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
    
    def evaluate_population(
        self, 
        population: PromptPopulation, 
        test_suite: List[TestCase]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate entire population in parallel."""
        if self.use_ray:
            return self._evaluate_with_ray(population, test_suite)
        else:
            return self._evaluate_with_threads(population, test_suite)
    
    def _evaluate_with_threads(
        self, 
        population: PromptPopulation, 
        test_suite: List[TestCase]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate using ThreadPoolExecutor."""
        results = {}
        
        # Submit evaluation tasks
        future_to_prompt = {}
        for prompt in population:
            future = self.executor.submit(self._evaluate_prompt, prompt, test_suite)
            future_to_prompt[future] = prompt
        
        # Collect results
        for future in as_completed(future_to_prompt, timeout=self.config.timeout_seconds):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                results[prompt.id] = result
            except Exception as e:
                logging.error(f"Evaluation failed for prompt {prompt.id}: {e}")
                results[prompt.id] = EvaluationResult(
                    prompt_id=prompt.id,
                    metrics={"fitness": 0.0},
                    execution_time=0.0,
                    model_used="unknown",
                    test_case_results=[],
                    timestamp=time.time(),
                    error=str(e)
                )
        
        return results
    
    def _evaluate_prompt(self, prompt: Prompt, test_suite: List[TestCase]) -> EvaluationResult:
        """Evaluate a single prompt."""
        start_time = time.time()
        
        try:
            metrics = self.fitness_function.evaluate(prompt, test_suite)
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                prompt_id=prompt.id,
                metrics=metrics,
                execution_time=execution_time,
                model_used=getattr(self.fitness_function.llm_provider, 'model_name', 'unknown'),
                test_case_results=[],  # Could be populated with detailed results
                timestamp=time.time(),
                error=None
            )
        
        except Exception as e:
            return EvaluationResult(
                prompt_id=prompt.id,
                metrics={"fitness": 0.0},
                execution_time=time.time() - start_time,
                model_used="unknown",
                test_case_results=[],
                timestamp=time.time(),
                error=str(e)
            )
    
    async def evaluate_population_async(
        self, 
        population: PromptPopulation, 
        test_suite: List[TestCase]
    ) -> Dict[str, EvaluationResult]:
        """Asynchronously evaluate entire population."""
        tasks = []
        for prompt in population:
            task = self._evaluate_prompt_async(prompt, test_suite)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        evaluation_results = {}
        for prompt, result in zip(population, results):
            if isinstance(result, Exception):
                evaluation_results[prompt.id] = EvaluationResult(
                    prompt_id=prompt.id,
                    metrics={"fitness": 0.0},
                    execution_time=0.0,
                    model_used="unknown",
                    test_case_results=[],
                    timestamp=time.time(),
                    error=str(result)
                )
            else:
                evaluation_results[prompt.id] = result
        
        return evaluation_results
    
    async def _evaluate_prompt_async(
        self, 
        prompt: Prompt, 
        test_suite: List[TestCase]
    ) -> EvaluationResult:
        """Asynchronously evaluate a single prompt."""
        start_time = time.time()
        
        try:
            metrics = await self.fitness_function.evaluate_async(prompt, test_suite)
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                prompt_id=prompt.id,
                metrics=metrics,
                execution_time=execution_time,
                model_used=getattr(self.fitness_function.llm_provider, 'model_name', 'unknown'),
                test_case_results=[],
                timestamp=time.time(),
                error=None
            )
        
        except Exception as e:
            return EvaluationResult(
                prompt_id=prompt.id,
                metrics={"fitness": 0.0},
                execution_time=time.time() - start_time,
                model_used="unknown",
                test_case_results=[],
                timestamp=time.time(),
                error=str(e)
            )


class EfficientEvaluator:
    """Evaluator that uses surrogate models to reduce evaluation costs."""
    
    def __init__(
        self,
        fast_fitness_fn: FitnessFunction,
        accurate_fitness_fn: FitnessFunction,
        confidence_threshold: float = 0.9,
        surrogate_ratio: float = 0.8
    ):
        """Initialize efficient evaluator with fast and accurate functions."""
        self.fast_fitness_fn = fast_fitness_fn
        self.accurate_fitness_fn = accurate_fitness_fn
        self.confidence_threshold = confidence_threshold
        self.surrogate_ratio = surrogate_ratio
        self.evaluation_history = {}
    
    def evaluate_with_surrogate(
        self, 
        prompt: Prompt, 
        test_cases: List[TestCase]
    ) -> Dict[str, float]:
        """Evaluate using surrogate model when appropriate."""
        # Quick surrogate evaluation
        surrogate_scores = self.fast_fitness_fn.evaluate(prompt, test_cases)
        confidence = self._calculate_confidence(prompt, surrogate_scores)
        
        # Use accurate evaluation if confidence is low
        if confidence < self.confidence_threshold:
            accurate_scores = self.accurate_fitness_fn.evaluate(prompt, test_cases)
            self._update_history(prompt, surrogate_scores, accurate_scores)
            return accurate_scores
        
        return surrogate_scores
    
    def _calculate_confidence(self, prompt: Prompt, scores: Dict[str, float]) -> float:
        """Calculate confidence in surrogate evaluation."""
        # Simple confidence based on score consistency and prompt similarity
        if prompt.id in self.evaluation_history:
            historical_score = self.evaluation_history[prompt.id]["surrogate_fitness"]
            current_score = scores.get("fitness", 0.0)
            consistency = 1.0 - abs(historical_score - current_score)
            return min(consistency, 1.0)
        
        # Default confidence for new prompts
        return 0.7
    
    def _update_history(
        self, 
        prompt: Prompt, 
        surrogate_scores: Dict[str, float], 
        accurate_scores: Dict[str, float]
    ):
        """Update evaluation history for learning."""
        self.evaluation_history[prompt.id] = {
            "surrogate_fitness": surrogate_scores.get("fitness", 0.0),
            "accurate_fitness": accurate_scores.get("fitness", 0.0),
            "error": abs(surrogate_scores.get("fitness", 0.0) - accurate_scores.get("fitness", 0.0))
        }