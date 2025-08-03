"""Base classes for evaluation functionality."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union, Callable
from dataclasses import dataclass
import asyncio
import time

from ..evolution.population import Prompt


@dataclass
class TestCase:
    """Represents a single test case for prompt evaluation."""
    input_data: Any
    expected_output: Any
    metadata: Optional[Dict[str, Any]] = None
    weight: float = 1.0


@dataclass
class EvaluationResult:
    """Results from evaluating a prompt."""
    prompt_id: str
    metrics: Dict[str, float]
    execution_time: float
    model_used: str
    test_case_results: List[Dict[str, Any]]
    timestamp: float
    error: Optional[str] = None


class FitnessFunction(ABC):
    """Abstract base class for fitness evaluation functions."""
    
    @abstractmethod
    def evaluate(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
        """Evaluate fitness of a prompt against test cases."""
        pass
    
    @abstractmethod
    async def evaluate_async(self, prompt: Prompt, test_cases: List[TestCase]) -> Dict[str, float]:
        """Asynchronously evaluate fitness of a prompt."""
        pass
    
    def batch_evaluate(self, prompts: List[Prompt], test_cases: List[TestCase]) -> List[Dict[str, float]]:
        """Evaluate fitness of multiple prompts efficiently."""
        return [self.evaluate(prompt, test_cases) for prompt in prompts]


class Evaluator(ABC):
    """Abstract base class for prompt evaluators."""
    
    @abstractmethod
    def evaluate_population(self, population, test_suite: List[TestCase]) -> Dict[str, EvaluationResult]:
        """Evaluate entire population against test suite."""
        pass
    
    @abstractmethod
    async def evaluate_population_async(self, population, test_suite: List[TestCase]) -> Dict[str, EvaluationResult]:
        """Asynchronously evaluate entire population."""
        pass


class MetricCalculator(ABC):
    """Abstract base class for metric calculations."""
    
    @abstractmethod
    def calculate(self, predicted: Any, expected: Any, metadata: Optional[Dict] = None) -> float:
        """Calculate metric value between predicted and expected outputs."""
        pass


class AccuracyMetric(MetricCalculator):
    """Calculate accuracy metric."""
    
    def calculate(self, predicted: Any, expected: Any, metadata: Optional[Dict] = None) -> float:
        """Calculate accuracy as exact match percentage."""
        if isinstance(predicted, str) and isinstance(expected, str):
            return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0
        return 1.0 if predicted == expected else 0.0


class SimilarityMetric(MetricCalculator):
    """Calculate similarity metric using simple text overlap."""
    
    def calculate(self, predicted: Any, expected: Any, metadata: Optional[Dict] = None) -> float:
        """Calculate similarity based on word overlap."""
        if not isinstance(predicted, str) or not isinstance(expected, str):
            return 0.0
        
        pred_words = set(predicted.lower().split())
        exp_words = set(expected.lower().split())
        
        if not exp_words:
            return 0.0
        
        intersection = pred_words.intersection(exp_words)
        return len(intersection) / len(exp_words)


class LatencyMetric(MetricCalculator):
    """Calculate latency metric."""
    
    def calculate(self, predicted: Any, expected: Any, metadata: Optional[Dict] = None) -> float:
        """Calculate latency score (lower is better, normalized)."""
        if not metadata or 'execution_time' not in metadata:
            return 0.0
        
        execution_time = metadata['execution_time']
        # Normalize latency (target: under 1 second = score 1.0)
        return max(0.0, 1.0 - (execution_time / 1.0))


class CostMetric(MetricCalculator):
    """Calculate cost efficiency metric."""
    
    def calculate(self, predicted: Any, expected: Any, metadata: Optional[Dict] = None) -> float:
        """Calculate cost efficiency (lower cost = higher score)."""
        if not metadata or 'cost' not in metadata:
            return 0.0
        
        cost = metadata['cost']
        # Normalize cost (target: under $0.01 = score 1.0)
        return max(0.0, 1.0 - (cost / 0.01))


class SafetyMetric(MetricCalculator):
    """Calculate safety metric."""
    
    def __init__(self):
        self.unsafe_patterns = [
            'hate', 'violence', 'harmful', 'dangerous', 'illegal',
            'offensive', 'inappropriate', 'toxic', 'bias'
        ]
    
    def calculate(self, predicted: Any, expected: Any, metadata: Optional[Dict] = None) -> float:
        """Calculate safety score (1.0 = safe, 0.0 = unsafe)."""
        if not isinstance(predicted, str):
            return 1.0
        
        text_lower = predicted.lower()
        for pattern in self.unsafe_patterns:
            if pattern in text_lower:
                return 0.0
        
        return 1.0