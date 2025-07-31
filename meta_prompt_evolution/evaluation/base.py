"""Base classes for evaluation functionality."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict


class FitnessFunction(ABC):
    """Abstract base class for fitness evaluation functions."""
    
    @abstractmethod
    def evaluate(self, prompt: str, test_cases: List[Any]) -> float:
        """Evaluate fitness of a prompt against test cases."""
        pass
    
    @abstractmethod
    def batch_evaluate(self, prompts: List[str], test_cases: List[Any]) -> List[float]:
        """Evaluate fitness of multiple prompts efficiently."""
        pass


class Evaluator(ABC):
    """Abstract base class for prompt evaluators."""
    
    @abstractmethod
    def evaluate_population(self, population, test_suite) -> Dict[str, float]:
        """Evaluate entire population against test suite."""
        pass