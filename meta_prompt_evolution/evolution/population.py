"""Prompt population management for evolutionary algorithms."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import random


@dataclass
class Prompt:
    """Individual prompt in the population."""
    text: str
    fitness_scores: Optional[Dict[str, float]] = None
    generation: int = 0
    parent_ids: Optional[List[str]] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"prompt_{random.randint(100000, 999999)}"


class PromptPopulation:
    """Container for managing populations of prompts."""
    
    def __init__(self, prompts: List[Prompt]):
        """Initialize population with list of prompts."""
        self.prompts = prompts
        self.generation = 0
    
    @classmethod
    def from_seeds(cls, seed_prompts: List[str]) -> "PromptPopulation":
        """Create initial population from seed prompt strings."""
        prompts = [Prompt(text=seed) for seed in seed_prompts]
        return cls(prompts)
    
    def get_top_k(self, k: int, metric: str = "fitness") -> List[Prompt]:
        """Get top k prompts by specified metric."""
        if not self.prompts:
            return []
        
        # Sort by fitness score (placeholder implementation)
        sorted_prompts = sorted(
            self.prompts,
            key=lambda p: p.fitness_scores.get(metric, 0) if p.fitness_scores else 0,
            reverse=True
        )
        return sorted_prompts[:k]
    
    def inject_prompts(self, new_prompts: List[Prompt]):
        """Inject new prompts into the population."""
        self.prompts.extend(new_prompts)
    
    def size(self) -> int:
        """Get population size."""
        return len(self.prompts)
    
    def __iter__(self):
        """Iterate over prompts in population."""
        return iter(self.prompts)
    
    def __len__(self):
        """Get population size."""
        return len(self.prompts)