"""Evolution module for genetic algorithms and prompt optimization."""

# Core components that work without dependencies
from .population import PromptPopulation, Prompt

__all__ = [
    "PromptPopulation",
    "Prompt",
]

# Try to import components requiring dependencies
try:
    from .hub import EvolutionHub
    from .algorithms import NSGA2, MAPElites, CMAES
    
    __all__.extend([
        "EvolutionHub",
        "NSGA2", 
        "CMAES",
        "MAPElites",
    ])
    
except ImportError:
    # Algorithms require numpy and other dependencies
    pass