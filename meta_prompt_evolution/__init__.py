"""
Meta-Prompt-Evolution-Hub: Evolutionary Prompt Optimization at Scale

Scale-tested evolutionary prompt search (EPS) platform hosting tens of thousands 
of prompts with evaluation scores. Integrates with Eval-Genius and 
Async-Toolformer-Orchestrator for continuous A/B testing and prompt optimization at scale.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core components that work without heavy dependencies
from .evolution.population import PromptPopulation, Prompt

# Always import core components
try:
    from .evolution.hub import EvolutionHub
    from .deployment.ab_testing import ABTestOrchestrator  
    from .evaluation.evaluator import DistributedEvaluator
    
    __all__ = [
        "Prompt",
        "PromptPopulation", 
        "EvolutionHub",
        "ABTestOrchestrator",
        "DistributedEvaluator",
    ]
    
except ImportError as e:
    print(f"Warning: Some features may be limited: {e}")
    
    __all__ = [
        "Prompt",
        "PromptPopulation",
    ]