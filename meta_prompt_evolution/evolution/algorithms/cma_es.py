"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

import random
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import math

from .base import EvolutionAlgorithm, AlgorithmConfig
from ..population import PromptPopulation, Prompt


@dataclass
class CMAESConfig(AlgorithmConfig):
    """Configuration for CMA-ES algorithm."""
    lambda_: int = None  # Population size (offspring)
    mu: int = None  # Number of parents
    sigma: float = 0.3  # Initial step size
    dimension: int = 50  # Dimensionality of search space
    
    def __post_init__(self):
        if self.lambda_ is None:
            self.lambda_ = 4 + int(3 * math.log(self.dimension))
        if self.mu is None:
            self.mu = self.lambda_ // 2


class CMAES(EvolutionAlgorithm):
    """CMA-ES algorithm for continuous parameter optimization of prompts."""
    
    def __init__(self, config: CMAESConfig):
        """Initialize CMA-ES with configuration."""
        super().__init__(config)
        self.config: CMAESConfig = config
        
        # CMA-ES parameters
        self.dim = config.dimension
        self.lambda_ = config.lambda_
        self.mu = config.mu
        self.sigma = config.sigma
        
        # Selection weights
        self.weights = np.array([math.log(self.mu + 0.5) - math.log(i + 1) 
                                for i in range(self.mu)])
        self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        
        # Adaptation parameters
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / 
                      ((self.dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Dynamic parameters
        self.mean = np.zeros(self.dim)
        self.C = np.eye(self.dim)  # Covariance matrix
        self.ps = np.zeros(self.dim)  # Evolution path for sigma
        self.pc = np.zeros(self.dim)  # Evolution path for C
        self.B = np.eye(self.dim)  # Eigenvectors
        self.D = np.ones(self.dim)  # Eigenvalues^0.5
        
        # Prompt template vocabulary for parameter mapping
        self.vocabulary = self._build_vocabulary()
        
    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary for prompt parameter mapping."""
        return [
            # Task instructions
            "analyze", "create", "explain", "solve", "help", "write", "generate",
            "describe", "summarize", "classify", "predict", "optimize", "design",
            
            # Modifiers
            "carefully", "quickly", "thoroughly", "precisely", "clearly", "simply",
            "detailed", "brief", "comprehensive", "specific", "general", "creative",
            
            # Context words
            "please", "kindly", "ensure", "make sure", "remember", "note",
            "important", "crucial", "key", "essential", "significant", "relevant",
            
            # Connectors
            "and", "but", "however", "therefore", "additionally", "furthermore",
            "meanwhile", "consequently", "similarly", "alternatively", "specifically",
            
            # Formatting
            "step-by-step", "bullet points", "numbered list", "paragraph", "outline",
            "summary", "conclusion", "introduction", "example", "illustration",
            
            # Quality indicators
            "accurate", "correct", "precise", "exact", "appropriate", "suitable",
            "effective", "efficient", "optimal", "best", "improved", "enhanced"
        ]
    
    def evolve_generation(
        self, 
        population: PromptPopulation, 
        fitness_fn: Callable[[Prompt], Dict[str, float]]
    ) -> PromptPopulation:
        """Evolve one generation using CMA-ES procedure."""
        # Generate offspring
        offspring = []
        parameter_vectors = []
        
        for _ in range(self.lambda_):
            # Sample from multivariate normal distribution
            z = np.random.standard_normal(self.dim)
            y = self.B @ (self.D * z)
            x = self.mean + self.sigma * y
            parameter_vectors.append(x)
            
            # Convert parameter vector to prompt
            prompt = self._parameters_to_prompt(x)
            prompt.fitness_scores = fitness_fn(prompt)
            offspring.append(prompt)
        
        # Sort offspring by fitness
        fitness_values = [p.fitness_scores.get('fitness', 0) for p in offspring]
        sorted_indices = np.argsort(fitness_values)[::-1]  # Descending order
        
        # Select best mu individuals
        selected_indices = sorted_indices[:self.mu]
        selected_parameters = [parameter_vectors[i] for i in selected_indices]
        selected_offspring = [offspring[i] for i in selected_indices]
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum([self.weights[i] * selected_parameters[i] 
                           for i in range(self.mu)], axis=0)
        
        # Update evolution paths
        mean_diff = self.mean - old_mean
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * \
                 self.B @ (self.D ** -1 * (self.B.T @ mean_diff)) / self.sigma
        
        # Update pc
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation + 2)) / 
                math.sqrt(self.dim) < 1.4 + 2 / (self.dim + 1))
        
        self.pc = (1 - self.cc) * self.pc + \
                 hsig * math.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * mean_diff / self.sigma
        
        # Update covariance matrix
        artmp = np.array([(selected_parameters[i] - old_mean) / self.sigma for i in range(self.mu)])
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                self.c1 * (np.outer(self.pc, self.pc) + 
                          (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                self.cmu * np.sum([self.weights[i] * np.outer(artmp[i], artmp[i]) 
                                  for i in range(self.mu)], axis=0)
        
        # Update step size
        self.sigma *= math.exp((self.cs / self.damps) * 
                              (np.linalg.norm(self.ps) / math.sqrt(self.dim) - 1))
        
        # Eigendecomposition (every few generations for efficiency)
        if self.generation % 10 == 0:
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(eigenvalues, 1e-14))
            self.B = eigenvectors
        
        return PromptPopulation(selected_offspring)
    
    def _parameters_to_prompt(self, parameters: np.ndarray) -> Prompt:
        """Convert parameter vector to prompt text."""
        # Normalize parameters to [0, 1]
        normalized_params = 1 / (1 + np.exp(-parameters))  # Sigmoid
        
        # Select words based on parameters
        selected_words = []
        vocab_size = len(self.vocabulary)
        
        # Use parameters to select words with some structure
        for i in range(0, min(len(normalized_params), 20), 2):  # Limit prompt length
            if i + 1 < len(normalized_params):
                word_idx = int(normalized_params[i] * vocab_size) % vocab_size
                include_prob = normalized_params[i + 1]
                
                if include_prob > 0.3:  # Threshold for word inclusion
                    selected_words.append(self.vocabulary[word_idx])
        
        # Ensure minimum prompt structure
        if not selected_words:
            selected_words = ["Please", "help", "with", "the", "task"]
        
        # Add basic structure
        if not any(word in selected_words for word in ["please", "you", "help", "assist"]):
            selected_words.insert(0, "Please")
        
        if not selected_words[-1].endswith(('.', '?', '!')):
            selected_words.append(".")
        
        prompt_text = " ".join(selected_words)
        
        return Prompt(
            text=prompt_text,
            generation=self.generation + 1
        )
    
    def _prompt_to_parameters(self, prompt: Prompt) -> np.ndarray:
        """Convert prompt text to parameter vector (for initialization)."""
        words = prompt.text.lower().split()
        parameters = np.zeros(self.dim)
        
        for i, word in enumerate(words[:self.dim//2]):
            if word in self.vocabulary:
                vocab_idx = self.vocabulary.index(word)
                parameters[i*2] = (vocab_idx / len(self.vocabulary)) * 6 - 3  # Scale to [-3, 3]
                parameters[i*2 + 1] = 1.0  # High inclusion probability
        
        # Fill remaining with small random values
        for i in range(len(words)*2, self.dim):
            parameters[i] = np.random.normal(0, 0.1)
        
        return parameters
    
    def initialize_from_population(self, population: PromptPopulation):
        """Initialize CMA-ES parameters from existing population."""
        if not population.prompts:
            return
        
        # Convert prompts to parameter vectors
        param_vectors = []
        for prompt in population.prompts[:self.mu]:
            param_vector = self._prompt_to_parameters(prompt)
            param_vectors.append(param_vector)
        
        if param_vectors:
            # Initialize mean as average of parameter vectors
            self.mean = np.mean(param_vectors, axis=0)
            
            # Initialize covariance based on parameter variance
            if len(param_vectors) > 1:
                param_matrix = np.array(param_vectors)
                self.C = np.cov(param_matrix.T)
                
                # Ensure positive definite
                eigenvalues, eigenvectors = np.linalg.eigh(self.C)
                eigenvalues = np.maximum(eigenvalues, 1e-8)
                self.C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                
                self.D = np.sqrt(eigenvalues)
                self.B = eigenvectors
    
    def selection(self, population: PromptPopulation, k: int) -> List[Prompt]:
        """Select k best prompts based on fitness."""
        sorted_prompts = sorted(
            population.prompts,
            key=lambda p: p.fitness_scores.get('fitness', 0) if p.fitness_scores else 0,
            reverse=True
        )
        return sorted_prompts[:k]
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get convergence metrics for monitoring."""
        return {
            "sigma": self.sigma,
            "condition_number": np.max(self.D) / np.min(self.D) if np.min(self.D) > 0 else float('inf'),
            "axis_ratio": np.max(self.D) / np.min(self.D) if np.min(self.D) > 0 else float('inf'),
            "mean_norm": np.linalg.norm(self.mean),
            "ps_norm": np.linalg.norm(self.ps),
            "pc_norm": np.linalg.norm(self.pc)
        }