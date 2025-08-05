#!/usr/bin/env python3
"""
Generation 2: ROBUST EVOLUTION HUB
Enhanced evolution hub with comprehensive error handling, validation, and monitoring.
"""

from typing import List, Dict, Any, Callable, Optional
import time
import logging
from contextlib import contextmanager

from meta_prompt_evolution.evolution.hub import EvolutionHub, EvolutionConfig
from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase, FitnessFunction
from meta_prompt_evolution.evaluation.evaluator import DistributedEvaluator, ComprehensiveFitnessFunction

from error_handling import error_handler, ErrorMetrics
from validation_system import prompt_validator, test_validator
from monitoring_system import health_checker, performance_tracker, EvolutionMetrics

class RobustEvolutionHub(EvolutionHub):
    """Enhanced Evolution Hub with robustness features."""
    
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_function: Optional[FitnessFunction] = None,
        evaluator: Optional[DistributedEvaluator] = None,
        enable_monitoring: bool = True,
        enable_validation: bool = True,
        max_retries: int = 3
    ):
        """Initialize robust evolution hub."""
        super().__init__(config, fitness_function, evaluator)
        
        self.enable_monitoring = enable_monitoring
        self.enable_validation = enable_validation
        self.max_retries = max_retries
        
        # Enhanced logging
        self.logger = logging.getLogger(f"{__name__}.RobustEvolutionHub")
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            health_checker.start_monitoring()
            self.logger.info("Health monitoring started")
            
    def evolve(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ) -> PromptPopulation:
        """Robust evolution with validation, error handling, and monitoring."""
        
        with error_handler.error_context("robust_evolution"):
            # Pre-evolution validation
            if self.enable_validation:
                self._validate_inputs(population, test_cases)
                
            # Record evolution start
            evolution_start = time.time()
            
            try:
                # Run evolution with retry logic
                result = self._evolve_with_retries(population, test_cases, termination_criteria)
                
                # Record successful evolution metrics
                if self.enable_monitoring:
                    self._record_evolution_success(result, time.time() - evolution_start)
                    
                return result
                
            except Exception as e:
                self.logger.error(f"Evolution failed after {self.max_retries} retries: {e}")
                
                # Record failure metrics
                if self.enable_monitoring:
                    self._record_evolution_failure(e, time.time() - evolution_start)
                    
                # Return best effort result
                return self._create_fallback_population(population)
                
    def _validate_inputs(self, population: PromptPopulation, test_cases: List[TestCase]):
        """Comprehensive input validation."""
        self.logger.info("Validating inputs...")
        
        # Validate population
        population_results = prompt_validator.validate_population(population)
        invalid_prompts = [pid for pid, result in population_results.items() if not result.is_valid]
        
        if invalid_prompts:
            self.logger.warning(f"Found {len(invalid_prompts)} invalid prompts")
            # Filter out invalid prompts
            valid_prompts = [p for p in population.prompts if p.id not in invalid_prompts]
            population.prompts = valid_prompts
            
        if len(population.prompts) == 0:
            raise ValueError("No valid prompts in population after validation")
            
        # Validate test cases
        test_results = test_validator.validate_test_suite(test_cases)
        invalid_tests = [i for i, result in test_results.items() if not result.is_valid]
        
        if invalid_tests:
            self.logger.warning(f"Found {len(invalid_tests)} invalid test cases")
            
        if len(test_cases) - len(invalid_tests) == 0:
            raise ValueError("No valid test cases after validation")
            
        self.logger.info(f"Validation complete: {len(population)} prompts, {len(test_cases)} test cases")
        
    def _evolve_with_retries(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ) -> PromptPopulation:
        """Evolution with automatic retry on failure."""
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Evolution attempt {attempt + 1}/{self.max_retries}")
                
                # Check system health before proceeding
                if self.enable_monitoring:
                    health_report = health_checker.get_health_report()
                    if health_report["status"] == "critical":
                        self.logger.warning("System in critical state, reducing workload")
                        # Reduce population size for this attempt
                        reduced_population = PromptPopulation(population.prompts[:len(population)//2])
                        result = super().evolve(reduced_population, test_cases, termination_criteria)
                    else:
                        result = super().evolve(population, test_cases, termination_criteria)
                else:
                    result = super().evolve(population, test_cases, termination_criteria)
                    
                self.logger.info(f"Evolution completed successfully on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Evolution attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry, increasing delay each time
                    retry_delay = 2 ** attempt
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    
        # All retries failed
        raise last_exception
        
    def _record_evolution_success(self, result: PromptPopulation, duration: float):
        """Record successful evolution metrics."""
        best_prompt = self._get_best_prompt(result)
        diversity = self._calculate_diversity(result)
        
        metrics = EvolutionMetrics(
            active_populations=1,
            total_evaluations=len(result) * self.config.generations,
            average_fitness=sum(p.fitness_scores.get('fitness', 0) for p in result) / len(result),
            best_fitness=best_prompt.fitness_scores.get('fitness', 0),
            diversity_score=diversity,
            generation_time=duration,
            algorithm_type=self.config.algorithm
        )
        
        performance_tracker.record_evolution_metrics(metrics)
        self.logger.info(f"Evolution metrics recorded: {metrics.best_fitness:.3f} fitness, {duration:.2f}s")
        
    def _record_evolution_failure(self, error: Exception, duration: float):
        """Record failed evolution metrics."""
        metrics = EvolutionMetrics(
            active_populations=0,
            total_evaluations=0,
            average_fitness=0.0,
            best_fitness=0.0,
            diversity_score=0.0,
            generation_time=duration,
            algorithm_type=self.config.algorithm
        )
        
        performance_tracker.record_evolution_metrics(metrics)
        self.logger.error(f"Evolution failure recorded: {str(error)}")
        
    def _create_fallback_population(self, original_population: PromptPopulation) -> PromptPopulation:
        """Create fallback population when evolution completely fails."""
        self.logger.info("Creating fallback population")
        
        # Return original population with zero fitness scores
        for prompt in original_population:
            if prompt.fitness_scores is None:
                prompt.fitness_scores = {
                    'fitness': 0.0,
                    'accuracy': 0.0,
                    'safety': 1.0  # Assume safe if we can't evaluate
                }
                
        return original_population
        
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        base_stats = self.get_evolution_statistics()
        
        status = {
            "evolution_stats": base_stats,
            "error_summary": error_handler.get_error_summary(),
            "performance_summary": performance_tracker.get_performance_summary()
        }
        
        if self.enable_monitoring:
            status["health_report"] = health_checker.get_health_report()
            
        return status
        
    def export_comprehensive_logs(self, filename_prefix: str = "robust_evolution"):
        """Export all system logs and metrics."""
        timestamp = int(time.time())
        
        # Export performance metrics
        performance_tracker.export_metrics(f"{filename_prefix}_performance_{timestamp}.json")
        
        # Export comprehensive status
        import json
        status_file = f"{filename_prefix}_status_{timestamp}.json"
        with open(status_file, 'w') as f:
            json.dump(self.get_comprehensive_status(), f, indent=2)
            
        self.logger.info(f"Exported logs: {status_file}")
        
    def shutdown(self):
        """Graceful shutdown with cleanup."""
        self.logger.info("Shutting down robust evolution hub...")
        
        if self.enable_monitoring:
            health_checker.stop_monitoring()
            
        # Export final logs
        self.export_comprehensive_logs("shutdown")
        
        self.logger.info("Robust evolution hub shutdown complete")

# Factory function for easy instantiation
def create_robust_hub(
    population_size: int = 20,
    generations: int = 10,
    algorithm: str = "nsga2",
    enable_all_features: bool = True
) -> RobustEvolutionHub:
    """Create a fully configured robust evolution hub."""
    
    config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        algorithm=algorithm,
        mutation_rate=0.1,
        crossover_rate=0.7,
        evaluation_parallel=True
    )
    
    return RobustEvolutionHub(
        config=config,
        enable_monitoring=enable_all_features,
        enable_validation=enable_all_features,
        max_retries=3
    )