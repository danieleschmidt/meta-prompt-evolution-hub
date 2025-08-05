#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - Scalable Evolution Hub
High-performance, scalable evolution hub with advanced optimizations.
"""

import asyncio
import time
import multiprocessing as mp
from typing import List, Dict, Any, Callable, Optional
import logging

from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase, FitnessFunction
from meta_prompt_evolution.evolution.hub import EvolutionConfig

from robust_evolution_hub import RobustEvolutionHub
from optimization_engine import PerformanceOptimizer, OptimizationConfig, AsyncOptimizer
from caching_system import evaluation_cache, population_cache, distributed_cache
from monitoring_system import performance_tracker, EvolutionMetrics

class ScalableEvolutionHub(RobustEvolutionHub):
    """High-performance, scalable evolution hub with comprehensive optimizations."""
    
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fitness_function: Optional[FitnessFunction] = None,
        enable_all_optimizations: bool = True
    ):
        """Initialize scalable evolution hub."""
        
        # Initialize parent with robustness features
        super().__init__(
            config=config,
            fitness_function=fitness_function,
            enable_monitoring=enable_all_optimizations,
            enable_validation=enable_all_optimizations
        )
        
        # Initialize optimization components
        self.optimization_config = optimization_config or OptimizationConfig()
        self.performance_optimizer = PerformanceOptimizer(self.optimization_config)
        self.async_optimizer = AsyncOptimizer(self.optimization_config)
        
        # Scaling metrics
        self.scaling_metrics = {
            "populations_processed": 0,
            "cache_hits": 0,
            "parallel_executions": 0,
            "optimization_time_saved": 0.0,
            "memory_optimizations": 0,
            "async_operations": 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.ScalableEvolutionHub")
        self.logger.info("Scalable Evolution Hub initialized with all optimizations")
        
    def evolve(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ) -> PromptPopulation:
        """High-performance evolution with all optimizations enabled."""
        
        evolution_start = time.time()
        self.logger.info(f"Starting scalable evolution: {len(population)} prompts, {len(test_cases)} test cases")
        
        try:
            # Pre-evolution optimizations
            optimized_population = self._pre_evolution_optimization(population, test_cases)
            
            # Check for cached populations with similar characteristics
            cached_result = self._check_population_cache(optimized_population)
            if cached_result:
                self.logger.info("Using cached population result")
                self.scaling_metrics["cache_hits"] += 1
                return cached_result
            
            # Run optimized evolution
            result = self._run_optimized_evolution(optimized_population, test_cases, termination_criteria)
            
            # Post-evolution optimizations
            final_result = self._post_evolution_optimization(result)
            
            # Update scaling metrics
            self._update_scaling_metrics(evolution_start, final_result)
            
            self.logger.info(f"Scalable evolution completed in {time.time() - evolution_start:.2f}s")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Scalable evolution failed: {e}")
            # Fallback to robust evolution
            return super().evolve(population, test_cases, termination_criteria)
            
    async def evolve_async(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ) -> PromptPopulation:
        """Asynchronous high-performance evolution."""
        
        self.logger.info("Starting async scalable evolution")
        self.scaling_metrics["async_operations"] += 1
        
        # Async optimization
        optimized_population = await self.async_optimizer.async_evaluate_population(
            population, test_cases, self.fitness_function.evaluate
        )
        
        # Run evolution generations asynchronously
        current_population = optimized_population
        
        for generation in range(self.config.generations):
            # Evolve generation
            next_population = self.algorithm.evolve_generation(
                current_population,
                lambda prompt: self.fitness_function.evaluate(prompt, test_cases)
            )
            
            # Async evaluation of new population
            current_population = await self.async_optimizer.async_evaluate_population(
                next_population, test_cases, self.fitness_function.evaluate
            )
            current_population.generation = generation + 1
            
            # Check termination criteria
            if termination_criteria and termination_criteria(current_population):
                break
                
        return current_population
    
    def _pre_evolution_optimization(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase]
    ) -> PromptPopulation:
        """Apply pre-evolution optimizations."""
        
        # Memory optimization
        if self.optimization_config.enable_memory_optimization:
            self.performance_optimizer.resource_pool.memory_monitor.optimize_memory()
            
        # Population deduplication
        unique_prompts = self._deduplicate_population(population)
        
        # Smart population sizing based on system resources
        optimal_size = self._calculate_optimal_population_size()
        if len(unique_prompts) > optimal_size:
            # Keep most diverse prompts
            unique_prompts = self._select_diverse_subset(unique_prompts, optimal_size)
            
        return PromptPopulation(unique_prompts)
    
    def _run_optimized_evolution(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        termination_criteria: Optional[Callable[[PromptPopulation], bool]] = None
    ) -> PromptPopulation:
        """Run evolution with performance optimizations."""
        
        current_population = population
        
        for generation in range(self.config.generations):
            generation_start = time.time()
            
            # Optimized evaluation
            current_population = self.performance_optimizer.optimize_population_evaluation(
                current_population, test_cases, self.fitness_function.evaluate
            )
            
            # Evolution step
            next_population = self.algorithm.evolve_generation(
                current_population,
                lambda prompt: prompt.fitness_scores or {"fitness": 0.0}
            )
            
            # Update population
            current_population = next_population
            current_population.generation = generation + 1
            
            # Dynamic optimization based on performance
            self._dynamic_optimization_adjustment(generation, time.time() - generation_start)
            
            # Check termination
            if termination_criteria and termination_criteria(current_population):
                break
                
        return current_population
    
    def _post_evolution_optimization(self, population: PromptPopulation) -> PromptPopulation:
        """Apply post-evolution optimizations."""
        
        # Cache best results
        if self.optimization_config.enable_caching:
            best_prompts = population.get_top_k(10)
            for prompt in best_prompts:
                if prompt.fitness_scores and prompt.fitness_scores.get("fitness", 0) > 0.5:
                    # Cache high-quality prompts for future use
                    cache_key = f"best_prompt_{hash(prompt.text)}"
                    distributed_cache.put(cache_key, prompt.fitness_scores, ttl=3600)
                    
        # Final memory cleanup
        self.performance_optimizer.resource_pool.memory_monitor.optimize_memory(force=True)
        
        return population
    
    def _deduplicate_population(self, population: PromptPopulation) -> List[Prompt]:
        """Remove duplicate prompts efficiently."""
        seen_texts = set()
        unique_prompts = []
        
        for prompt in population.prompts:
            text_hash = hash(prompt.text.lower().strip())
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_prompts.append(prompt)
                
        self.logger.info(f"Deduplicated population: {len(population)} -> {len(unique_prompts)}")
        return unique_prompts
    
    def _calculate_optimal_population_size(self) -> int:
        """Calculate optimal population size based on system resources."""
        memory_info = self.performance_optimizer.resource_pool.memory_monitor.check_memory_usage()
        
        base_size = self.config.population_size
        
        # Adjust based on memory usage
        if memory_info["usage_percent"] > 80:
            return max(10, base_size // 2)
        elif memory_info["usage_percent"] < 30:
            return min(500, base_size * 2)
        else:
            return base_size
    
    def _select_diverse_subset(self, prompts: List[Prompt], target_size: int) -> List[Prompt]:
        """Select diverse subset of prompts."""
        if len(prompts) <= target_size:
            return prompts
            
        # Simple diversity-based selection
        selected = []
        remaining = prompts.copy()
        
        # Always include first prompt
        selected.append(remaining.pop(0))
        
        while len(selected) < target_size and remaining:
            # Find most diverse prompt from remaining
            best_prompt = None
            max_diversity = -1
            
            for candidate in remaining:
                diversity = self._calculate_prompt_diversity(candidate, selected)
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_prompt = candidate
                    
            if best_prompt:
                selected.append(best_prompt)
                remaining.remove(best_prompt)
            else:
                break
                
        return selected
    
    def _calculate_prompt_diversity(self, candidate: Prompt, existing: List[Prompt]) -> float:
        """Calculate diversity of candidate prompt relative to existing set."""
        if not existing:
            return 1.0
            
        candidate_words = set(candidate.text.lower().split())
        
        min_similarity = float('inf')
        for existing_prompt in existing:
            existing_words = set(existing_prompt.text.lower().split())
            
            union = candidate_words.union(existing_words)
            intersection = candidate_words.intersection(existing_words)
            
            similarity = len(intersection) / len(union) if union else 0
            min_similarity = min(min_similarity, similarity)
            
        return 1.0 - min_similarity
    
    def _check_population_cache(self, population: PromptPopulation) -> Optional[PromptPopulation]:
        """Check for cached population with similar characteristics."""
        # Simple cache check based on population characteristics
        pop_signature = self._calculate_population_signature(population)
        
        cache_key = f"population_{pop_signature}"
        cached_data = distributed_cache.get(cache_key)
        
        if cached_data:
            # Reconstruct population from cached data
            cached_prompts = []
            for prompt_data in cached_data.get("prompts", []):
                prompt = Prompt(
                    text=prompt_data["text"],
                    fitness_scores=prompt_data["fitness_scores"],
                    generation=prompt_data.get("generation", 0)
                )
                cached_prompts.append(prompt)
                
            return PromptPopulation(cached_prompts)
            
        return None
    
    def _calculate_population_signature(self, population: PromptPopulation) -> str:
        """Calculate signature for population caching."""
        # Simple signature based on prompt lengths and algorithm
        lengths = sorted([len(p.text) for p in population.prompts])
        signature_data = {
            "lengths": lengths[:10],  # First 10 lengths
            "size": len(population),
            "algorithm": self.config.algorithm
        }
        return str(hash(str(signature_data)))
    
    def _dynamic_optimization_adjustment(self, generation: int, generation_time: float):
        """Dynamically adjust optimization settings based on performance."""
        
        # Adjust batch size based on generation time
        if generation_time > 10.0:  # Slow generation
            self.optimization_config.batch_size = max(10, self.optimization_config.batch_size // 2)
            self.logger.info(f"Reduced batch size to {self.optimization_config.batch_size}")
        elif generation_time < 2.0:  # Fast generation
            self.optimization_config.batch_size = min(100, self.optimization_config.batch_size * 2)
            self.logger.info(f"Increased batch size to {self.optimization_config.batch_size}")
            
        # Memory optimization frequency adjustment
        if generation % 5 == 0:
            self.performance_optimizer.resource_pool.memory_monitor.optimize_memory()
            self.scaling_metrics["memory_optimizations"] += 1
    
    def _update_scaling_metrics(self, start_time: float, result: PromptPopulation):
        """Update scaling performance metrics."""
        duration = time.time() - start_time
        
        self.scaling_metrics["populations_processed"] += 1
        
        # Record evolution metrics for monitoring
        best_prompt = max(result.prompts, key=lambda p: p.fitness_scores.get('fitness', 0))
        
        metrics = EvolutionMetrics(
            active_populations=1,
            total_evaluations=len(result) * self.config.generations,
            average_fitness=sum(p.fitness_scores.get('fitness', 0) for p in result) / len(result),
            best_fitness=best_prompt.fitness_scores.get('fitness', 0),
            diversity_score=self.performance_optimizer._calculate_diversity(result),
            generation_time=duration,
            algorithm_type=f"scalable_{self.config.algorithm}"
        )
        
        performance_tracker.record_evolution_metrics(metrics)
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling metrics."""
        base_metrics = self.get_comprehensive_status()
        optimization_metrics = self.performance_optimizer.get_optimization_metrics()
        
        scaling_report = {
            "scaling_metrics": self.scaling_metrics,
            "optimization_metrics": optimization_metrics,
            "cache_performance": {
                "evaluation_cache": evaluation_cache.get_cache_stats(),
                "distributed_cache": distributed_cache.get_combined_stats()
            },
            "system_status": base_metrics
        }
        
        return scaling_report
    
    def benchmark_performance(self, test_sizes: List[int] = [10, 50, 100, 200]) -> Dict[str, Any]:
        """Benchmark performance at different scales."""
        
        self.logger.info("Starting performance benchmark")
        benchmark_results = {}
        
        for size in test_sizes:
            self.logger.info(f"Benchmarking population size: {size}")
            
            # Create test population
            test_population = PromptPopulation.from_seeds([
                f"Test prompt {i}: Help me with task {i}" for i in range(size)
            ])
            
            test_cases = [
                TestCase(f"input_{i}", f"output_{i}", weight=1.0) for i in range(3)
            ]
            
            # Benchmark evolution
            start_time = time.time()
            result = self.evolve(test_population, test_cases)
            duration = time.time() - start_time
            
            benchmark_results[f"size_{size}"] = {
                "duration": duration,
                "prompts_per_second": size / duration,
                "best_fitness": max(p.fitness_scores.get('fitness', 0) for p in result.prompts),
                "memory_usage": self.performance_optimizer.resource_pool.memory_monitor.check_memory_usage()
            }
            
        self.logger.info("Performance benchmark completed")
        return benchmark_results
    
    def shutdown(self):
        """Shutdown with comprehensive cleanup."""
        self.logger.info("Shutting down scalable evolution hub")
        
        # Shutdown optimization components
        self.performance_optimizer.shutdown()
        
        # Export final metrics
        final_metrics = self.get_scaling_metrics()
        
        import json
        with open('/root/repo/scalable_evolution_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
            
        # Call parent shutdown
        super().shutdown()
        
        self.logger.info("Scalable evolution hub shutdown complete")

# Factory functions for easy instantiation
def create_scalable_hub(
    population_size: int = 100,
    generations: int = 20,
    algorithm: str = "nsga2",
    max_workers: int = None,
    enable_all_optimizations: bool = True
) -> ScalableEvolutionHub:
    """Create a fully optimized scalable evolution hub."""
    
    evolution_config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        algorithm=algorithm,
        evaluation_parallel=True
    )
    
    optimization_config = OptimizationConfig(
        max_workers=max_workers or mp.cpu_count(),
        batch_size=min(50, population_size // 2),
        enable_caching=enable_all_optimizations,
        enable_parallel_evaluation=enable_all_optimizations,
        enable_memory_optimization=enable_all_optimizations
    )
    
    return ScalableEvolutionHub(
        config=evolution_config,
        optimization_config=optimization_config,
        enable_all_optimizations=enable_all_optimizations
    )

def create_high_throughput_hub() -> ScalableEvolutionHub:
    """Create hub optimized for high-throughput processing."""
    return create_scalable_hub(
        population_size=500,
        generations=50,
        algorithm="nsga2",
        max_workers=mp.cpu_count() * 2,
        enable_all_optimizations=True
    )