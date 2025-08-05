#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - Performance Optimization Engine
Advanced performance optimizations, parallel processing, and resource pooling.
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import time
import threading
from queue import Queue, Empty
import resource
import gc

from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase
from caching_system import evaluation_cache, population_cache

@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    max_workers: int = mp.cpu_count()
    batch_size: int = 50
    enable_caching: bool = True
    enable_parallel_evaluation: bool = True
    enable_memory_optimization: bool = True
    gc_frequency: int = 100  # Garbage collection frequency
    memory_limit_mb: int = 1024  # Memory limit in MB

class ResourcePool:
    """Manages reusable resources for performance optimization."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="evolution_worker"
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(config.max_workers, mp.cpu_count())
        )
        self.memory_monitor = MemoryMonitor(config.memory_limit_mb)
        
    def shutdown(self):
        """Shutdown all resource pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class MemoryMonitor:
    """Monitors and optimizes memory usage."""
    
    def __init__(self, limit_mb: int):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.gc_counter = 0
        
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = usage.ru_maxrss / 1024  # Convert to MB (on Linux)
            
            return {
                "memory_mb": memory_mb,
                "limit_mb": self.limit_bytes / (1024 * 1024),
                "usage_percent": (memory_mb * 1024 * 1024) / self.limit_bytes * 100,
                "gc_collections": self.gc_counter
            }
        except:
            return {"memory_mb": 0, "limit_mb": 0, "usage_percent": 0, "gc_collections": 0}
    
    def optimize_memory(self, force: bool = False) -> bool:
        """Perform memory optimization."""
        memory_info = self.check_memory_usage()
        
        if force or memory_info["usage_percent"] > 80:
            # Force garbage collection
            collected = gc.collect()
            self.gc_counter += 1
            
            # Clear evaluation cache if memory is still high
            if memory_info["usage_percent"] > 90:
                evaluation_cache.cache.clear()
                
            return True
        return False

class BatchProcessor:
    """Processes evaluation batches efficiently."""
    
    def __init__(self, config: OptimizationConfig, resource_pool: ResourcePool):
        self.config = config
        self.resource_pool = resource_pool
        
    def process_population_batch(
        self,
        prompts: List[Prompt],
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> List[Dict[str, float]]:
        """Process a batch of prompts efficiently."""
        
        if not prompts:
            return []
            
        # Check cache first if enabled
        cached_results = []
        uncached_prompts = []
        
        if self.config.enable_caching:
            test_inputs = [str(tc.input_data) for tc in test_cases]
            
            for prompt in prompts:
                cached_result = evaluation_cache.get_evaluation_result(
                    prompt.text, test_inputs
                )
                if cached_result:
                    cached_results.append((prompt, cached_result))
                else:
                    uncached_prompts.append(prompt)
        else:
            uncached_prompts = prompts
            
        # Process uncached prompts
        if uncached_prompts:
            if self.config.enable_parallel_evaluation and len(uncached_prompts) > 1:
                new_results = self._parallel_evaluate(uncached_prompts, test_cases, fitness_fn)
            else:
                new_results = self._sequential_evaluate(uncached_prompts, test_cases, fitness_fn)
        else:
            new_results = []
            
        # Combine results
        all_results = []
        
        # Add cached results
        for prompt, result in cached_results:
            all_results.append(result)
            
        # Add new results and cache them
        for prompt, result in zip(uncached_prompts, new_results):
            all_results.append(result)
            
            if self.config.enable_caching:
                test_inputs = [str(tc.input_data) for tc in test_cases]
                evaluation_cache.cache_evaluation_result(prompt.text, test_inputs, result)
                
        return all_results
    
    def _parallel_evaluate(
        self,
        prompts: List[Prompt],
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> List[Dict[str, float]]:
        """Evaluate prompts in parallel."""
        
        futures = []
        for prompt in prompts:
            future = self.resource_pool.thread_pool.submit(
                self._safe_evaluate_single, prompt, test_cases, fitness_fn
            )
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Fallback result for failed evaluations
                results.append({"fitness": 0.0, "error": str(e)})
                
        return results
    
    def _sequential_evaluate(
        self,
        prompts: List[Prompt],
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> List[Dict[str, float]]:
        """Evaluate prompts sequentially."""
        results = []
        for prompt in prompts:
            result = self._safe_evaluate_single(prompt, test_cases, fitness_fn)
            results.append(result)
        return results
    
    def _safe_evaluate_single(
        self,
        prompt: Prompt,
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> Dict[str, float]:
        """Safely evaluate a single prompt with error handling."""
        try:
            return fitness_fn(prompt, test_cases)
        except Exception as e:
            return {"fitness": 0.0, "error": str(e)}

class PerformanceOptimizer:
    """Coordinates all performance optimizations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.resource_pool = ResourcePool(self.config)
        self.batch_processor = BatchProcessor(self.config, self.resource_pool)
        self.optimization_metrics = {
            "total_evaluations": 0,
            "cached_evaluations": 0,
            "parallel_batches": 0,
            "memory_optimizations": 0,
            "total_time": 0.0
        }
        
    def optimize_population_evaluation(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> PromptPopulation:
        """Optimize evaluation of entire population."""
        
        start_time = time.time()
        
        # Memory optimization check
        if self.config.enable_memory_optimization:
            if self.resource_pool.memory_monitor.optimize_memory():
                self.optimization_metrics["memory_optimizations"] += 1
        
        # Split population into batches
        batches = self._create_batches(population.prompts, self.config.batch_size)
        
        # Process batches
        all_results = []
        for batch in batches:
            batch_results = self.batch_processor.process_population_batch(
                batch, test_cases, fitness_fn
            )
            all_results.extend(batch_results)
            
            if len(batch) > 1:
                self.optimization_metrics["parallel_batches"] += 1
        
        # Update prompts with results
        for prompt, result in zip(population.prompts, all_results):
            prompt.fitness_scores = result
            
        # Update metrics
        self.optimization_metrics["total_evaluations"] += len(population)
        self.optimization_metrics["total_time"] += time.time() - start_time
        
        # Cache population if significant improvement
        if self.config.enable_caching:
            best_fitness = max(p.fitness_scores.get("fitness", 0) for p in population.prompts)
            if best_fitness > 0.5:  # Threshold for caching
                self._cache_population(population, best_fitness)
                
        return population
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Split items into batches."""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches
    
    def _cache_population(self, population: PromptPopulation, best_fitness: float):
        """Cache population data."""
        population_data = {
            "prompts": [
                {
                    "text": p.text,
                    "fitness_scores": p.fitness_scores,
                    "generation": p.generation
                }
                for p in population.prompts
            ],
            "best_fitness": best_fitness,
            "diversity": self._calculate_diversity(population),
            "population_size": len(population),
            "timestamp": time.time()
        }
        
        population_cache.cache_population(
            "optimized", population.generation, population_data
        )
    
    def _calculate_diversity(self, population: PromptPopulation) -> float:
        """Calculate population diversity efficiently."""
        if len(population) < 2:
            return 0.0
        
        # Sample-based diversity calculation for large populations
        sample_size = min(50, len(population))
        sample = population.prompts[:sample_size]
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                # Simple word-based distance
                words1 = set(sample[i].text.lower().split())
                words2 = set(sample[j].text.lower().split())
                
                union = words1.union(words2)
                intersection = words1.intersection(words2)
                
                if union:
                    distance = 1.0 - (len(intersection) / len(union))
                    total_distance += distance
                    comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics."""
        cache_stats = evaluation_cache.get_cache_stats()
        memory_info = self.resource_pool.memory_monitor.check_memory_usage()
        
        return {
            "optimization_metrics": self.optimization_metrics,
            "cache_stats": cache_stats,
            "memory_info": memory_info,
            "efficiency_ratios": {
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                "parallel_batch_ratio": (
                    self.optimization_metrics["parallel_batches"] / 
                    max(1, self.optimization_metrics["total_evaluations"] // self.config.batch_size)
                ),
                "avg_evaluation_time": (
                    self.optimization_metrics["total_time"] / 
                    max(1, self.optimization_metrics["total_evaluations"])
                )
            }
        }
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        self.resource_pool.shutdown()

# Async optimization for high-throughput scenarios
class AsyncOptimizer:
    """Asynchronous optimization for high-throughput processing."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_workers)
        
    async def async_evaluate_population(
        self,
        population: PromptPopulation,
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> PromptPopulation:
        """Asynchronously evaluate population."""
        
        tasks = []
        for prompt in population.prompts:
            task = self._async_evaluate_single(prompt, test_cases, fitness_fn)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update prompts with results
        for prompt, result in zip(population.prompts, results):
            if isinstance(result, dict):
                prompt.fitness_scores = result
            else:
                prompt.fitness_scores = {"fitness": 0.0, "error": str(result)}
                
        return population
    
    async def _async_evaluate_single(
        self,
        prompt: Prompt,
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> Dict[str, float]:
        """Asynchronously evaluate single prompt."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._sync_evaluate, prompt, test_cases, fitness_fn
            )
    
    def _sync_evaluate(
        self,
        prompt: Prompt,
        test_cases: List[TestCase],
        fitness_fn: Callable
    ) -> Dict[str, float]:
        """Synchronous evaluation wrapper."""
        try:
            return fitness_fn(prompt, test_cases)
        except Exception as e:
            return {"fitness": 0.0, "error": str(e)}

# Global optimizer instance
performance_optimizer = PerformanceOptimizer()