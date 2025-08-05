#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - Advanced Caching System
High-performance caching with memory optimization and distributed storage.
"""

import hashlib
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import pickle

@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

class LRUCache:
    """Thread-safe LRU cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
                
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
                
            # Update access and move to end (most recently used)
            entry.update_access()
            self._cache.move_to_end(key)
            self._hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size estimate
            size_bytes = len(pickle.dumps(value))
            
            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
                
            # Add new entry
            self._cache[key] = entry
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                self._evict_lru()
                
            return True
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self._cache:
            self._cache.popitem(last=False)  # Remove first item (LRU)
            self._evictions += 1
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = self._hits / total_accesses if total_accesses > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "total_size_bytes": total_size,
                "average_size_bytes": total_size / len(self._cache) if self._cache else 0
            }

class EvaluationCache:
    """Specialized cache for prompt evaluation results."""
    
    def __init__(self, max_size: int = 5000, ttl: float = 7200):  # 2 hours
        self.cache = LRUCache(max_size, ttl)
        self.hash_cache = {}  # Cache for prompt hashes
        
    def _compute_cache_key(self, prompt_text: str, test_case_inputs: List[str]) -> str:
        """Compute cache key for prompt and test cases."""
        # Create deterministic hash of prompt and test inputs
        content = {
            "prompt": prompt_text,
            "test_inputs": sorted(test_case_inputs)  # Sort for consistency
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def get_evaluation_result(self, prompt_text: str, test_case_inputs: List[str]) -> Optional[Dict[str, float]]:
        """Get cached evaluation result."""
        cache_key = self._compute_cache_key(prompt_text, test_case_inputs)
        return self.cache.get(cache_key)
    
    def cache_evaluation_result(
        self, 
        prompt_text: str, 
        test_case_inputs: List[str], 
        result: Dict[str, float]
    ) -> bool:
        """Cache evaluation result."""
        cache_key = self._compute_cache_key(prompt_text, test_case_inputs)
        return self.cache.put(cache_key, result)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get evaluation cache statistics."""
        base_stats = self.cache.get_stats()
        base_stats["cache_type"] = "evaluation"
        return base_stats

class PopulationCache:
    """Cache for evolved populations and genetic history."""
    
    def __init__(self, max_generations: int = 100):
        self.max_generations = max_generations
        self.cache = LRUCache(max_generations * 10, default_ttl=None)  # No TTL for populations
        self.generation_history = OrderedDict()
        
    def cache_population(
        self, 
        algorithm: str, 
        generation: int, 
        population_data: Dict[str, Any]
    ) -> bool:
        """Cache population for a specific algorithm and generation."""
        cache_key = f"{algorithm}_gen_{generation}"
        
        # Store in main cache
        self.cache.put(cache_key, population_data)
        
        # Track generation history
        if algorithm not in self.generation_history:
            self.generation_history[algorithm] = []
            
        self.generation_history[algorithm].append({
            "generation": generation,
            "timestamp": time.time(),
            "best_fitness": population_data.get("best_fitness", 0.0),
            "diversity": population_data.get("diversity", 0.0),
            "population_size": population_data.get("population_size", 0)
        })
        
        # Limit history size
        if len(self.generation_history[algorithm]) > self.max_generations:
            self.generation_history[algorithm].pop(0)
            
        return True
    
    def get_population(self, algorithm: str, generation: int) -> Optional[Dict[str, Any]]:
        """Get cached population."""
        cache_key = f"{algorithm}_gen_{generation}"
        return self.cache.get(cache_key)
    
    def get_best_populations(self, algorithm: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get best populations for an algorithm."""
        if algorithm not in self.generation_history:
            return []
            
        # Sort by best fitness and return top N
        history = self.generation_history[algorithm]
        sorted_history = sorted(history, key=lambda x: x["best_fitness"], reverse=True)
        
        results = []
        for entry in sorted_history[:limit]:
            pop_data = self.get_population(algorithm, entry["generation"])
            if pop_data:
                results.append(pop_data)
                
        return results
    
    def get_algorithm_history(self, algorithm: str) -> List[Dict[str, Any]]:
        """Get evolution history for an algorithm."""
        return self.generation_history.get(algorithm, [])

class DistributedCache:
    """Distributed cache using multiple cache instances."""
    
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shards = [LRUCache(max_size=1000) for _ in range(num_shards)]
        self.executor = ThreadPoolExecutor(max_workers=num_shards)
        
    def _get_shard(self, key: str) -> int:
        """Get shard index for key."""
        return hash(key) % self.num_shards
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from appropriate shard."""
        shard_idx = self._get_shard(key)
        return self.shards[shard_idx].get(key)
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in appropriate shard."""
        shard_idx = self._get_shard(key)
        return self.shards[shard_idx].put(key, value, ttl)
    
    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values concurrently."""
        futures = []
        results = {}
        
        for key in keys:
            future = self.executor.submit(self.get, key)
            futures.append((key, future))
        
        for key, future in futures:
            try:
                value = future.result(timeout=1.0)
                if value is not None:
                    results[key] = value
            except:
                continue  # Skip failed retrievals
                
        return results
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all shards."""
        combined_stats = {
            "total_size": 0,
            "total_hits": 0,
            "total_misses": 0,
            "total_evictions": 0,
            "shard_stats": []
        }
        
        for i, shard in enumerate(self.shards):
            stats = shard.get_stats()
            combined_stats["total_size"] += stats["size"]
            combined_stats["total_hits"] += stats["hits"]
            combined_stats["total_misses"] += stats["misses"]
            combined_stats["total_evictions"] += stats["evictions"]
            combined_stats["shard_stats"].append({
                "shard_id": i,
                **stats
            })
        
        total_accesses = combined_stats["total_hits"] + combined_stats["total_misses"]
        combined_stats["overall_hit_rate"] = (
            combined_stats["total_hits"] / total_accesses if total_accesses > 0 else 0.0
        )
        
        return combined_stats

# Global cache instances
evaluation_cache = EvaluationCache()
population_cache = PopulationCache()
distributed_cache = DistributedCache()