"""
Scalable Sentiment Analyzer: Generation 3 - Performance Optimization & Scaling

Adds high-performance features including caching, concurrent processing,
auto-scaling, distributed computing, and advanced monitoring.
"""

import asyncio
import json
import time
import logging
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import multiprocessing as mp
from collections import defaultdict, deque
import gc
import psutil
import numpy as np

# Advanced imports for scaling
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using memory cache only")

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available - using standard multiprocessing")

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

@dataclass
class PerformanceMetrics:
    avg_processing_time: float = 0.0
    p95_processing_time: float = 0.0
    p99_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    active_workers: int = 0
    queue_size: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_stats: CacheStats = field(default_factory=CacheStats)

class AdaptiveCache:
    """High-performance adaptive cache with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.stats = CacheStats()
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU tracking"""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and not expired
            if key in self.cache:
                if current_time < self.expiry_times.get(key, float('inf')):
                    self.access_times[key] = current_time
                    self.stats.hits += 1
                    return self.cache[key]
                else:
                    # Expired
                    self._remove_key(key)
            
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with TTL"""
        with self._lock:
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            # Remove if exists (to update stats)
            if key in self.cache:
                self._remove_key(key)
            
            # Evict if at capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.expiry_times[key] = current_time + ttl
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all data structures"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.expiry_times:
            del self.expiry_times[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
        self.stats.evictions += 1
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired items"""
        while True:
            try:
                time.sleep(300)  # Every 5 minutes
                current_time = time.time()
                
                with self._lock:
                    expired_keys = [
                        key for key, expiry in self.expiry_times.items()
                        if current_time >= expiry
                    ]
                    
                    for key in expired_keys:
                        self._remove_key(key)
                        self.stats.evictions += 1
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

class DistributedCache:
    """Redis-based distributed cache for scaling across multiple instances"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.stats = CacheStats()
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                self.available = True
                logger.info("Connected to Redis for distributed caching")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to local cache")
                self.available = False
        else:
            self.available = False
        
        # Fallback to local cache
        if not self.available:
            self.local_cache = AdaptiveCache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get from distributed cache"""
        try:
            if self.available:
                data = self.redis_client.get(key)
                if data:
                    self.stats.hits += 1
                    return json.loads(data.decode('utf-8'))
                else:
                    self.stats.misses += 1
                    return None
            else:
                return self.local_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set in distributed cache"""
        try:
            ttl = ttl or self.default_ttl
            
            if self.available:
                data = json.dumps(value, default=str)
                self.redis_client.setex(key, ttl, data)
            else:
                self.local_cache.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")

class WorkerPool:
    """Adaptive worker pool with auto-scaling"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = None, scale_threshold: float = 0.8):
        self.min_workers = min_workers
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.scale_threshold = scale_threshold
        
        self.executor = ThreadPoolExecutor(max_workers=self.min_workers)
        self.current_workers = self.min_workers
        self.pending_tasks = deque()
        self.active_tasks = 0
        self._lock = threading.Lock()
        
        # Metrics
        self.processing_times = deque(maxlen=1000)
        self.last_scale_time = time.time()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_and_scale, daemon=True)
        self._monitor_thread.start()
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit task with auto-scaling"""
        with self._lock:
            self.active_tasks += 1
        
        future = self.executor.submit(self._wrapped_execution, func, *args, **kwargs)
        return future
    
    def _wrapped_execution(self, func: Callable, *args, **kwargs):
        """Wrapped execution with timing"""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return result
        finally:
            with self._lock:
                self.active_tasks -= 1
    
    def _monitor_and_scale(self):
        """Monitor performance and scale workers"""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                with self._lock:
                    utilization = self.active_tasks / self.current_workers
                    current_time = time.time()
                    
                    # Scale up if high utilization
                    if (utilization > self.scale_threshold and 
                        self.current_workers < self.max_workers and
                        current_time - self.last_scale_time > 30):
                        
                        new_workers = min(self.current_workers * 2, self.max_workers)
                        self._scale_workers(new_workers)
                        self.last_scale_time = current_time
                        logger.info(f"Scaled up to {new_workers} workers (utilization: {utilization:.2f})")
                    
                    # Scale down if low utilization
                    elif (utilization < 0.2 and 
                          self.current_workers > self.min_workers and
                          current_time - self.last_scale_time > 60):
                        
                        new_workers = max(self.current_workers // 2, self.min_workers)
                        self._scale_workers(new_workers)
                        self.last_scale_time = current_time
                        logger.info(f"Scaled down to {new_workers} workers (utilization: {utilization:.2f})")
                
            except Exception as e:
                logger.error(f"Worker scaling error: {e}")
    
    def _scale_workers(self, new_size: int):
        """Scale worker pool to new size"""
        if new_size != self.current_workers:
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(max_workers=new_size)
            self.current_workers = new_size
            
            # Shutdown old executor (gracefully)
            threading.Thread(target=lambda: old_executor.shutdown(wait=True), daemon=True).start()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker pool metrics"""
        with self._lock:
            if self.processing_times:
                processing_times_array = np.array(list(self.processing_times))
                avg_time = np.mean(processing_times_array)
                p95_time = np.percentile(processing_times_array, 95)
                p99_time = np.percentile(processing_times_array, 99)
            else:
                avg_time = p95_time = p99_time = 0.0
            
            return {
                "current_workers": self.current_workers,
                "active_tasks": self.active_tasks,
                "utilization": self.active_tasks / self.current_workers,
                "avg_processing_time": avg_time,
                "p95_processing_time": p95_time,
                "p99_processing_time": p99_time,
                "total_processed": len(self.processing_times)
            }

class ScalableSentimentAnalyzer:
    """High-performance scalable sentiment analyzer"""
    
    def __init__(self, 
                 cache_size: int = 50000,
                 cache_ttl: int = 3600,
                 min_workers: int = 4,
                 max_workers: int = None,
                 enable_distributed_cache: bool = False,
                 redis_url: str = "redis://localhost:6379"):
        
        logger.info("Initializing ScalableSentimentAnalyzer...")
        
        # Initialize base analyzer
        try:
            from robust_sentiment_analyzer import RobustSentimentAnalyzer
            self.base_analyzer = RobustSentimentAnalyzer(
                population_size=100,  # Larger population for better quality
                mutation_rate=0.05,   # Lower mutation rate for stability
                enable_monitoring=True
            )
        except ImportError:
            logger.warning("RobustSentimentAnalyzer not available, using basic analyzer")
            from sentiment_analyzer import SentimentEvolutionHub
            self.base_analyzer = SentimentEvolutionHub(population_size=100)
        
        # Initialize caching
        if enable_distributed_cache:
            self.cache = DistributedCache(redis_url, cache_ttl)
        else:
            self.cache = AdaptiveCache(cache_size, cache_ttl)
        
        # Initialize worker pool
        self.worker_pool = WorkerPool(min_workers, max_workers)
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.request_times = deque(maxlen=10000)
        self.throughput_counter = defaultdict(int)
        
        # Batch processing optimization
        self.batch_queue = asyncio.Queue(maxsize=1000)
        self.batch_processor_task = None
        
        # System monitoring
        self.process = psutil.Process()
        self.start_time = time.time()
        
        logger.info("ScalableSentimentAnalyzer initialized successfully")
    
    def _generate_cache_key(self, text: str, **kwargs) -> str:
        """Generate deterministic cache key"""
        # Include relevant parameters in key
        key_data = {
            "text": text,
            "model": "evolutionary-optimized",
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def analyze_sentiment_sync(self, text: str, **kwargs) -> Any:
        """Synchronous sentiment analysis with caching"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(text, **kwargs)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.debug(f"Cache hit for key {cache_key[:8]}...")
            cached_result['from_cache'] = True
            return cached_result
        
        # Perform analysis
        try:
            if hasattr(self.base_analyzer, 'analyze_sentiment'):
                result = self.base_analyzer.analyze_sentiment(text, **kwargs)
            else:
                result = self.base_analyzer.analyze_sentiment(text)
            
            # Convert to dict for caching
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_dict = result
            
            result_dict['from_cache'] = False
            
            # Cache the result
            self.cache.set(cache_key, result_dict)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.request_times.append(processing_time)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    async def analyze_sentiment(self, text: str, **kwargs) -> Any:
        """Async sentiment analysis with worker pool"""
        loop = asyncio.get_event_loop()
        future = self.worker_pool.submit(self.analyze_sentiment_sync, text, **kwargs)
        result = await loop.run_in_executor(None, future.result)
        
        # Update throughput counter
        current_minute = int(time.time() // 60)
        self.throughput_counter[current_minute] += 1
        
        return result
    
    async def batch_analyze(self, 
                          texts: List[str], 
                          batch_size: int = 100,
                          max_concurrency: int = 50) -> List[Any]:
        """High-performance batch analysis"""
        
        if not texts:
            return []
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def analyze_with_semaphore(text):
                async with semaphore:
                    return await self.analyze_sentiment(text)
            
            # Process batch concurrently
            batch_tasks = [analyze_with_semaphore(text) for text in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to analyze text {i+j}: {result}")
                    results.append({
                        'text': batch[j],
                        'error': str(result),
                        'label': 'unknown',
                        'confidence': 0.0
                    })
                else:
                    results.append(result)
        
        return results
    
    def start_batch_processor(self):
        """Start background batch processor for optimal throughput"""
        async def batch_processor():
            batch_buffer = []
            last_process_time = time.time()
            
            while True:
                try:
                    # Collect items for batch (with timeout)
                    try:
                        item = await asyncio.wait_for(self.batch_queue.get(), timeout=0.1)
                        batch_buffer.append(item)
                    except asyncio.TimeoutError:
                        pass
                    
                    # Process batch if buffer is full or timeout reached
                    current_time = time.time()
                    should_process = (
                        len(batch_buffer) >= 50 or  # Buffer full
                        (batch_buffer and current_time - last_process_time > 1.0)  # Timeout
                    )
                    
                    if should_process and batch_buffer:
                        texts = [item['text'] for item in batch_buffer]
                        futures = [item['future'] for item in batch_buffer]
                        
                        try:
                            results = await self.batch_analyze(texts, max_concurrency=20)
                            
                            # Set results for futures
                            for future, result in zip(futures, results):
                                if not future.done():
                                    future.set_result(result)
                        
                        except Exception as e:
                            # Set exception for all futures
                            for future in futures:
                                if not future.done():
                                    future.set_exception(e)
                        
                        batch_buffer.clear()
                        last_process_time = current_time
                
                except Exception as e:
                    logger.error(f"Batch processor error: {e}")
                    await asyncio.sleep(1)
        
        if self.batch_processor_task is None or self.batch_processor_task.done():
            self.batch_processor_task = asyncio.create_task(batch_processor())
    
    async def analyze_with_batch_queue(self, text: str) -> Any:
        """Add analysis to batch queue for optimal throughput"""
        future = asyncio.Future()
        
        try:
            self.batch_queue.put_nowait({
                'text': text,
                'future': future
            })
        except asyncio.QueueFull:
            # Fallback to direct processing
            return await self.analyze_sentiment(text)
        
        return await future
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        
        # Calculate processing time metrics
        if self.request_times:
            times_array = np.array(list(self.request_times))
            avg_time = np.mean(times_array)
            p95_time = np.percentile(times_array, 95)
            p99_time = np.percentile(times_array, 99)
        else:
            avg_time = p95_time = p99_time = 0.0
        
        # Calculate throughput (requests per second over last minute)
        current_minute = int(time.time() // 60)
        recent_throughput = sum(
            count for minute, count in self.throughput_counter.items()
            if minute >= current_minute - 1
        ) / 60.0
        
        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = self.process.cpu_percent()
        
        # Worker pool metrics
        worker_metrics = self.worker_pool.get_metrics()
        
        return PerformanceMetrics(
            avg_processing_time=avg_time,
            p95_processing_time=p95_time,
            p99_processing_time=p99_time,
            throughput_per_second=recent_throughput,
            active_workers=worker_metrics['current_workers'],
            queue_size=worker_metrics['active_tasks'],
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            cache_stats=self.cache.stats
        )
    
    def optimize_performance(self):
        """Trigger performance optimizations"""
        
        # Garbage collection
        gc.collect()
        
        # Cache optimization
        if hasattr(self.cache, '_periodic_cleanup'):
            threading.Thread(target=self.cache._periodic_cleanup, daemon=True).start()
        
        # Evolution step for better prompts
        if hasattr(self.base_analyzer, 'core_analyzer') and hasattr(self.base_analyzer.core_analyzer, 'evolve_generation'):
            try:
                self.base_analyzer.core_analyzer.evolve_generation()
                logger.info("Evolved prompt generation for better performance")
            except Exception as e:
                logger.warning(f"Evolution step failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        metrics = self.get_performance_metrics()
        
        # Health scoring
        health_score = 100
        issues = []
        
        if metrics.avg_processing_time > 1.0:
            health_score -= 20
            issues.append("High processing time")
        
        if metrics.memory_usage_mb > 1000:
            health_score -= 15
            issues.append("High memory usage")
        
        if metrics.cpu_usage_percent > 80:
            health_score -= 15
            issues.append("High CPU usage")
        
        if metrics.cache_stats.hit_rate < 0.3:
            health_score -= 10
            issues.append("Low cache hit rate")
        
        # Determine status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "health_score": health_score,
            "issues": issues,
            "metrics": asdict(metrics),
            "uptime_seconds": time.time() - self.start_time
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        # Stop batch processor
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
        
        # Shutdown worker pool
        self.worker_pool.executor.shutdown(wait=True)
        
        # Close cache connections
        if hasattr(self.cache, 'redis_client'):
            self.cache.redis_client.close()
        
        logger.info("Graceful shutdown completed")

# Ray-based distributed processing (if available)
if RAY_AVAILABLE:
    @ray.remote
    class DistributedSentimentWorker:
        """Ray remote worker for distributed sentiment analysis"""
        
        def __init__(self):
            from sentiment_analyzer import SentimentEvolutionHub
            self.analyzer = SentimentEvolutionHub(population_size=50)
        
        def analyze(self, text: str) -> dict:
            result = self.analyzer.analyze_sentiment(text)
            return asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
        
        def batch_analyze(self, texts: List[str]) -> List[dict]:
            return [self.analyze(text) for text in texts]

    class RayDistributedAnalyzer:
        """Ray-based distributed sentiment analyzer"""
        
        def __init__(self, num_workers: int = 4):
            if not ray.is_initialized():
                ray.init()
            
            self.workers = [DistributedSentimentWorker.remote() for _ in range(num_workers)]
            logger.info(f"Initialized {num_workers} Ray workers")
        
        async def analyze_distributed(self, texts: List[str]) -> List[dict]:
            """Distribute analysis across Ray workers"""
            
            if not texts:
                return []
            
            # Split work across workers
            chunk_size = max(1, len(texts) // len(self.workers))
            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            
            # Submit work to workers
            futures = []
            for i, chunk in enumerate(chunks):
                worker = self.workers[i % len(self.workers)]
                future = worker.batch_analyze.remote(chunk)
                futures.append(future)
            
            # Collect results
            chunk_results = await asyncio.gather(*[asyncio.to_thread(ray.get, future) for future in futures])
            
            # Flatten results
            results = []
            for chunk_result in chunk_results:
                results.extend(chunk_result)
            
            return results

# Demo and testing
async def performance_test():
    """Performance test for scalable analyzer"""
    
    analyzer = ScalableSentimentAnalyzer(
        cache_size=10000,
        min_workers=4,
        max_workers=16,
        enable_distributed_cache=False
    )
    
    # Start batch processor
    analyzer.start_batch_processor()
    
    # Test data
    test_texts = [
        "I love this amazing product!",
        "This service is terrible and disappointing",
        "It's an average experience, nothing special",
        "Outstanding quality and excellent service",
        "Poor customer support, very frustrating",
    ] * 100  # 500 texts total
    
    print("Starting performance test...")
    
    # Test individual requests
    start_time = time.time()
    individual_results = []
    for text in test_texts[:50]:  # Test first 50
        result = await analyzer.analyze_sentiment(text)
        individual_results.append(result)
    
    individual_time = time.time() - start_time
    
    # Test batch processing
    start_time = time.time()
    batch_results = await analyzer.batch_analyze(test_texts[:100])
    batch_time = time.time() - start_time
    
    # Test batch queue processing
    start_time = time.time()
    queue_tasks = [analyzer.analyze_with_batch_queue(text) for text in test_texts[:100]]
    queue_results = await asyncio.gather(*queue_tasks)
    queue_time = time.time() - start_time
    
    # Performance metrics
    metrics = analyzer.get_performance_metrics()
    health = analyzer.health_check()
    
    print(f"\nPerformance Test Results:")
    print(f"Individual processing (50 texts): {individual_time:.2f}s ({50/individual_time:.1f} req/s)")
    print(f"Batch processing (100 texts): {batch_time:.2f}s ({100/batch_time:.1f} req/s)")
    print(f"Queue processing (100 texts): {queue_time:.2f}s ({100/queue_time:.1f} req/s)")
    print(f"Cache hit rate: {metrics.cache_stats.hit_rate:.2f}")
    print(f"Average processing time: {metrics.avg_processing_time:.3f}s")
    print(f"P95 processing time: {metrics.p95_processing_time:.3f}s")
    print(f"Health status: {health['status']} (score: {health['health_score']})")
    
    await analyzer.shutdown()

if __name__ == "__main__":
    # Run performance test
    asyncio.run(performance_test())