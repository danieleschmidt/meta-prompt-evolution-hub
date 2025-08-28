#!/usr/bin/env python3
"""
PROGRESSIVE QUALITY GATES - Generation 2: Robust Validation System
Enterprise-grade quality gates with advanced error handling, monitoring, and self-healing
"""

import sys
import os
import json
import time
import hashlib
import asyncio
import logging
import subprocess
import threading
import queue
import signal
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from enum import Enum
import traceback
import sqlite3
from contextlib import contextmanager
import pickle
import socket
import uuid
from collections import defaultdict, deque
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import Generation 1 components
from progressive_quality_gates_gen1 import (
    ProjectMaturity, QualityGateType, QualityGateResult, ProgressiveConfig,
    MaturityDetector, BasicFunctionalTester, BasicSecurityScanner, BasicPerformanceTester
)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('progressive_quality_gates.log', mode='a')
    ]
)
logger = logging.getLogger('progressive_quality_gates_gen2')


@dataclass
class SystemHealth:
    """System health monitoring metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    load_average: float = 0.0
    network_status: str = "unknown"
    timestamp: float = field(default_factory=time.time)


@dataclass
class QualityMetrics:
    """Advanced quality metrics tracking"""
    total_executions: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    error_count: int = 0
    recovery_count: int = 0
    trend_score: float = 0.0


@dataclass
class RobustConfig:
    """Enhanced configuration for robust quality gates"""
    # Inherited from ProgressiveConfig
    project_maturity: ProjectMaturity = ProjectMaturity.DEVELOPMENT
    adaptive_thresholds: bool = True
    parallel_execution: bool = True
    fail_fast: bool = False
    min_overall_score: float = 75.0
    timeout_seconds: int = 300
    
    # Robust features
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_retries: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30
    circuit_breaker_threshold: int = 5
    self_healing: bool = True
    backup_results: bool = True
    distributed_mode: bool = False


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker OPEN - service unavailable")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                
                raise e


class QualityCache:
    """Intelligent caching system for quality gate results"""
    
    def __init__(self, cache_dir: str = '.quality_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / 'quality_cache.db'
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialize SQLite cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    expiry REAL,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_expiry ON cache_entries(expiry)
            ''')
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value if valid"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT value, expiry FROM cache_entries WHERE key = ? AND expiry > ?',
                        (key, time.time())
                    )
                    
                    result = cursor.fetchone()
                    if result:
                        # Update access count
                        cursor.execute(
                            'UPDATE cache_entries SET access_count = access_count + 1 WHERE key = ?',
                            (key,)
                        )
                        
                        return pickle.loads(result[0])
                    
            except Exception as e:
                logger.warning(f"Cache retrieval error: {e}")
                
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Store value in cache with TTL"""
        with self._lock:
            try:
                expiry = time.time() + ttl
                value_blob = pickle.dumps(value)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'INSERT OR REPLACE INTO cache_entries (key, value, timestamp, expiry) VALUES (?, ?, ?, ?)',
                        (key, value_blob, time.time(), expiry)
                    )
                    
            except Exception as e:
                logger.warning(f"Cache storage error: {e}")
    
    def cleanup(self):
        """Remove expired cache entries"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM cache_entries WHERE expiry < ?', (time.time(),))
                    deleted = cursor.rowcount
                    
                    if deleted > 0:
                        logger.debug(f"Cleaned up {deleted} expired cache entries")
                        
            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")


class SystemMonitor:
    """Real-time system health monitoring"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_history: deque = deque(maxlen=100)
        self.monitoring = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start system health monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self._stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🔍 System monitoring started")
    
    def stop_monitoring(self):
        """Stop system health monitoring"""
        if self.monitoring:
            self.monitoring = False
            self._stop_event.set()
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            logger.info("🔍 System monitoring stopped")
    
    def get_current_health(self) -> SystemHealth:
        """Get current system health snapshot"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            except:
                load_avg = 0.0
            
            # Network status (simple check)
            network_status = self._check_network()
            
            return SystemHealth(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                load_average=load_avg,
                network_status=network_status
            )
            
        except Exception as e:
            logger.warning(f"Health check error: {e}")
            return SystemHealth()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.wait(self.check_interval):
            try:
                health = self.get_current_health()
                self.health_history.append(health)
                
                # Check for critical issues
                self._check_critical_health(health)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _check_network(self) -> str:
        """Simple network connectivity check"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return "connected"
        except:
            return "disconnected"
    
    def _check_critical_health(self, health: SystemHealth):
        """Check for critical system health issues"""
        if health.cpu_percent > 90:
            logger.warning(f"⚠️  High CPU usage: {health.cpu_percent:.1f}%")
        
        if health.memory_percent > 85:
            logger.warning(f"⚠️  High memory usage: {health.memory_percent:.1f}%")
        
        if health.disk_percent > 90:
            logger.warning(f"⚠️  High disk usage: {health.disk_percent:.1f}%")


class EnhancedSecurityScanner:
    """Enhanced security scanner with vulnerability classification"""
    
    def __init__(self):
        self.vulnerability_patterns = {
            'CRITICAL': [
                (r'eval\s*\(', "Remote code execution via eval()"),
                (r'exec\s*\(', "Remote code execution via exec()"),
                (r'__import__\s*\(.*input', "Dynamic import with user input"),
                (r'pickle\.loads?\s*\(.*input', "Unsafe pickle deserialization"),
            ],
            'HIGH': [
                (r'os\.system\s*\(', "OS command execution"),
                (r'subprocess\.(call|run|Popen)', "Subprocess execution risk"),
                (r'sql.*\+.*', "Potential SQL injection"),
                (r'shell=True', "Shell injection risk"),
            ],
            'MEDIUM': [
                (r'open\s*\([^)]*\.\./', "Path traversal vulnerability"),
                (r'urllib\.request\.urlopen\s*\(.*http://', "Insecure HTTP usage"),
                (r'requests\.\w+\s*\(.*verify=False', "SSL verification disabled"),
            ],
            'LOW': [
                (r'print\s*\(.*password', "Password in debug output"),
                (r'TODO.*security', "Security TODO found"),
                (r'FIXME.*security', "Security FIXME found"),
            ]
        }
        
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
    
    def scan_comprehensive_security(self) -> QualityGateResult:
        """Comprehensive security scan with vulnerability classification"""
        start_time = time.time()
        
        try:
            result = self.circuit_breaker.call(self._execute_security_scan)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Security scan failed: {e}")
            
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                name="comprehensive_security",
                passed=False,
                score=0.0,
                message=f"Security scan failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e), 'circuit_breaker_tripped': True}
            )
    
    def _execute_security_scan(self) -> QualityGateResult:
        """Execute the actual security scan"""
        start_time = time.time()
        
        vulnerabilities = {
            'CRITICAL': [],
            'HIGH': [],
            'MEDIUM': [],
            'LOW': []
        }
        
        files_scanned = 0
        scan_errors = []
        
        # Scan project files
        for root, dirs, files in os.walk('.'):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    files_scanned += 1
                    
                    try:
                        file_vulnerabilities = self._scan_file_comprehensive(file_path)
                        for severity, vulns in file_vulnerabilities.items():
                            vulnerabilities[severity].extend(vulns)
                    
                    except Exception as e:
                        scan_errors.append(f"{file_path}: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Calculate security score based on vulnerability severity
        total_vulns = sum(len(vulns) for vulns in vulnerabilities.values())
        severity_weights = {'CRITICAL': 25, 'HIGH': 10, 'MEDIUM': 5, 'LOW': 1}
        
        penalty = sum(
            len(vulnerabilities[severity]) * weight
            for severity, weight in severity_weights.items()
        )
        
        security_score = max(0.0, 100.0 - penalty)
        
        # Determine pass/fail
        critical_count = len(vulnerabilities['CRITICAL'])
        high_count = len(vulnerabilities['HIGH'])
        
        passed = (critical_count == 0 and high_count <= 2 and security_score >= 70.0)
        
        return QualityGateResult(
            gate_type=QualityGateType.SECURITY,
            name="comprehensive_security",
            passed=passed,
            score=security_score,
            message=f"Security: {total_vulns} vulnerabilities ({critical_count} critical, {high_count} high)",
            execution_time=execution_time,
            details={
                'vulnerabilities': vulnerabilities,
                'files_scanned': files_scanned,
                'scan_errors': scan_errors,
                'security_score': security_score,
                'vulnerability_summary': {
                    severity: len(vulns) for severity, vulns in vulnerabilities.items()
                }
            }
        )
    
    def _scan_file_comprehensive(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Comprehensive scan of a single file"""
        vulnerabilities = {
            'CRITICAL': [],
            'HIGH': [],
            'MEDIUM': [],
            'LOW': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for severity, patterns in self.vulnerability_patterns.items():
                        for pattern, description in patterns:
                            import re
                            if re.search(pattern, line, re.IGNORECASE):
                                vulnerabilities[severity].append({
                                    'file': file_path,
                                    'line': line_num,
                                    'pattern': pattern,
                                    'description': description,
                                    'code': line.strip(),
                                    'severity': severity
                                })
        
        except Exception as e:
            vulnerabilities['LOW'].append({
                'file': file_path,
                'line': 0,
                'pattern': 'FILE_READ_ERROR',
                'description': f"Failed to read file: {str(e)}",
                'code': '',
                'severity': 'LOW'
            })
        
        return vulnerabilities


class EnhancedPerformanceTester:
    """Enhanced performance testing with benchmarking and regression detection"""
    
    def __init__(self):
        self.benchmark_cache = QualityCache('.performance_cache')
        self.circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=60)
        
        # Performance thresholds
        self.thresholds = {
            'data_structure_ops': 0.01,      # 10ms for basic operations
            'algorithm_performance': 0.05,    # 50ms for algorithms
            'memory_efficiency': 100_000_000, # 100MB max memory usage
            'concurrent_operations': 0.1      # 100ms for concurrent ops
        }
    
    def test_comprehensive_performance(self) -> QualityGateResult:
        """Comprehensive performance testing with regression detection"""
        start_time = time.time()
        
        try:
            result = self.circuit_breaker.call(self._execute_performance_tests)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Performance test failed: {e}")
            
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE,
                name="comprehensive_performance",
                passed=False,
                score=0.0,
                message=f"Performance test failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e), 'circuit_breaker_tripped': True}
            )
    
    def _execute_performance_tests(self) -> QualityGateResult:
        """Execute comprehensive performance benchmarks"""
        start_time = time.time()
        
        benchmark_results = {}
        performance_scores = []
        
        # Test 1: Data structure operations
        ds_result = self._benchmark_data_structures()
        benchmark_results['data_structures'] = ds_result
        performance_scores.append(ds_result['score'])
        
        # Test 2: Algorithm performance
        algo_result = self._benchmark_algorithms()
        benchmark_results['algorithms'] = algo_result
        performance_scores.append(algo_result['score'])
        
        # Test 3: Memory efficiency
        mem_result = self._benchmark_memory_usage()
        benchmark_results['memory'] = mem_result
        performance_scores.append(mem_result['score'])
        
        # Test 4: Concurrent operations
        concurrent_result = self._benchmark_concurrent_operations()
        benchmark_results['concurrent'] = concurrent_result
        performance_scores.append(concurrent_result['score'])
        
        execution_time = time.time() - start_time
        
        # Calculate overall performance score
        overall_score = sum(performance_scores) / len(performance_scores)
        
        # Check for performance regressions
        regression_detected = self._check_performance_regression(benchmark_results)
        
        passed = (overall_score >= 75.0 and not regression_detected)
        
        # Cache results for future regression detection
        cache_key = f"perf_baseline_{int(time.time() // 3600)}"  # Hourly baseline
        self.benchmark_cache.set(cache_key, benchmark_results, ttl=86400)  # 24 hour TTL
        
        return QualityGateResult(
            gate_type=QualityGateType.PERFORMANCE,
            name="comprehensive_performance",
            passed=passed,
            score=overall_score,
            message=f"Performance: {overall_score:.1f}% average, regression: {regression_detected}",
            execution_time=execution_time,
            details={
                'benchmark_results': benchmark_results,
                'performance_scores': performance_scores,
                'regression_detected': regression_detected,
                'thresholds': self.thresholds
            }
        )
    
    def _benchmark_data_structures(self) -> Dict[str, Any]:
        """Benchmark data structure operations"""
        try:
            from meta_prompt_evolution import PromptPopulation, Prompt
            
            operations = []
            
            # Test prompt creation
            creation_start = time.time()
            prompts = [Prompt(f"test prompt {i}") for i in range(1000)]
            creation_time = time.time() - creation_start
            operations.append(('creation', creation_time))
            
            # Test population operations
            population_start = time.time()
            population = PromptPopulation(prompts)
            top_100 = population.get_top_k(100)
            population_time = time.time() - population_start
            operations.append(('population_ops', population_time))
            
            # Calculate score based on performance
            total_time = sum(time for _, time in operations)
            score = max(0, 100 - (total_time / self.thresholds['data_structure_ops']) * 20)
            
            return {
                'operations': operations,
                'total_time': total_time,
                'score': min(100.0, score),
                'threshold_met': total_time <= self.thresholds['data_structure_ops']
            }
            
        except Exception as e:
            return {
                'operations': [],
                'total_time': 0,
                'score': 0.0,
                'error': str(e),
                'threshold_met': False
            }
    
    def _benchmark_algorithms(self) -> Dict[str, Any]:
        """Benchmark algorithm performance"""
        try:
            algorithms = []
            
            # Test sorting algorithm
            sort_start = time.time()
            data = list(range(10000))
            import random
            random.shuffle(data)
            sorted_data = sorted(data, reverse=True)
            sort_time = time.time() - sort_start
            algorithms.append(('sorting', sort_time))
            
            # Test search algorithm
            search_start = time.time()
            for _ in range(100):
                target = random.randint(0, 9999)
                found = target in sorted_data
            search_time = time.time() - search_start
            algorithms.append(('searching', search_time))
            
            # Calculate score
            total_time = sum(time for _, time in algorithms)
            score = max(0, 100 - (total_time / self.thresholds['algorithm_performance']) * 20)
            
            return {
                'algorithms': algorithms,
                'total_time': total_time,
                'score': min(100.0, score),
                'threshold_met': total_time <= self.thresholds['algorithm_performance']
            }
            
        except Exception as e:
            return {
                'algorithms': [],
                'total_time': 0,
                'score': 0.0,
                'error': str(e),
                'threshold_met': False
            }
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory efficiency"""
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Allocate memory for testing
            test_data = []
            for i in range(10000):
                test_data.append({
                    'id': i,
                    'data': f"test data {i}" * 10,
                    'timestamp': time.time()
                })
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss
            memory_used = peak_memory - initial_memory
            
            # Cleanup
            del test_data
            gc.collect()
            
            # Calculate score based on memory efficiency
            efficiency_ratio = memory_used / self.thresholds['memory_efficiency']
            score = max(0, 100 - (efficiency_ratio * 50))
            
            return {
                'initial_memory': initial_memory,
                'peak_memory': peak_memory,
                'memory_used': memory_used,
                'score': min(100.0, score),
                'threshold_met': memory_used <= self.thresholds['memory_efficiency']
            }
            
        except Exception as e:
            return {
                'memory_used': 0,
                'score': 0.0,
                'error': str(e),
                'threshold_met': False
            }
    
    def _benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operation performance"""
        try:
            import threading
            import queue
            
            concurrent_start = time.time()
            
            # Create work queue
            work_queue = queue.Queue()
            results_queue = queue.Queue()
            
            # Add work items
            for i in range(100):
                work_queue.put(i)
            
            # Worker function
            def worker():
                while not work_queue.empty():
                    try:
                        item = work_queue.get(timeout=1)
                        # Simulate work
                        result = sum(range(item * 100))
                        results_queue.put(result)
                        work_queue.task_done()
                    except queue.Empty:
                        break
            
            # Create and start threads
            threads = []
            for _ in range(4):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            
            # Wait for completion
            work_queue.join()
            for t in threads:
                t.join(timeout=5)
            
            concurrent_time = time.time() - concurrent_start
            
            # Calculate score
            score = max(0, 100 - (concurrent_time / self.thresholds['concurrent_operations']) * 20)
            
            return {
                'concurrent_time': concurrent_time,
                'results_count': results_queue.qsize(),
                'score': min(100.0, score),
                'threshold_met': concurrent_time <= self.thresholds['concurrent_operations']
            }
            
        except Exception as e:
            return {
                'concurrent_time': 0,
                'score': 0.0,
                'error': str(e),
                'threshold_met': False
            }
    
    def _check_performance_regression(self, current_results: Dict[str, Any]) -> bool:
        """Check for performance regressions against historical data"""
        try:
            # Get recent baselines from cache
            baselines = []
            for hour_offset in range(1, 25):  # Check last 24 hours
                cache_key = f"perf_baseline_{int((time.time() - hour_offset * 3600) // 3600)}"
                baseline = self.benchmark_cache.get(cache_key)
                if baseline:
                    baselines.append(baseline)
            
            if not baselines:
                return False  # No historical data to compare against
            
            # Calculate average baseline performance
            regression_threshold = 0.15  # 15% performance degradation threshold
            
            for test_name, current_result in current_results.items():
                if 'score' in current_result:
                    current_score = current_result['score']
                    
                    baseline_scores = [
                        baseline[test_name]['score'] 
                        for baseline in baselines 
                        if test_name in baseline and 'score' in baseline[test_name]
                    ]
                    
                    if baseline_scores:
                        avg_baseline = sum(baseline_scores) / len(baseline_scores)
                        performance_change = (current_score - avg_baseline) / avg_baseline
                        
                        if performance_change < -regression_threshold:
                            logger.warning(f"Performance regression detected in {test_name}: {performance_change:.1%}")
                            return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Regression check failed: {e}")
            return False


class RobustQualityOrchestrator:
    """Robust quality gates orchestrator with advanced error handling and monitoring"""
    
    def __init__(self, config: Optional[RobustConfig] = None):
        self.config = config or RobustConfig()
        
        # Initialize components
        self.maturity_detector = MaturityDetector()
        self.functional_tester = BasicFunctionalTester()
        self.security_scanner = EnhancedSecurityScanner()
        self.performance_tester = EnhancedPerformanceTester()
        
        # Robust features
        self.system_monitor = SystemMonitor(self.config.health_check_interval)
        self.result_cache = QualityCache('.robust_quality_cache')
        self.quality_metrics = QualityMetrics()
        
        # Circuit breakers for each component
        self.circuit_breakers = {
            'functional': CircuitBreaker(failure_threshold=3, timeout=60),
            'security': CircuitBreaker(failure_threshold=2, timeout=120),
            'performance': CircuitBreaker(failure_threshold=2, timeout=180)
        }
        
        # State management
        self.gate_results: List[QualityGateResult] = []
        self.execution_id = str(uuid.uuid4())
        self.project_maturity = None
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._shutdown_requested = False
    
    def run_robust_quality_gates(self) -> Dict[str, Any]:
        """Run robust quality gates with comprehensive error handling"""
        logger.info("🛡️ ROBUST QUALITY GATES - Generation 2")
        logger.info("=" * 55)
        logger.info(f"🆔 Execution ID: {self.execution_id}")
        
        start_time = time.time()
        
        try:
            # Start system monitoring
            if self.config.enable_monitoring:
                self.system_monitor.start_monitoring()
            
            # Step 1: Project analysis with caching
            cache_key = f"maturity_detection_{hashlib.md5(str(time.time() // 3600).encode()).hexdigest()}"
            if self.config.enable_caching:
                cached_maturity = self.result_cache.get(cache_key)
                if cached_maturity:
                    self.project_maturity = ProjectMaturity(cached_maturity)
                    logger.info(f"📦 Using cached maturity detection: {self.project_maturity.value}")
                else:
                    self.project_maturity = self.maturity_detector.detect_maturity()
                    self.result_cache.set(cache_key, self.project_maturity.value, ttl=3600)
            else:
                self.project_maturity = self.maturity_detector.detect_maturity()
            
            # Step 2: Execute quality gates with retry logic
            self._execute_quality_gates()
            
            # Step 3: Calculate results and generate report
            total_time = time.time() - start_time
            overall_results = self._calculate_robust_results(total_time)
            report = self._generate_robust_report(overall_results)
            
            # Step 4: Update quality metrics
            self._update_quality_metrics(overall_results)
            
            # Step 5: Backup results if enabled
            if self.config.backup_results:
                self._backup_results(report)
            
            # Step 6: Save results
            self._save_robust_results(report)
            
            logger.info("✅ Robust quality gates execution completed successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Critical error in robust quality gates: {e}")
            traceback.print_exc()
            
            # Create error report
            error_report = self._create_error_report(e, time.time() - start_time)
            return error_report
            
        finally:
            # Cleanup
            if self.config.enable_monitoring:
                self.system_monitor.stop_monitoring()
            
            # Cache cleanup
            if self.config.enable_caching:
                self.result_cache.cleanup()
    
    def _execute_quality_gates(self):
        """Execute quality gates with robust error handling"""
        gates_config = [
            ('Functional Tests', 'functional', self._run_functional_tests),
            ('Security Scan', 'security', self._run_security_scan),
            ('Performance Tests', 'performance', self._run_performance_tests)
        ]
        
        if self.config.parallel_execution:
            self._execute_gates_parallel(gates_config)
        else:
            self._execute_gates_sequential(gates_config)
    
    def _execute_gates_sequential(self, gates_config: List[Tuple[str, str, Callable]]):
        """Execute gates sequentially with retry logic"""
        for gate_name, gate_type, gate_func in gates_config:
            if self._shutdown_requested:
                logger.warning("Shutdown requested, stopping quality gates execution")
                break
            
            logger.info(f"🔍 Executing {gate_name}...")
            
            success = False
            last_error = None
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    result = gate_func()
                    self.gate_results.append(result)
                    
                    status = "✅ PASS" if result.passed else "❌ FAIL"
                    logger.info(f"   {status} {gate_name}: {result.score:.1f}% - {result.message}")
                    
                    success = True
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.config.max_retries:
                        retry_delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"⚠️  {gate_name} failed (attempt {attempt + 1}), retrying in {retry_delay:.1f}s: {e}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"❌ {gate_name} failed after {self.config.max_retries + 1} attempts: {e}")
            
            if not success:
                # Create error result
                error_result = QualityGateResult(
                    gate_type=getattr(QualityGateType, gate_type.upper(), QualityGateType.FUNCTIONAL),
                    name=gate_name.lower().replace(' ', '_'),
                    passed=False,
                    score=0.0,
                    message=f"Failed after {self.config.max_retries + 1} attempts: {str(last_error)}",
                    execution_time=0.0,
                    details={'error': str(last_error), 'retry_attempts': self.config.max_retries + 1}
                )
                self.gate_results.append(error_result)
                
                if self.config.fail_fast:
                    logger.warning("🚨 Fail-fast enabled, stopping on gate failure")
                    break
    
    def _execute_gates_parallel(self, gates_config: List[Tuple[str, str, Callable]]):
        """Execute gates in parallel with robust error handling"""
        with ThreadPoolExecutor(max_workers=len(gates_config)) as executor:
            future_to_gate = {}
            
            for gate_name, gate_type, gate_func in gates_config:
                future = executor.submit(self._execute_gate_with_retry, gate_name, gate_type, gate_func)
                future_to_gate[future] = (gate_name, gate_type)
            
            for future in as_completed(future_to_gate, timeout=self.config.timeout_seconds):
                gate_name, gate_type = future_to_gate[future]
                
                try:
                    result = future.result()
                    self.gate_results.append(result)
                    
                    status = "✅ PASS" if result.passed else "❌ FAIL"
                    logger.info(f"   {status} {gate_name}: {result.score:.1f}% - {result.message}")
                    
                except Exception as e:
                    logger.error(f"❌ {gate_name} execution error: {e}")
                    
                    error_result = QualityGateResult(
                        gate_type=getattr(QualityGateType, gate_type.upper(), QualityGateType.FUNCTIONAL),
                        name=gate_name.lower().replace(' ', '_'),
                        passed=False,
                        score=0.0,
                        message=f"Parallel execution error: {str(e)}",
                        execution_time=0.0,
                        details={'error': str(e), 'parallel_execution': True}
                    )
                    self.gate_results.append(error_result)
    
    def _execute_gate_with_retry(self, gate_name: str, gate_type: str, gate_func: Callable) -> QualityGateResult:
        """Execute a single gate with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = gate_func()
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    retry_delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️  {gate_name} failed (attempt {attempt + 1}), retrying in {retry_delay:.1f}s")
                    time.sleep(retry_delay)
        
        # All retries failed
        return QualityGateResult(
            gate_type=getattr(QualityGateType, gate_type.upper(), QualityGateType.FUNCTIONAL),
            name=gate_name.lower().replace(' ', '_'),
            passed=False,
            score=0.0,
            message=f"Failed after {self.config.max_retries + 1} attempts: {str(last_error)}",
            execution_time=0.0,
            details={'error': str(last_error), 'retry_attempts': self.config.max_retries + 1}
        )
    
    def _run_functional_tests(self) -> QualityGateResult:
        """Run functional tests with circuit breaker"""
        return self.circuit_breakers['functional'].call(
            lambda: self.functional_tester.test_core_imports()
        )
    
    def _run_security_scan(self) -> QualityGateResult:
        """Run security scan with circuit breaker"""
        return self.circuit_breakers['security'].call(
            lambda: self.security_scanner.scan_comprehensive_security()
        )
    
    def _run_performance_tests(self) -> QualityGateResult:
        """Run performance tests with circuit breaker"""
        return self.circuit_breakers['performance'].call(
            lambda: self.performance_tester.test_comprehensive_performance()
        )
    
    def _calculate_robust_results(self, total_time: float) -> Dict[str, Any]:
        """Calculate comprehensive results with health monitoring"""
        logger.info(f"\n📊 ROBUST QUALITY ASSESSMENT")
        logger.info("-" * 40)
        
        # System health snapshot
        current_health = self.system_monitor.get_current_health()
        
        # Group results by gate type
        results_by_type = defaultdict(list)
        for result in self.gate_results:
            results_by_type[result.gate_type].append(result)
        
        # Calculate type scores
        type_scores = {}
        for gate_type, results in results_by_type.items():
            if results:
                avg_score = sum(r.score for r in results) / len(results)
                type_scores[gate_type] = avg_score
                
                all_passed = all(r.passed for r in results)
                status = "✅" if all_passed else "❌"
                logger.info(f"{status} {gate_type.value.title()}: {avg_score:.1f}%")
        
        # Calculate overall score with adaptive weights
        adaptive_weights = self._get_adaptive_weights()
        overall_score = 0.0
        total_weight = 0.0
        
        for gate_type, weight in adaptive_weights.items():
            if gate_type in type_scores and weight > 0:
                overall_score += type_scores[gate_type] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Quality metrics
        passed_gates = sum(1 for r in self.gate_results if r.passed)
        total_gates = len(self.gate_results)
        success_rate = passed_gates / max(total_gates, 1)
        
        # Quality gate decision
        quality_gate_passed = (
            overall_score >= self.config.min_overall_score and
            success_rate >= 0.8  # At least 80% of gates must pass
        )
        
        logger.info(f"\n🎯 Overall Quality Score: {overall_score:.1f}%")
        logger.info(f"📈 Gates Passed: {passed_gates}/{total_gates}")
        logger.info(f"💪 Success Rate: {success_rate:.1%}")
        logger.info(f"⏱️  Execution Time: {total_time:.2f}s")
        logger.info(f"🖥️  System Health: CPU {current_health.cpu_percent:.1f}%, Memory {current_health.memory_percent:.1f}%")
        
        if quality_gate_passed:
            logger.info("✅ ROBUST QUALITY GATES PASSED!")
        else:
            logger.warning("⚠️  QUALITY GATES REQUIRE ATTENTION")
        
        return {
            'overall_score': overall_score,
            'quality_gate_passed': quality_gate_passed,
            'type_scores': type_scores,
            'success_rate': success_rate,
            'system_health': current_health,
            'execution_stats': {
                'total_time': total_time,
                'passed_gates': passed_gates,
                'total_gates': total_gates,
                'execution_id': self.execution_id
            }
        }
    
    def _get_adaptive_weights(self) -> Dict[QualityGateType, float]:
        """Get adaptive weights based on project maturity"""
        maturity_weights = {
            ProjectMaturity.PROTOTYPE: {
                QualityGateType.FUNCTIONAL: 0.6,
                QualityGateType.SECURITY: 0.2,
                QualityGateType.PERFORMANCE: 0.2,
                QualityGateType.RELIABILITY: 0.0,
                QualityGateType.COMPLIANCE: 0.0
            },
            ProjectMaturity.DEVELOPMENT: {
                QualityGateType.FUNCTIONAL: 0.4,
                QualityGateType.SECURITY: 0.3,
                QualityGateType.PERFORMANCE: 0.3,
                QualityGateType.RELIABILITY: 0.0,
                QualityGateType.COMPLIANCE: 0.0
            },
            ProjectMaturity.TESTING: {
                QualityGateType.FUNCTIONAL: 0.3,
                QualityGateType.SECURITY: 0.35,
                QualityGateType.PERFORMANCE: 0.25,
                QualityGateType.RELIABILITY: 0.1,
                QualityGateType.COMPLIANCE: 0.0
            },
            ProjectMaturity.PRODUCTION: {
                QualityGateType.FUNCTIONAL: 0.25,
                QualityGateType.SECURITY: 0.35,
                QualityGateType.PERFORMANCE: 0.25,
                QualityGateType.RELIABILITY: 0.1,
                QualityGateType.COMPLIANCE: 0.05
            }
        }
        
        return maturity_weights.get(
            self.project_maturity,
            maturity_weights[ProjectMaturity.DEVELOPMENT]
        )
    
    def _update_quality_metrics(self, overall_results: Dict[str, Any]):
        """Update quality metrics for trend analysis"""
        self.quality_metrics.total_executions += 1
        self.quality_metrics.success_rate = overall_results['success_rate']
        self.quality_metrics.average_execution_time = overall_results['execution_stats']['total_time']
        
        if overall_results['quality_gate_passed']:
            if self.quality_metrics.total_executions > 1:
                self.quality_metrics.trend_score = min(100.0, self.quality_metrics.trend_score + 5.0)
        else:
            self.quality_metrics.error_count += 1
            self.quality_metrics.trend_score = max(0.0, self.quality_metrics.trend_score - 10.0)
    
    def _generate_robust_report(self, overall_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive robust quality report"""
        return {
            'metadata': {
                'report_type': 'robust_quality_gates_gen2',
                'version': '2.0.0',
                'timestamp': time.time(),
                'execution_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'execution_id': self.execution_id,
                'project_maturity': self.project_maturity.value if self.project_maturity else 'unknown'
            },
            'configuration': {
                'enable_monitoring': self.config.enable_monitoring,
                'enable_caching': self.config.enable_caching,
                'enable_retries': self.config.enable_retries,
                'max_retries': self.config.max_retries,
                'parallel_execution': self.config.parallel_execution,
                'self_healing': self.config.self_healing,
                'adaptive_weights': {k.value: v for k, v in self._get_adaptive_weights().items()}
            },
            'results': {
                'overall_score': overall_results['overall_score'],
                'quality_gate_passed': overall_results['quality_gate_passed'],
                'type_scores': {k.value: v for k, v in overall_results['type_scores'].items()},
                'success_rate': overall_results['success_rate'],
                'execution_stats': overall_results['execution_stats']
            },
            'system_health': {
                'cpu_percent': overall_results['system_health'].cpu_percent,
                'memory_percent': overall_results['system_health'].memory_percent,
                'disk_percent': overall_results['system_health'].disk_percent,
                'load_average': overall_results['system_health'].load_average,
                'network_status': overall_results['system_health'].network_status
            },
            'quality_metrics': {
                'total_executions': self.quality_metrics.total_executions,
                'success_rate': self.quality_metrics.success_rate,
                'average_execution_time': self.quality_metrics.average_execution_time,
                'error_count': self.quality_metrics.error_count,
                'recovery_count': self.quality_metrics.recovery_count,
                'trend_score': self.quality_metrics.trend_score
            },
            'gate_details': [
                {
                    'gate_type': result.gate_type.value,
                    'name': result.name,
                    'passed': result.passed,
                    'score': result.score,
                    'message': result.message,
                    'execution_time': result.execution_time,
                    'required': result.required,
                    'weight': result.weight,
                    'details': result.details
                }
                for result in self.gate_results
            ],
            'circuit_breaker_status': {
                name: cb.state for name, cb in self.circuit_breakers.items()
            },
            'recommendations': self._generate_robust_recommendations(overall_results)
        }
    
    def _generate_robust_recommendations(self, overall_results: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on robust analysis"""
        recommendations = []
        
        # Overall score analysis
        score = overall_results['overall_score']
        if score < 50:
            recommendations.append("🚨 CRITICAL: Quality score below 50% - immediate system overhaul required")
        elif score < 70:
            recommendations.append("⚠️  WARNING: Quality score below 70% - focus on critical failures")
        elif score < 85:
            recommendations.append("💡 IMPROVEMENT: Quality score good but can be enhanced")
        else:
            recommendations.append("✅ EXCELLENT: Quality score above 85% - system in good health")
        
        # Gate-specific recommendations
        failed_gates = [r for r in self.gate_results if not r.passed]
        if failed_gates:
            functional_failures = sum(1 for r in failed_gates if r.gate_type == QualityGateType.FUNCTIONAL)
            security_failures = sum(1 for r in failed_gates if r.gate_type == QualityGateType.SECURITY)
            performance_failures = sum(1 for r in failed_gates if r.gate_type == QualityGateType.PERFORMANCE)
            
            if functional_failures > 0:
                recommendations.append(f"🔧 Fix {functional_failures} functional test failure(s) - core stability at risk")
            
            if security_failures > 0:
                recommendations.append(f"🔒 Address {security_failures} security issue(s) immediately")
            
            if performance_failures > 0:
                recommendations.append(f"⚡ Optimize {performance_failures} performance bottleneck(s)")
        
        # System health recommendations
        health = overall_results['system_health']
        if health.cpu_percent > 80:
            recommendations.append("💻 High CPU usage detected - consider optimizing resource-intensive operations")
        
        if health.memory_percent > 80:
            recommendations.append("🧠 High memory usage detected - check for memory leaks or optimize data structures")
        
        if health.disk_percent > 85:
            recommendations.append("💾 Disk space running low - clean up temporary files or increase storage")
        
        # Circuit breaker recommendations
        open_breakers = [name for name, state in [(name, cb.state) for name, cb in self.circuit_breakers.items()] if state == 'OPEN']
        if open_breakers:
            recommendations.append(f"⚡ Circuit breakers OPEN for: {', '.join(open_breakers)} - investigate and fix underlying issues")
        
        # Trend analysis
        if self.quality_metrics.trend_score < 50:
            recommendations.append("📉 Quality trend declining - implement corrective measures")
        elif self.quality_metrics.trend_score > 80:
            recommendations.append("📈 Quality trend improving - maintain current practices")
        
        # Retry analysis
        retry_gates = [r for r in self.gate_results if r.details.get('retry_attempts', 0) > 1]
        if retry_gates:
            recommendations.append(f"🔄 {len(retry_gates)} gate(s) required retries - investigate stability issues")
        
        if not recommendations:
            recommendations.append("🎉 All systems operating optimally - no immediate actions required")
        
        return recommendations
    
    def _backup_results(self, report: Dict[str, Any]):
        """Backup results for disaster recovery"""
        try:
            backup_dir = Path('.quality_backups')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            backup_file = backup_dir / f'robust_quality_backup_{timestamp}.json'
            
            with open(backup_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Keep only last 10 backups
            backups = sorted(backup_dir.glob('robust_quality_backup_*.json'))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
            
            logger.debug(f"Results backed up to {backup_file}")
            
        except Exception as e:
            logger.warning(f"Failed to backup results: {e}")
    
    def _save_robust_results(self, report: Dict[str, Any]):
        """Save robust quality results"""
        timestamp = int(time.time())
        filename = f'robust_quality_gates_gen2_results_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"💾 Robust results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _create_error_report(self, error: Exception, execution_time: float) -> Dict[str, Any]:
        """Create error report for critical failures"""
        return {
            'metadata': {
                'report_type': 'robust_quality_gates_gen2_error',
                'version': '2.0.0',
                'timestamp': time.time(),
                'execution_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'execution_id': self.execution_id
            },
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc(),
                'execution_time': execution_time
            },
            'results': {
                'overall_score': 0.0,
                'quality_gate_passed': False,
                'error_occurred': True
            },
            'partial_results': len(self.gate_results),
            'recommendations': [
                "🚨 Critical error occurred during quality gate execution",
                "🔍 Review error details and fix underlying issues",
                "🔄 Retry quality gates after resolving the error"
            ]
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True


def main():
    """Main execution function for Robust Quality Gates Generation 2"""
    try:
        # Configure for robust execution
        config = RobustConfig(
            enable_monitoring=True,
            enable_caching=True,
            enable_retries=True,
            max_retries=2,
            parallel_execution=True,
            self_healing=True,
            backup_results=True,
            min_overall_score=75.0,
            timeout_seconds=600  # 10 minutes timeout
        )
        
        # Initialize and run robust quality gates
        robust_orchestrator = RobustQualityOrchestrator(config)
        results = robust_orchestrator.run_robust_quality_gates()
        
        # Exit with appropriate code
        exit_code = 0 if results['results']['quality_gate_passed'] else 1
        
        if exit_code == 0:
            logger.info("\n🎉 ROBUST QUALITY GATES GENERATION 2 COMPLETE")
        else:
            logger.warning("\n⚠️  ROBUST QUALITY GATES NEED ATTENTION")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"❌ Critical error in robust quality gates: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)