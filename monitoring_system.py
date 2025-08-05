#!/usr/bin/env python3
"""
Generation 2: Monitoring and Health Checks
Real-time monitoring, health checks, and performance tracking.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import json

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass  
class EvolutionMetrics:
    """Evolution-specific metrics."""
    active_populations: int = 0
    total_evaluations: int = 0
    average_fitness: float = 0.0
    best_fitness: float = 0.0
    diversity_score: float = 0.0
    generation_time: float = 0.0
    algorithm_type: str = ""
    timestamp: float = field(default_factory=time.time)

class HealthChecker:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.health_status = "unknown"
        self.last_check_time = 0.0
        self.metrics_history = deque(maxlen=100)  # Keep last 100 metrics
        
    def start_monitoring(self):
        """Start background health monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)
                self._update_health_status(metrics)
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.check_interval)
                
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O (if available)
            network_io = {}
            try:
                net_io = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                }
            except:
                network_io = {'bytes_sent': 0, 'bytes_recv': 0}
                
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io=network_io
            )
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            return SystemMetrics()  # Return default metrics
            
    def _update_health_status(self, metrics: SystemMetrics):
        """Update overall health status based on metrics."""
        issues = []
        
        if metrics.cpu_usage > 90:
            issues.append("High CPU usage")
        if metrics.memory_usage > 85:
            issues.append("High memory usage")  
        if metrics.disk_usage > 90:
            issues.append("High disk usage")
            
        if not issues:
            self.health_status = "healthy"
        elif len(issues) == 1:
            self.health_status = "warning"
        else:
            self.health_status = "critical"
            
        self.last_check_time = time.time()
        
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available"}
            
        latest_metrics = self.metrics_history[-1]
        
        # Calculate averages over last 10 measurements
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        return {
            "status": self.health_status,
            "last_check": self.last_check_time,
            "current_metrics": {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "disk_usage": latest_metrics.disk_usage
            },
            "averages": {
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory
            },
            "recommendations": self._get_recommendations(latest_metrics)
        }
        
    def _get_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """Get system optimization recommendations."""
        recommendations = []
        
        if metrics.cpu_usage > 80:
            recommendations.append("Consider reducing population size or generations")
        if metrics.memory_usage > 75:
            recommendations.append("Enable result caching to reduce memory usage")
        if metrics.disk_usage > 80:
            recommendations.append("Clean up old log files and temporary data")
            
        return recommendations

class PerformanceTracker:
    """Track evolution performance metrics."""
    
    def __init__(self):
        self.evolution_metrics = []
        self.start_time = time.time()
        
    def record_evolution_metrics(self, metrics: EvolutionMetrics):
        """Record evolution performance metrics."""
        self.evolution_metrics.append(metrics)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get evolution performance summary."""
        if not self.evolution_metrics:
            return {"status": "no_data"}
            
        recent_metrics = self.evolution_metrics[-10:]  # Last 10 evolutions
        
        return {
            "total_evolutions": len(self.evolution_metrics),
            "uptime_seconds": time.time() - self.start_time,
            "recent_performance": {
                "average_generation_time": sum(m.generation_time for m in recent_metrics) / len(recent_metrics),
                "best_fitness_achieved": max(m.best_fitness for m in recent_metrics),
                "average_diversity": sum(m.diversity_score for m in recent_metrics) / len(recent_metrics)
            },
            "algorithms_used": list(set(m.algorithm_type for m in self.evolution_metrics)),
            "total_evaluations": sum(m.total_evaluations for m in self.evolution_metrics)
        }
        
    def export_metrics(self, filename: str):
        """Export metrics to JSON file."""
        data = {
            "export_time": time.time(),
            "evolution_metrics": [
                {
                    "active_populations": m.active_populations,
                    "total_evaluations": m.total_evaluations,
                    "average_fitness": m.average_fitness,
                    "best_fitness": m.best_fitness,
                    "diversity_score": m.diversity_score,
                    "generation_time": m.generation_time,
                    "algorithm_type": m.algorithm_type,
                    "timestamp": m.timestamp
                }
                for m in self.evolution_metrics
            ],
            "performance_summary": self.get_performance_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

# Global monitoring instances
health_checker = HealthChecker()
performance_tracker = PerformanceTracker()