"""
Adaptive Real-Time Monitoring System
Next-generation monitoring with predictive analytics and auto-healing capabilities.
"""

import time
import json
import threading
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
import math


@dataclass
class MetricPoint:
    """Individual metric measurement point."""
    timestamp: float
    value: float
    metric_name: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq", "trend_up", "trend_down", "anomaly"
    threshold: float
    duration_seconds: int = 60
    severity: str = "warning"  # "info", "warning", "critical"
    auto_action: Optional[str] = None  # Auto-healing action


class AdaptiveMetricsCollector:
    """Intelligent metrics collection with adaptive sampling."""
    
    def __init__(self, max_history_points: int = 10000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_history_points))
        self.sampling_rates = defaultdict(lambda: 1.0)  # Adaptive sampling
        self.alert_rules = []
        self.active_alerts = {}
        self.auto_healing_actions = {}
        
        # Anomaly detection state
        self.baseline_stats = {}
        self.anomaly_thresholds = {}
        
        # Real-time monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def register_metric(self, metric_name: str, initial_sampling_rate: float = 1.0):
        """Register a new metric for monitoring."""
        self.sampling_rates[metric_name] = initial_sampling_rate
        print(f"📊 Registered metric: {metric_name} (sampling: {initial_sampling_rate})")
    
    def collect_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Collect a metric point with adaptive sampling."""
        # Apply adaptive sampling
        if self.sampling_rates[metric_name] < 1.0:
            import random
            if random.random() > self.sampling_rates[metric_name]:
                return  # Skip this sample
        
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            metric_name=metric_name,
            tags=tags or {}
        )
        
        self.metrics[metric_name].append(point)
        self._update_adaptive_sampling(metric_name)
        self._check_alerts(metric_name, point)
    
    def _update_adaptive_sampling(self, metric_name: str):
        """Adapt sampling rate based on metric volatility."""
        points = list(self.metrics[metric_name])
        if len(points) < 10:
            return
        
        # Calculate recent volatility
        recent_values = [p.value for p in points[-10:]]
        volatility = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        
        # Adapt sampling rate
        if volatility > 0.5:  # High volatility - increase sampling
            self.sampling_rates[metric_name] = min(1.0, self.sampling_rates[metric_name] * 1.1)
        elif volatility < 0.1:  # Low volatility - decrease sampling  
            self.sampling_rates[metric_name] = max(0.1, self.sampling_rates[metric_name] * 0.95)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        print(f"🚨 Added alert rule: {rule.name} ({rule.condition} {rule.threshold})")
    
    def register_auto_healing_action(self, action_name: str, action_func: Callable):
        """Register an auto-healing action."""
        self.auto_healing_actions[action_name] = action_func
        print(f"🔧 Registered auto-healing action: {action_name}")
    
    def _check_alerts(self, metric_name: str, point: MetricPoint):
        """Check if any alert rules are triggered."""
        for rule in self.alert_rules:
            if rule.metric != metric_name:
                continue
                
            triggered = self._evaluate_alert_condition(rule, point)
            
            if triggered and rule.name not in self.active_alerts:
                self._trigger_alert(rule, point)
            elif not triggered and rule.name in self.active_alerts:
                self._resolve_alert(rule, point)
    
    def _evaluate_alert_condition(self, rule: AlertRule, point: MetricPoint) -> bool:
        """Evaluate if alert condition is met."""
        if rule.condition == "gt":
            return point.value > rule.threshold
        elif rule.condition == "lt": 
            return point.value < rule.threshold
        elif rule.condition == "eq":
            return abs(point.value - rule.threshold) < 0.001
        elif rule.condition == "trend_up":
            return self._detect_trend(rule.metric, "up", rule.duration_seconds)
        elif rule.condition == "trend_down":
            return self._detect_trend(rule.metric, "down", rule.duration_seconds)
        elif rule.condition == "anomaly":
            return self._detect_anomaly(rule.metric, point.value)
        
        return False
    
    def _detect_trend(self, metric_name: str, direction: str, duration_seconds: int) -> bool:
        """Detect trends over specified duration."""
        points = list(self.metrics[metric_name])
        if len(points) < 5:
            return False
        
        # Filter points within duration
        cutoff_time = time.time() - duration_seconds
        recent_points = [p for p in points if p.timestamp >= cutoff_time]
        
        if len(recent_points) < 3:
            return False
        
        # Calculate trend
        values = [p.value for p in recent_points]
        times = [p.timestamp for p in recent_points]
        
        # Simple linear regression slope
        n = len(values)
        sum_xy = sum(t * v for t, v in zip(times, values))
        sum_x = sum(times)
        sum_y = sum(values)
        sum_x2 = sum(t * t for t in times)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if direction == "up":
            return slope > 0.1  # Positive trend
        else:
            return slope < -0.1  # Negative trend
    
    def _detect_anomaly(self, metric_name: str, value: float) -> bool:
        """Detect statistical anomalies."""
        points = list(self.metrics[metric_name])
        if len(points) < 50:  # Need baseline
            return False
        
        # Update baseline statistics
        if metric_name not in self.baseline_stats:
            values = [p.value for p in points[-50:]]  # Recent baseline
            self.baseline_stats[metric_name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.1
            }
        
        stats = self.baseline_stats[metric_name]
        
        # Z-score based anomaly detection
        z_score = abs(value - stats["mean"]) / max(stats["std"], 0.1)
        return z_score > 3.0  # 3-sigma rule
    
    def _trigger_alert(self, rule: AlertRule, point: MetricPoint):
        """Trigger an alert."""
        alert_info = {
            "rule": rule.name,
            "metric": rule.metric,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "current_value": point.value,
            "timestamp": point.timestamp,
            "severity": rule.severity,
            "tags": point.tags
        }
        
        self.active_alerts[rule.name] = alert_info
        
        print(f"🚨 ALERT TRIGGERED: {rule.name}")
        print(f"   Metric: {rule.metric} = {point.value}")
        print(f"   Condition: {rule.condition} {rule.threshold}")
        print(f"   Severity: {rule.severity}")
        
        # Execute auto-healing action
        if rule.auto_action and rule.auto_action in self.auto_healing_actions:
            print(f"🔧 Executing auto-healing action: {rule.auto_action}")
            try:
                self.auto_healing_actions[rule.auto_action](alert_info)
            except Exception as e:
                print(f"❌ Auto-healing action failed: {e}")
    
    def _resolve_alert(self, rule: AlertRule, point: MetricPoint):
        """Resolve an active alert."""
        if rule.name in self.active_alerts:
            del self.active_alerts[rule.name]
            print(f"✅ ALERT RESOLVED: {rule.name}")
    
    def start_real_time_monitoring(self, update_interval: float = 1.0):
        """Start real-time monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                self._update_baseline_stats()
                self._print_dashboard()
                time.sleep(update_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"📊 Started real-time monitoring (interval: {update_interval}s)")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  Stopped real-time monitoring")
    
    def _update_baseline_stats(self):
        """Update baseline statistics for anomaly detection."""
        for metric_name, points in self.metrics.items():
            if len(points) >= 20:
                recent_values = [p.value for p in list(points)[-20:]]
                self.baseline_stats[metric_name] = {
                    "mean": statistics.mean(recent_values),
                    "std": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.1,
                    "min": min(recent_values),
                    "max": max(recent_values)
                }
    
    def _print_dashboard(self):
        """Print real-time dashboard to console."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
        
        print("=" * 80)
        print("🔥 ADAPTIVE MONITORING DASHBOARD")
        print("=" * 80)
        print(f"📊 Metrics: {len(self.metrics)} | 🚨 Active Alerts: {len(self.active_alerts)}")
        print(f"⏰ Last Update: {time.strftime('%H:%M:%S')}")
        print("-" * 80)
        
        # Show active alerts
        if self.active_alerts:
            print("🚨 ACTIVE ALERTS:")
            for alert_name, alert_info in self.active_alerts.items():
                print(f"   {alert_info['severity'].upper()}: {alert_name}")
                print(f"   {alert_info['metric']} = {alert_info['current_value']:.3f}")
            print("-" * 80)
        
        # Show metrics summary
        print("📈 METRICS SUMMARY:")
        for metric_name, points in self.metrics.items():
            if not points:
                continue
                
            recent_points = list(points)[-10:]  # Last 10 points
            current_value = recent_points[-1].value if recent_points else 0
            
            if len(recent_points) > 1:
                trend = "↗️" if recent_points[-1].value > recent_points[0].value else "↘️"
            else:
                trend = "➡️"
            
            sampling_rate = self.sampling_rates[metric_name]
            
            print(f"   {metric_name:20} | Current: {current_value:8.3f} {trend} | "
                  f"Sampling: {sampling_rate:.2f} | Points: {len(points)}")
        
        print("=" * 80)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {
            "total_metrics": len(self.metrics),
            "active_alerts": len(self.active_alerts),
            "alert_rules": len(self.alert_rules),
            "timestamp": time.time(),
            "metrics": {},
            "alerts": list(self.active_alerts.values())
        }
        
        for metric_name, points in self.metrics.items():
            if not points:
                continue
                
            values = [p.value for p in points]
            
            metric_summary = {
                "current_value": values[-1] if values else 0,
                "count": len(values),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": statistics.mean(values) if values else 0,
                "sampling_rate": self.sampling_rates[metric_name]
            }
            
            if len(values) > 1:
                metric_summary["std"] = statistics.stdev(values)
                metric_summary["recent_trend"] = values[-1] - values[max(0, len(values)-10)]
            
            summary["metrics"][metric_name] = metric_summary
        
        return summary
    
    def export_monitoring_data(self, filename: str):
        """Export all monitoring data for analysis."""
        export_data = {
            "export_timestamp": time.time(),
            "summary": self.get_metrics_summary(),
            "raw_metrics": {},
            "alert_rules": [asdict(rule) for rule in self.alert_rules],
            "baseline_stats": self.baseline_stats
        }
        
        # Export raw metric points
        for metric_name, points in self.metrics.items():
            export_data["raw_metrics"][metric_name] = [
                asdict(point) for point in points
            ]
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Monitoring data exported to {filename}")


# Example auto-healing actions
def restart_evolution_action(alert_info: Dict[str, Any]):
    """Auto-healing action: restart evolution process."""
    print(f"🔧 Auto-healing: Restarting evolution due to {alert_info['rule']}")
    # In practice, this would restart the evolution hub
    time.sleep(1)  # Simulate restart
    print("✅ Evolution process restarted")


def scale_resources_action(alert_info: Dict[str, Any]):
    """Auto-healing action: scale computing resources."""
    print(f"🔧 Auto-healing: Scaling resources due to {alert_info['rule']}")
    # In practice, this would trigger auto-scaling
    time.sleep(1)  # Simulate scaling
    print("✅ Resources scaled up")


def optimize_sampling_action(alert_info: Dict[str, Any]):
    """Auto-healing action: optimize sampling rates."""
    print(f"🔧 Auto-healing: Optimizing sampling for {alert_info['metric']}")
    # In practice, this would adjust sampling parameters
    time.sleep(0.5)
    print("✅ Sampling rates optimized")


# Demonstration
if __name__ == "__main__":
    # Create adaptive monitoring system
    monitor = AdaptiveMetricsCollector()
    
    # Register metrics
    monitor.register_metric("evolution_fitness", 1.0)
    monitor.register_metric("population_diversity", 0.8)
    monitor.register_metric("evaluation_latency", 1.0)
    monitor.register_metric("memory_usage", 0.5)
    
    # Add alert rules
    monitor.add_alert_rule(AlertRule(
        name="low_fitness_alert",
        metric="evolution_fitness",
        condition="lt",
        threshold=0.5,
        severity="warning",
        auto_action="restart_evolution"
    ))
    
    monitor.add_alert_rule(AlertRule(
        name="high_latency_alert", 
        metric="evaluation_latency",
        condition="gt",
        threshold=2.0,
        severity="critical",
        auto_action="scale_resources"
    ))
    
    monitor.add_alert_rule(AlertRule(
        name="diversity_trend_alert",
        metric="population_diversity",
        condition="trend_down",
        duration_seconds=30,
        severity="warning"
    ))
    
    # Register auto-healing actions
    monitor.register_auto_healing_action("restart_evolution", restart_evolution_action)
    monitor.register_auto_healing_action("scale_resources", scale_resources_action)
    monitor.register_auto_healing_action("optimize_sampling", optimize_sampling_action)
    
    # Start real-time monitoring
    monitor.start_real_time_monitoring(update_interval=2.0)
    
    # Simulate metric collection
    print("🚀 Starting metric simulation...")
    try:
        for i in range(60):  # 2 minute simulation
            import random
            
            # Simulate evolution metrics
            fitness = 0.7 + 0.3 * math.sin(i * 0.1) + random.gauss(0, 0.1)
            diversity = 0.5 + 0.2 * math.cos(i * 0.15) + random.gauss(0, 0.05)
            latency = 1.0 + 0.5 * random.random() + (0.5 if i > 40 else 0)  # Spike after 40s
            memory = 50 + 20 * (i / 60) + random.gauss(0, 5)
            
            monitor.collect_metric("evolution_fitness", max(0, min(1, fitness)))
            monitor.collect_metric("population_diversity", max(0, min(1, diversity)))
            monitor.collect_metric("evaluation_latency", max(0, latency))
            monitor.collect_metric("memory_usage", max(0, memory))
            
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped by user")
    
    finally:
        monitor.stop_real_time_monitoring()
        
        # Export final data
        monitor.export_monitoring_data("adaptive_monitoring_export.json")
        
        # Print final summary
        print("\n📊 FINAL MONITORING SUMMARY:")
        summary = monitor.get_metrics_summary()
        print(f"Total metrics collected: {summary['total_metrics']}")
        print(f"Total alerts triggered: {len(summary['alerts'])}")
        
        for metric_name, metric_data in summary["metrics"].items():
            print(f"{metric_name}: {metric_data['count']} points, "
                  f"final value: {metric_data['current_value']:.3f}")