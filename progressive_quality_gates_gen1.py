#!/usr/bin/env python3
"""
Progressive Quality Gates - Generation 1: Adaptive Threshold Evolution
TERRAGON LABS - Autonomous SDLC Implementation

This implements intelligent quality gates that learn and adapt thresholds
based on system performance history and capability evolution.
"""

import json
import time
import asyncio
import statistics
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import random
import math
from pathlib import Path

@dataclass
class QualityMetrics:
    """Container for quality measurement data"""
    test_coverage: float
    performance_score: float
    security_score: float
    reliability_score: float
    code_quality: float
    documentation_score: float
    maintainability: float
    complexity_score: float
    timestamp: datetime
    system_version: str = "1.0.0"

@dataclass 
class QualityThreshold:
    """Dynamic quality threshold with evolution tracking"""
    metric_name: str
    current_value: float
    target_value: float
    minimum_acceptable: float
    evolution_rate: float
    confidence_level: float
    last_updated: datetime
    historical_performance: List[float]

class AdaptiveQualityThresholds:
    """
    Intelligent threshold management system that learns from historical
    performance to set optimal quality gates.
    """
    
    def __init__(self, evolution_rate: float = 0.1, history_window: int = 1000):
        self.evolution_rate = evolution_rate
        self.history_window = history_window
        self.historical_metrics = deque(maxlen=history_window)
        self.thresholds: Dict[str, QualityThreshold] = {}
        self.performance_baselines: Dict[str, float] = {}
        self.quality_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize default thresholds with adaptive capabilities
        self._initialize_adaptive_thresholds()
        
    def _initialize_adaptive_thresholds(self):
        """Initialize starting thresholds that will evolve over time"""
        default_thresholds = {
            'test_coverage': QualityThreshold(
                metric_name='test_coverage',
                current_value=0.85,
                target_value=0.90,
                minimum_acceptable=0.80,
                evolution_rate=0.05,
                confidence_level=0.95,
                last_updated=datetime.now(),
                historical_performance=[]
            ),
            'performance_score': QualityThreshold(
                metric_name='performance_score',
                current_value=0.80,
                target_value=0.95,
                minimum_acceptable=0.70,
                evolution_rate=0.10,
                confidence_level=0.90,
                last_updated=datetime.now(),
                historical_performance=[]
            ),
            'security_score': QualityThreshold(
                metric_name='security_score',
                current_value=0.95,
                target_value=1.00,
                minimum_acceptable=0.90,
                evolution_rate=0.02,
                confidence_level=0.99,
                last_updated=datetime.now(),
                historical_performance=[]
            ),
            'reliability_score': QualityThreshold(
                metric_name='reliability_score',
                current_value=0.85,
                target_value=0.95,
                minimum_acceptable=0.75,
                evolution_rate=0.08,
                confidence_level=0.92,
                last_updated=datetime.now(),
                historical_performance=[]
            )
        }
        
        self.thresholds.update(default_thresholds)
        
    def record_quality_measurement(self, metrics: QualityMetrics):
        """Record new quality measurement and trigger threshold evolution"""
        self.historical_metrics.append(metrics)
        
        # Update performance baselines
        self._update_performance_baselines(metrics)
        
        # Evolve thresholds based on new data
        self._evolve_thresholds(metrics)
        
        # Update quality trends
        self._update_quality_trends(metrics)
        
    def _update_performance_baselines(self, metrics: QualityMetrics):
        """Update rolling performance baselines"""
        metric_dict = asdict(metrics)
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)) and key != 'timestamp':
                if key not in self.performance_baselines:
                    self.performance_baselines[key] = value
                else:
                    # Use exponential smoothing for baseline updates
                    alpha = 0.1
                    self.performance_baselines[key] = (
                        alpha * value + (1 - alpha) * self.performance_baselines[key]
                    )
                    
    def _evolve_thresholds(self, metrics: QualityMetrics):
        """Intelligently evolve quality thresholds based on system capability"""
        metric_dict = asdict(metrics)
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metric_dict:
                current_performance = metric_dict[metric_name]
                threshold.historical_performance.append(current_performance)
                
                # Keep only recent history for threshold evolution
                if len(threshold.historical_performance) > 100:
                    threshold.historical_performance = threshold.historical_performance[-100:]
                    
                # Calculate system capability metrics
                if len(threshold.historical_performance) >= 10:
                    recent_avg = statistics.mean(threshold.historical_performance[-10:])
                    recent_std = statistics.stdev(threshold.historical_performance[-10:])
                    system_capability = recent_avg - (1.0 * recent_std)  # Conservative estimate
                    
                    # Evolve threshold toward system capability
                    if system_capability > threshold.current_value:
                        # System is capable of higher standards
                        improvement_potential = min(
                            (system_capability - threshold.current_value) * threshold.evolution_rate,
                            threshold.target_value - threshold.current_value
                        )
                        threshold.current_value += improvement_potential
                    elif system_capability < threshold.minimum_acceptable:
                        # System struggling - may need to lower threshold temporarily
                        degradation_adjustment = (
                            threshold.current_value - system_capability
                        ) * (threshold.evolution_rate * 0.5)  # More conservative degradation
                        threshold.current_value = max(
                            threshold.current_value - degradation_adjustment,
                            threshold.minimum_acceptable
                        )
                        
                    threshold.last_updated = datetime.now()
                    
    def _update_quality_trends(self, metrics: QualityMetrics):
        """Track quality trends for predictive analysis"""
        metric_dict = asdict(metrics)
        
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)) and key != 'timestamp':
                self.quality_trends[key].append(value)
                # Keep only recent trend data
                if len(self.quality_trends[key]) > 50:
                    self.quality_trends[key] = self.quality_trends[key][-50:]
                    
    def validate_quality_gates(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Validate metrics against adaptive quality thresholds"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_pass': True,
            'gate_results': {},
            'adaptive_insights': {},
            'recommendations': []
        }
        
        metric_dict = asdict(metrics)
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metric_dict:
                current_value = metric_dict[metric_name]
                passed = current_value >= threshold.current_value
                
                gate_result = {
                    'metric': metric_name,
                    'current_value': current_value,
                    'threshold': threshold.current_value,
                    'target': threshold.target_value,
                    'minimum_acceptable': threshold.minimum_acceptable,
                    'passed': passed,
                    'threshold_evolution': self._calculate_threshold_evolution(threshold),
                    'system_capability_score': self._calculate_capability_score(threshold)
                }
                
                results['gate_results'][metric_name] = gate_result
                
                if not passed:
                    results['overall_pass'] = False
                    
        # Generate adaptive insights
        results['adaptive_insights'] = self._generate_adaptive_insights()
        results['recommendations'] = self._generate_recommendations(results['gate_results'])
        
        return results
        
    def _calculate_threshold_evolution(self, threshold: QualityThreshold) -> Dict[str, float]:
        """Calculate how the threshold has evolved over time"""
        if len(threshold.historical_performance) < 5:
            return {'trend': 0.0, 'volatility': 0.0, 'improvement_rate': 0.0}
            
        recent_performance = threshold.historical_performance[-10:]
        older_performance = threshold.historical_performance[-20:-10] if len(threshold.historical_performance) >= 20 else []
        
        # Calculate trend
        trend = 0.0
        if len(older_performance) > 0:
            trend = statistics.mean(recent_performance) - statistics.mean(older_performance)
            
        # Calculate volatility
        volatility = statistics.stdev(recent_performance) if len(recent_performance) > 1 else 0.0
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if len(threshold.historical_performance) >= 2:
            improvement_rate = (
                threshold.historical_performance[-1] - threshold.historical_performance[0]
            ) / len(threshold.historical_performance)
            
        return {
            'trend': trend,
            'volatility': volatility,
            'improvement_rate': improvement_rate
        }
        
    def _calculate_capability_score(self, threshold: QualityThreshold) -> float:
        """Calculate system capability score for this metric"""
        if len(threshold.historical_performance) < 5:
            return 0.5  # Unknown capability
            
        recent_avg = statistics.mean(threshold.historical_performance[-10:])
        capability_score = min(1.0, max(0.0, 
            (recent_avg - threshold.minimum_acceptable) / 
            (threshold.target_value - threshold.minimum_acceptable)
        ))
        
        return capability_score
        
    def _generate_adaptive_insights(self) -> Dict[str, Any]:
        """Generate insights about system quality evolution"""
        insights = {
            'system_maturity': self._assess_system_maturity(),
            'quality_momentum': self._calculate_quality_momentum(),
            'threshold_adaptability': self._assess_threshold_adaptability(),
            'performance_stability': self._assess_performance_stability()
        }
        return insights
        
    def _assess_system_maturity(self) -> float:
        """Assess overall system quality maturity (0.0 - 1.0)"""
        if not self.historical_metrics:
            return 0.0
            
        # Calculate based on consistency and performance levels
        recent_metrics = list(self.historical_metrics)[-10:] if len(self.historical_metrics) >= 10 else list(self.historical_metrics)
        
        consistency_scores = []
        performance_scores = []
        
        for metric_name, threshold in self.thresholds.items():
            if len(threshold.historical_performance) >= 5:
                # Consistency: Lower standard deviation = higher consistency
                std_dev = statistics.stdev(threshold.historical_performance[-10:])
                consistency = max(0.0, 1.0 - std_dev)
                consistency_scores.append(consistency)
                
                # Performance: Average performance relative to targets
                avg_performance = statistics.mean(threshold.historical_performance[-10:])
                performance = min(1.0, avg_performance / threshold.target_value) 
                performance_scores.append(performance)
                
        if not consistency_scores or not performance_scores:
            return 0.0
            
        maturity = (statistics.mean(consistency_scores) + statistics.mean(performance_scores)) / 2
        return maturity
        
    def _calculate_quality_momentum(self) -> float:
        """Calculate quality improvement momentum"""
        momentum_scores = []
        
        for threshold in self.thresholds.values():
            if len(threshold.historical_performance) >= 10:
                recent = threshold.historical_performance[-5:]
                older = threshold.historical_performance[-10:-5]
                
                recent_avg = statistics.mean(recent)
                older_avg = statistics.mean(older)
                
                momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                momentum_scores.append(momentum)
                
        return statistics.mean(momentum_scores) if momentum_scores else 0.0
        
    def _assess_threshold_adaptability(self) -> float:
        """Assess how well thresholds are adapting to system performance"""
        adaptability_scores = []
        
        for threshold in self.thresholds.values():
            if len(threshold.historical_performance) >= 10:
                # Check if threshold is appropriately challenging but achievable
                recent_performance = statistics.mean(threshold.historical_performance[-10:])
                threshold_gap = threshold.current_value - recent_performance
                
                # Ideal gap is small but positive (threshold slightly ahead of performance)
                ideal_gap = 0.05
                adaptability = max(0.0, 1.0 - abs(threshold_gap - ideal_gap) / ideal_gap)
                adaptability_scores.append(adaptability)
                
        return statistics.mean(adaptability_scores) if adaptability_scores else 0.5
        
    def _assess_performance_stability(self) -> float:
        """Assess how stable system performance is over time"""
        stability_scores = []
        
        for threshold in self.thresholds.values():
            if len(threshold.historical_performance) >= 10:
                # Lower coefficient of variation = higher stability
                mean_perf = statistics.mean(threshold.historical_performance[-10:])
                std_perf = statistics.stdev(threshold.historical_performance[-10:])
                
                if mean_perf > 0:
                    cv = std_perf / mean_perf
                    stability = max(0.0, 1.0 - cv)
                    stability_scores.append(stability)
                    
        return statistics.mean(stability_scores) if stability_scores else 0.0
        
    def _generate_recommendations(self, gate_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on quality gate results"""
        recommendations = []
        
        for metric_name, result in gate_results.items():
            if not result['passed']:
                gap = result['threshold'] - result['current_value']
                recommendations.append(
                    f"PRIORITY: Improve {metric_name} by {gap:.2f} to meet adaptive threshold"
                )
                
            # Check if threshold is too easy (performance significantly exceeds threshold)
            excess = result['current_value'] - result['threshold']
            if excess > 0.1:  # Performance significantly exceeds threshold
                recommendations.append(
                    f"OPTIMIZE: {metric_name} threshold can be raised - current performance exceeds by {excess:.2f}"
                )
                
        # System-level recommendations
        insights = self._generate_adaptive_insights()
        
        if insights['system_maturity'] < 0.7:
            recommendations.append("FOCUS: System maturity below 70% - prioritize consistency improvements")
            
        if insights['quality_momentum'] < 0:
            recommendations.append("ALERT: Negative quality momentum detected - investigate quality degradation")
            
        if insights['performance_stability'] < 0.6:
            recommendations.append("STABILIZE: Performance variance high - focus on process consistency")
            
        return recommendations
        
    def get_threshold_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all adaptive thresholds"""
        summary = {
            'thresholds': {},
            'system_overview': {
                'total_measurements': len(self.historical_metrics),
                'measurement_window': self.history_window,
                'evolution_rate': self.evolution_rate,
                'last_measurement': self.historical_metrics[-1].timestamp.isoformat() if self.historical_metrics else None
            },
            'performance_baselines': self.performance_baselines,
            'adaptive_insights': self._generate_adaptive_insights()
        }
        
        for name, threshold in self.thresholds.items():
            summary['thresholds'][name] = {
                'current_threshold': threshold.current_value,
                'target_value': threshold.target_value,
                'minimum_acceptable': threshold.minimum_acceptable,
                'last_updated': threshold.last_updated.isoformat(),
                'measurements_count': len(threshold.historical_performance),
                'evolution_stats': self._calculate_threshold_evolution(threshold),
                'capability_score': self._calculate_capability_score(threshold)
            }
            
        return summary


class QualityGateDemo:
    """Demonstration of adaptive quality gates functionality"""
    
    def __init__(self):
        self.adaptive_gates = AdaptiveQualityThresholds()
        
    async def simulate_quality_evolution(self, num_cycles: int = 20) -> Dict[str, Any]:
        """Simulate system quality evolution over multiple cycles"""
        print("🎯 Starting Progressive Quality Gates Demo - Generation 1")
        print("=" * 60)
        
        simulation_results = {
            'cycles': [],
            'threshold_evolution': {},
            'final_summary': {}
        }
        
        # Simulate quality measurements over time with realistic patterns
        base_metrics = {
            'test_coverage': 0.82,
            'performance_score': 0.75,
            'security_score': 0.92,
            'reliability_score': 0.80
        }
        
        for cycle in range(num_cycles):
            # Simulate realistic quality evolution patterns
            cycle_metrics = self._generate_realistic_metrics(base_metrics, cycle, num_cycles)
            
            # Record measurement and evolve thresholds
            self.adaptive_gates.record_quality_measurement(cycle_metrics)
            
            # Validate against adaptive gates
            validation_results = self.adaptive_gates.validate_quality_gates(cycle_metrics)
            
            cycle_result = {
                'cycle': cycle + 1,
                'metrics': asdict(cycle_metrics),
                'validation': validation_results,
                'insights': validation_results['adaptive_insights']
            }
            
            simulation_results['cycles'].append(cycle_result)
            
            # Print cycle results
            self._print_cycle_results(cycle + 1, cycle_metrics, validation_results)
            
            # Brief delay to simulate real-world timing
            await asyncio.sleep(0.1)
            
        # Generate final summary
        final_summary = self.adaptive_gates.get_threshold_summary()
        simulation_results['final_summary'] = final_summary
        simulation_results['threshold_evolution'] = self._analyze_threshold_evolution()
        
        self._print_final_summary(final_summary)
        
        return simulation_results
        
    def _generate_realistic_metrics(self, base_metrics: Dict[str, float], 
                                  cycle: int, total_cycles: int) -> QualityMetrics:
        """Generate realistic quality metrics that improve over time with some noise"""
        
        # Overall improvement trend
        improvement_factor = cycle / total_cycles * 0.2  # Up to 20% improvement
        
        # Add some realistic noise and patterns
        noise_level = 0.03
        
        metrics = {}
        for key, base_value in base_metrics.items():
            # Base improvement
            improved_value = base_value + improvement_factor
            
            # Add noise
            noise = random.gauss(0, noise_level)
            final_value = max(0.0, min(1.0, improved_value + noise))
            
            # Some metrics have specific patterns
            if key == 'performance_score' and cycle % 5 == 0:
                # Performance occasionally dips (maybe after feature additions)
                final_value *= 0.95
            elif key == 'security_score':
                # Security tends to be more stable but occasionally has issues
                if cycle == 12:  # Simulate a security issue discovery
                    final_value *= 0.88
            elif key == 'test_coverage' and cycle > 15:
                # Test coverage accelerates improvement in later cycles
                final_value += 0.05
                
            metrics[key] = final_value
            
        return QualityMetrics(
            test_coverage=metrics['test_coverage'],
            performance_score=metrics['performance_score'],
            security_score=metrics['security_score'],
            reliability_score=metrics['reliability_score'],
            code_quality=0.85 + improvement_factor + random.gauss(0, noise_level),
            documentation_score=0.78 + improvement_factor + random.gauss(0, noise_level),
            maintainability=0.82 + improvement_factor + random.gauss(0, noise_level),
            complexity_score=0.75 + improvement_factor + random.gauss(0, noise_level),
            timestamp=datetime.now(),
            system_version=f"1.{cycle}.0"
        )
        
    def _print_cycle_results(self, cycle: int, metrics: QualityMetrics, 
                           validation: Dict[str, Any]):
        """Print formatted results for each cycle"""
        print(f"\n📊 Cycle {cycle} Results:")
        print("-" * 40)
        
        overall_status = "✅ PASS" if validation['overall_pass'] else "❌ FAIL"
        print(f"Overall Status: {overall_status}")
        
        for metric_name, result in validation['gate_results'].items():
            status = "✅" if result['passed'] else "❌"
            threshold_change = result['threshold_evolution']['improvement_rate']
            change_indicator = "↗️" if threshold_change > 0.01 else "↘️" if threshold_change < -0.01 else "➡️"
            
            print(f"  {status} {metric_name}: {result['current_value']:.3f} "
                  f"(threshold: {result['threshold']:.3f}) {change_indicator}")
                  
        # Show adaptive insights
        insights = validation['adaptive_insights']
        print(f"\n🧠 Adaptive Insights:")
        print(f"  System Maturity: {insights['system_maturity']:.2f}")
        print(f"  Quality Momentum: {insights['quality_momentum']:.3f}")
        print(f"  Adaptability Score: {insights['threshold_adaptability']:.2f}")
        
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print comprehensive final summary"""
        print("\n" + "=" * 60)
        print("🎯 ADAPTIVE QUALITY GATES - FINAL SUMMARY")
        print("=" * 60)
        
        print(f"\n📈 System Overview:")
        overview = summary['system_overview']
        print(f"  Total Measurements: {overview['total_measurements']}")
        print(f"  Evolution Rate: {overview['evolution_rate']}")
        
        print(f"\n🎚️ Threshold Evolution:")
        for name, threshold_info in summary['thresholds'].items():
            evolution = threshold_info['evolution_stats']
            capability = threshold_info['capability_score']
            
            print(f"  {name}:")
            print(f"    Current Threshold: {threshold_info['current_threshold']:.3f}")
            print(f"    Capability Score: {capability:.2f}")
            print(f"    Improvement Rate: {evolution['improvement_rate']:.4f}")
            print(f"    Volatility: {evolution['volatility']:.3f}")
            
        print(f"\n🧠 Overall Adaptive Insights:")
        insights = summary['adaptive_insights']
        print(f"  System Maturity: {insights['system_maturity']:.2f} "
              f"({'Mature' if insights['system_maturity'] > 0.8 else 'Developing'})")
        print(f"  Quality Momentum: {insights['quality_momentum']:.3f} "
              f"({'Improving' if insights['quality_momentum'] > 0 else 'Declining'})")
        print(f"  Threshold Adaptability: {insights['threshold_adaptability']:.2f}")
        print(f"  Performance Stability: {insights['performance_stability']:.2f}")
        
    def _analyze_threshold_evolution(self) -> Dict[str, Any]:
        """Analyze how thresholds evolved during simulation"""
        evolution_analysis = {
            'total_threshold_changes': 0,
            'average_evolution_rate': 0.0,
            'most_adaptive_metric': '',
            'stability_metrics': {}
        }
        
        threshold_changes = []
        adaptivity_scores = []
        
        for name, threshold in self.adaptive_gates.thresholds.items():
            if len(threshold.historical_performance) >= 10:
                # Calculate how much threshold changed
                initial_performance = threshold.historical_performance[0]
                final_performance = threshold.historical_performance[-1] 
                performance_change = final_performance - initial_performance
                
                threshold_changes.append(abs(performance_change))
                adaptivity_scores.append((name, abs(performance_change)))
                
                evolution_analysis['stability_metrics'][name] = {
                    'performance_change': performance_change,
                    'threshold_responsiveness': abs(performance_change) / len(threshold.historical_performance)
                }
                
        if threshold_changes:
            evolution_analysis['total_threshold_changes'] = sum(threshold_changes)
            evolution_analysis['average_evolution_rate'] = statistics.mean(threshold_changes)
            
        if adaptivity_scores:
            evolution_analysis['most_adaptive_metric'] = max(adaptivity_scores, key=lambda x: x[1])[0]
            
        return evolution_analysis
        
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save simulation results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"progressive_quality_gates_gen1_results_{timestamp}.json"
            
        # Convert datetime objects to strings for JSON serialization
        json_results = self._prepare_for_json(results)
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"\n💾 Results saved to: {filepath.absolute()}")
        return filepath
        
    def _prepare_for_json(self, obj):
        """Recursively convert datetime objects to strings for JSON serialization"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj


async def main():
    """Main execution for Progressive Quality Gates Generation 1"""
    print("🚀 TERRAGON LABS - Progressive Quality Gates Generation 1")
    print("🎯 ADAPTIVE THRESHOLD EVOLUTION SYSTEM")
    print("=" * 70)
    
    start_time = time.time()
    
    # Initialize and run demo
    demo = QualityGateDemo()
    
    # Run simulation with realistic quality evolution
    results = await demo.simulate_quality_evolution(num_cycles=25)
    
    # Save results
    results_file = demo.save_results(results)
    
    execution_time = time.time() - start_time
    
    # Final execution report
    print(f"\n🏆 GENERATION 1 EXECUTION COMPLETE")
    print("=" * 50)
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    print(f"📊 Cycles Completed: {len(results['cycles'])}")
    print(f"🎚️  Thresholds Adapted: {len(results['final_summary']['thresholds'])}")
    print(f"🧠 System Maturity: {results['final_summary']['adaptive_insights']['system_maturity']:.2f}")
    print(f"📈 Quality Momentum: {results['final_summary']['adaptive_insights']['quality_momentum']:.3f}")
    print(f"📋 Results File: {results_file}")
    
    # Key achievements
    print(f"\n✅ KEY ACHIEVEMENTS:")
    print(f"   • Implemented adaptive threshold evolution system")
    print(f"   • Successfully tracked {len(results['cycles'])} quality measurement cycles")
    print(f"   • Demonstrated intelligent threshold adaptation")
    print(f"   • Generated comprehensive quality insights")
    print(f"   • Achieved self-improving quality gate system")
    
    return results


if __name__ == "__main__":
    # Run the progressive quality gates demonstration
    asyncio.run(main())