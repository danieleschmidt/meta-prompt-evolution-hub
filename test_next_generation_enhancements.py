#!/usr/bin/env python3
"""
Test Suite for Next-Generation Enhancements
Validates all new components work correctly together.
"""

import time
import json
import tempfile
import os
from typing import Dict, Any


def test_lightweight_engine():
    """Test the lightweight evolution engine."""
    print("🔬 Testing Lightweight Evolution Engine...")
    
    try:
        from meta_prompt_evolution.core.lightweight_engine import (
            MinimalEvolutionEngine, simple_fitness_evaluator
        )
        
        # Create engine
        engine = MinimalEvolutionEngine(population_size=10, mutation_rate=0.2)
        
        # Test evolution
        seed_prompts = ["Help me", "Please explain", "Can you assist"]
        evolved = engine.evolve_prompts(
            seed_prompts=seed_prompts,
            fitness_evaluator=simple_fitness_evaluator,
            generations=3
        )
        
        # Validate results
        assert len(evolved) == 10, f"Expected 10 prompts, got {len(evolved)}"
        assert all(p.fitness >= 0 for p in evolved), "All fitness scores should be non-negative"
        assert evolved[0].fitness >= evolved[-1].fitness, "Results should be sorted by fitness"
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            engine.export_results(evolved, f.name)
            assert os.path.exists(f.name), "Export file should be created"
            
            with open(f.name, 'r') as rf:
                data = json.load(rf)
                assert "final_population" in data
                assert "evolution_history" in data
            
            os.unlink(f.name)
        
        print("✅ Lightweight engine test passed")
        return True
        
    except Exception as e:
        print(f"❌ Lightweight engine test failed: {e}")
        return False


def test_quantum_inspired_algorithm():
    """Test the quantum-inspired evolution algorithm."""
    print("🔬 Testing Quantum-Inspired Algorithm...")
    
    try:
        from meta_prompt_evolution.evolution.algorithms.quantum_inspired import (
            QuantumInspiredEvolution, QuantumConfig, QuantumIndividual
        )
        
        # Test quantum individual
        q_individual = QuantumIndividual(dimensions=16)
        assert len(q_individual.qubits) == 16, "Should have 16 qubits"
        
        # Test quantum evolution config
        config = QuantumConfig(
            population_size=8,
            quantum_population_size=2,
            max_generations=3
        )
        
        qiea = QuantumInspiredEvolution(config)
        assert len(qiea.quantum_population) == 2, "Should have 2 quantum individuals"
        
        # Test quantum state analysis
        analysis = qiea.get_quantum_state_analysis()
        assert "quantum_coherence" in analysis
        assert "generation" in analysis
        
        # Test state export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            qiea.export_quantum_state(f.name)
            assert os.path.exists(f.name), "Quantum state export should be created"
            
            with open(f.name, 'r') as rf:
                data = json.load(rf)
                assert "algorithm" in data
                assert data["algorithm"] == "quantum_inspired_evolution"
            
            os.unlink(f.name)
        
        print("✅ Quantum-inspired algorithm test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quantum-inspired algorithm test failed: {e}")
        return False


def test_adaptive_monitoring():
    """Test the adaptive monitoring system."""
    print("🔬 Testing Adaptive Monitoring System...")
    
    try:
        from meta_prompt_evolution.monitoring.adaptive_monitor import (
            AdaptiveMetricsCollector, AlertRule, MetricPoint
        )
        
        # Create monitor
        monitor = AdaptiveMetricsCollector(max_history_points=100)
        
        # Register metrics
        monitor.register_metric("test_fitness", 1.0)
        monitor.register_metric("test_diversity", 0.8)
        
        # Add alert rule
        alert_rule = AlertRule(
            name="test_alert",
            metric="test_fitness", 
            condition="lt",
            threshold=0.3,
            severity="warning"
        )
        monitor.add_alert_rule(alert_rule)
        
        # Collect some metrics
        for i in range(20):
            fitness_value = 0.5 + 0.1 * (i % 5) # Varying fitness
            diversity_value = 0.6 + 0.05 * i
            
            monitor.collect_metric("test_fitness", fitness_value)
            monitor.collect_metric("test_diversity", diversity_value)
        
        # Check metric collection
        assert len(monitor.metrics["test_fitness"]) == 20, "Should have collected 20 fitness points"
        assert len(monitor.metrics["test_diversity"]) == 20, "Should have collected 20 diversity points"
        
        # Test low fitness alert
        monitor.collect_metric("test_fitness", 0.2)  # Should trigger alert
        time.sleep(0.1)  # Brief pause for alert processing
        
        # Test summary generation
        summary = monitor.get_metrics_summary()
        assert "total_metrics" in summary
        assert summary["total_metrics"] == 2
        assert "test_fitness" in summary["metrics"]
        
        # Test data export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            monitor.export_monitoring_data(f.name)
            assert os.path.exists(f.name), "Monitoring data export should be created"
            
            with open(f.name, 'r') as rf:
                data = json.load(rf)
                assert "summary" in data
                assert "raw_metrics" in data
                assert "alert_rules" in data
            
            os.unlink(f.name)
        
        print("✅ Adaptive monitoring test passed")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive monitoring test failed: {e}")
        return False


def test_research_analytics():
    """Test the research analytics platform."""
    print("🔬 Testing Research Analytics Platform...")
    
    try:
        from meta_prompt_evolution.research.analytics_platform import (
            ResearchAnalyticsPlatform, ResearchHypothesis
        )
        
        # Create platform
        platform = ResearchAnalyticsPlatform()
        
        # Register hypothesis
        hypothesis = ResearchHypothesis(
            id="test_hypothesis",
            title="Test Algorithm Comparison",
            description="Testing comparative performance",
            variables=["algorithm_type"],
            outcome_metrics=["fitness", "diversity"],
            expected_result="Algorithm A will outperform Algorithm B"
        )
        
        h_id = platform.register_hypothesis(hypothesis)
        assert h_id == "test_hypothesis"
        assert "test_hypothesis" in platform.hypotheses
        
        # Design experiment
        experiment_design = platform.design_experiment(
            hypothesis_id="test_hypothesis",
            experimental_conditions=[
                {"algorithm": "nsga2", "params": {"mutation_rate": 0.1}},
                {"algorithm": "map_elites", "params": {"mutation_rate": 0.1}}
            ],
            control_condition={"algorithm": "baseline", "params": {}},
            sample_size_per_condition=10
        )
        
        assert "power_analysis" in experiment_design
        assert experiment_design["sample_size_per_condition"] >= 10
        
        # Run comparative study
        study_results = platform.run_comparative_study(
            algorithms=["nsga2", "map_elites", "baseline"],
            test_scenarios=[
                {"name": "simple", "difficulty": 0.3},
                {"name": "complex", "difficulty": 0.7}
            ],
            iterations_per_scenario=5
        )
        
        assert "study_id" in study_results
        assert "statistical_analysis" in study_results
        assert "conclusions" in study_results
        
        # Generate research report
        report = platform.generate_research_report(study_results["study_id"])
        assert "title" in report
        assert "abstract" in report
        assert "methodology" in report
        assert "results" in report
        
        # Test data export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            platform.export_research_data(f.name)
            assert os.path.exists(f.name), "Research data export should be created"
            
            with open(f.name, 'r') as rf:
                data = json.load(rf)
                assert "platform" in data
                assert "hypotheses" in data
                assert "research_studies" in data
            
            os.unlink(f.name)
        
        print("✅ Research analytics test passed")
        return True
        
    except Exception as e:
        print(f"❌ Research analytics test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("🔬 Testing Component Integration...")
    
    try:
        from meta_prompt_evolution.core.lightweight_engine import MinimalEvolutionEngine
        from meta_prompt_evolution.monitoring.adaptive_monitor import AdaptiveMetricsCollector
        
        # Create integrated system
        engine = MinimalEvolutionEngine(population_size=8, mutation_rate=0.15)
        monitor = AdaptiveMetricsCollector()
        
        # Register monitoring metrics
        monitor.register_metric("evolution_fitness", 1.0)
        monitor.register_metric("population_diversity", 1.0)
        
        # Define fitness function with monitoring
        def monitored_fitness(prompt_text: str) -> float:
            # Calculate fitness
            score = len(prompt_text.split()) * 0.1
            score += 0.2 if "help" in prompt_text.lower() else 0
            score += 0.1 if "?" in prompt_text else 0
            
            # Monitor the fitness
            monitor.collect_metric("evolution_fitness", score)
            
            return max(0, min(1, score))
        
        # Run evolution with monitoring
        seed_prompts = ["Help me understand", "Can you explain?"]
        evolved = engine.evolve_prompts(
            seed_prompts=seed_prompts,
            fitness_evaluator=monitored_fitness,
            generations=3
        )
        
        # Validate integration
        assert len(evolved) == 8, "Evolution should produce 8 prompts"
        assert len(monitor.metrics["evolution_fitness"]) > 0, "Monitoring should have collected metrics"
        
        # Check monitoring worked
        fitness_points = list(monitor.metrics["evolution_fitness"])
        assert len(fitness_points) >= 8, "Should have monitored at least 8 fitness evaluations"
        
        print("✅ Component integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Component integration test failed: {e}")
        return False


def run_comprehensive_test_suite() -> Dict[str, bool]:
    """Run all enhancement tests."""
    print("🚀 NEXT-GENERATION ENHANCEMENTS TEST SUITE")
    print("=" * 60)
    
    test_results = {}
    
    # Run individual component tests
    test_results["lightweight_engine"] = test_lightweight_engine()
    test_results["quantum_inspired"] = test_quantum_inspired_algorithm() 
    test_results["adaptive_monitoring"] = test_adaptive_monitoring()
    test_results["research_analytics"] = test_research_analytics()
    test_results["integration"] = test_integration()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} | {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL NEXT-GENERATION ENHANCEMENTS WORKING CORRECTLY!")
        return {"status": "success", "tests": test_results}
    else:
        print("⚠️  Some enhancement tests failed - check implementation")
        return {"status": "partial", "tests": test_results}


if __name__ == "__main__":
    results = run_comprehensive_test_suite()
    
    # Export test results
    with open('/root/repo/next_generation_test_results.json', 'w') as f:
        json.dump({
            "timestamp": time.time(),
            "test_suite": "next_generation_enhancements",
            "version": "1.0.0",
            **results
        }, f, indent=2)
    
    print(f"\n📁 Test results saved to: next_generation_test_results.json")