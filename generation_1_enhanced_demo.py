#!/usr/bin/env python3
"""
Generation 1: Enhanced Simple Demo
Progressive enhancement of basic evolutionary prompt optimization.
"""

import json
import time
from typing import List, Dict, Any
from meta_prompt_evolution.core.lightweight_engine import (
    MinimalEvolutionEngine, 
    LightweightPrompt,
    simple_fitness_evaluator
)


def enhanced_fitness_evaluator(prompt_text: str) -> float:
    """Enhanced fitness evaluator with more sophisticated scoring."""
    score = 0.0
    words = prompt_text.split()
    text_lower = prompt_text.lower()
    
    # Optimal length (8-15 words)
    length_score = 1.0 - abs(len(words) - 12) / 15.0
    score += max(0, length_score) * 0.25
    
    # Quality instruction words
    quality_words = [
        "please", "help", "explain", "describe", "analyze", 
        "step", "detail", "clearly", "concisely", "specifically"
    ]
    quality_score = sum(1 for word in quality_words if word in text_lower) / len(quality_words)
    score += quality_score * 0.35
    
    # Structure and clarity
    structure_indicators = [":", "?", "step by step", "first", "then", "how to"]
    structure_score = sum(0.1 for indicator in structure_indicators if indicator in text_lower)
    score += min(structure_score, 0.3)
    
    # Avoid repetition penalty
    unique_words = len(set(words))
    repetition_penalty = max(0, (len(words) - unique_words) * 0.05)
    score -= repetition_penalty
    
    # Professional tone bonus
    professional_words = ["could you", "would you", "i need", "assist", "provide"]
    professional_score = sum(0.05 for phrase in professional_words if phrase in text_lower)
    score += min(professional_score, 0.1)
    
    return max(0.0, min(1.0, score))


def run_generation_1_enhancement():
    """Run Generation 1 enhanced evolution with improved algorithms."""
    print("=" * 60)
    print("🚀 GENERATION 1: ENHANCED SIMPLE EVOLUTION")
    print("=" * 60)
    
    # Configure enhanced engine
    engine = MinimalEvolutionEngine(
        population_size=25,
        mutation_rate=0.12
    )
    
    # Enhanced seed prompts for diverse starting population
    seed_prompts = [
        "Help me understand this topic clearly",
        "Please explain the concept step by step",
        "I need assistance with detailed analysis",
        "Can you describe the process thoroughly",
        "Could you help me analyze this information",
        "Please provide a clear explanation",
        "I would like you to explain this",
        "Help me comprehend this subject"
    ]
    
    print(f"🌱 Starting with {len(seed_prompts)} seed prompts")
    print(f"🧬 Population size: {engine.population_size}")
    print(f"🔀 Mutation rate: {engine.mutation_rate}")
    print()
    
    start_time = time.time()
    
    # Run evolution
    evolved_prompts = engine.evolve_prompts(
        seed_prompts=seed_prompts,
        fitness_evaluator=enhanced_fitness_evaluator,
        generations=12
    )
    
    total_time = time.time() - start_time
    
    # Results analysis
    print("\n" + "=" * 60)
    print("📊 GENERATION 1 RESULTS")
    print("=" * 60)
    
    print(f"⏱️  Total evolution time: {total_time:.2f} seconds")
    print(f"🏆 Best fitness achieved: {evolved_prompts[0].fitness:.3f}")
    print(f"📈 Fitness improvement: {evolved_prompts[0].fitness - min(p.fitness for p in evolved_prompts):.3f}")
    
    print("\n🏆 TOP 10 EVOLVED PROMPTS:")
    for i, prompt in enumerate(evolved_prompts[:10], 1):
        print(f"{i:2d}. [{prompt.fitness:.3f}] {prompt.text}")
    
    # Export enhanced results
    results = {
        "generation": 1,
        "phase": "enhanced_simple",
        "config": {
            "population_size": engine.population_size,
            "mutation_rate": engine.mutation_rate,
            "generations": engine.generation,
            "fitness_evaluator": "enhanced_fitness_evaluator"
        },
        "performance_metrics": {
            "total_time_seconds": total_time,
            "best_fitness": evolved_prompts[0].fitness,
            "average_fitness": sum(p.fitness for p in evolved_prompts) / len(evolved_prompts),
            "fitness_variance": sum((p.fitness - sum(p.fitness for p in evolved_prompts) / len(evolved_prompts))**2 for p in evolved_prompts) / len(evolved_prompts)
        },
        "top_prompts": [
            {
                "rank": i + 1,
                "fitness": prompt.fitness,
                "text": prompt.text,
                "generation": prompt.generation,
                "metadata": prompt.metadata
            }
            for i, prompt in enumerate(evolved_prompts[:10])
        ],
        "evolution_history": engine.history,
        "timestamp": time.time()
    }
    
    # Save results
    filename = "generation_1_enhanced_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Enhanced results saved to {filename}")
    
    # Quality gates validation
    print("\n🔍 GENERATION 1 QUALITY GATES:")
    best_fitness = evolved_prompts[0].fitness
    avg_fitness = sum(p.fitness for p in evolved_prompts) / len(evolved_prompts)
    
    gates_passed = 0
    total_gates = 4
    
    # Gate 1: Best fitness threshold
    if best_fitness >= 0.6:
        print("✅ Gate 1: Best fitness >= 0.6 PASSED")
        gates_passed += 1
    else:
        print(f"❌ Gate 1: Best fitness >= 0.6 FAILED ({best_fitness:.3f})")
    
    # Gate 2: Evolution improvement
    initial_avg = sum(p.fitness for p in evolved_prompts[-5:]) / 5
    if best_fitness > initial_avg:
        print("✅ Gate 2: Evolution improvement PASSED")
        gates_passed += 1
    else:
        print("❌ Gate 2: Evolution improvement FAILED")
    
    # Gate 3: Execution time
    if total_time < 5.0:
        print("✅ Gate 3: Execution time < 5s PASSED")
        gates_passed += 1
    else:
        print(f"❌ Gate 3: Execution time < 5s FAILED ({total_time:.2f}s)")
    
    # Gate 4: Diversity maintenance
    diversity = engine.history[-1]["diversity"] if engine.history else 0
    if diversity > 0.2:
        print("✅ Gate 4: Diversity > 0.2 PASSED")
        gates_passed += 1
    else:
        print(f"❌ Gate 4: Diversity > 0.2 FAILED ({diversity:.3f})")
    
    print(f"\n🎯 Quality Gates: {gates_passed}/{total_gates} passed")
    
    if gates_passed == total_gates:
        print("🎉 Generation 1 completed successfully - proceeding to Generation 2")
        return True
    else:
        print("⚠️  Some quality gates failed - reviewing implementation")
        return False


if __name__ == "__main__":
    success = run_generation_1_enhancement()
    
    if success:
        print("\n" + "="*60)
        print("✨ GENERATION 1 ENHANCEMENT COMPLETE")
        print("Ready for Generation 2: Robust Implementation")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("🔧 GENERATION 1 NEEDS OPTIMIZATION")
        print("Reviewing and improving before proceeding")
        print("="*60)