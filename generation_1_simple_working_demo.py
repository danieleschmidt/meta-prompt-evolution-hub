#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple Demo
Autonomous SDLC Evolution - Basic functional implementation
"""

import sys
import os
sys.path.insert(0, '.')

from meta_prompt_evolution import PromptPopulation, Prompt
import json
import time
from typing import List, Dict


def demo_basic_evolution():
    """Demonstrate basic prompt evolution functionality."""
    print("🚀 Generation 1: MAKE IT WORK - Basic Evolution Demo")
    print("=" * 60)
    
    # Create initial population from seed prompts
    seed_prompts = [
        "You are a helpful AI assistant. Please help with {task}.",
        "As an AI assistant, I will assist you with {task}.",
        "Let me help you solve {task} step by step.",
        "I'm here to help with {task}. What do you need?",
        "To complete {task}, I'll provide clear guidance."
    ]
    
    population = PromptPopulation.from_seeds(seed_prompts)
    print(f"✅ Initial population created: {len(population)} prompts")
    
    # Simulate basic fitness evaluation
    for i, prompt in enumerate(population):
        # Simple fitness scoring (in real use, this would be LLM evaluation)
        fitness_score = 0.5 + (i * 0.1) + (len(prompt.text.split()) * 0.01)
        prompt.fitness_scores = {
            "fitness": fitness_score,
            "clarity": 0.7 + (i * 0.05),
            "helpfulness": 0.6 + (i * 0.08)
        }
    
    # Get top performers
    top_prompts = population.get_top_k(3)
    print(f"✅ Top 3 prompts identified")
    
    for i, prompt in enumerate(top_prompts):
        print(f"   {i+1}. Fitness: {prompt.fitness_scores['fitness']:.3f}")
        print(f"      Text: {prompt.text[:50]}...")
    
    # Simulate simple evolution (mutation)
    new_prompts = []
    for prompt in top_prompts:
        # Simple mutation: add variation to text
        variations = [
            prompt.text.replace("help", "assist"),
            prompt.text.replace("AI assistant", "helpful AI"),
            prompt.text + " I'm ready to begin."
        ]
        
        for variation in variations:
            new_prompt = Prompt(
                text=variation,
                generation=1,
                parent_ids=[prompt.id]
            )
            # Simple fitness inheritance with small variation
            base_fitness = prompt.fitness_scores["fitness"]
            new_prompt.fitness_scores = {
                "fitness": base_fitness + (0.1 * (hash(variation) % 10) / 10 - 0.05),
                "clarity": base_fitness * 0.9,
                "helpfulness": base_fitness * 1.1
            }
            new_prompts.append(new_prompt)
    
    # Create new population
    next_generation = PromptPopulation(new_prompts)
    next_generation.generation = 1
    
    print(f"✅ Generation 1 evolved: {len(next_generation)} new prompts")
    
    # Get best from new generation
    best_evolved = next_generation.get_top_k(1)[0]
    print(f"✅ Best evolved prompt:")
    print(f"   Fitness: {best_evolved.fitness_scores['fitness']:.3f}")
    print(f"   Text: {best_evolved.text}")
    
    return {
        "status": "SUCCESS",
        "generation": 1,
        "initial_population_size": len(population),
        "evolved_population_size": len(next_generation),
        "best_fitness": best_evolved.fitness_scores['fitness'],
        "execution_time": time.time()
    }


def run_simple_benchmark():
    """Run a simple benchmark to verify functionality."""
    print("\n🧪 Running Simple Benchmark")
    print("-" * 30)
    
    start_time = time.time()
    
    # Test prompt creation performance
    prompts = []
    for i in range(100):
        prompt = Prompt(f"Test prompt {i}", {"fitness": i * 0.01})
        prompts.append(prompt)
    
    population = PromptPopulation(prompts)
    
    # Test sorting performance
    top_10 = population.get_top_k(10)
    
    end_time = time.time()
    
    print(f"✅ Created and sorted {len(population)} prompts")
    print(f"✅ Top fitness: {top_10[0].fitness_scores['fitness']:.3f}")
    print(f"✅ Benchmark completed in {(end_time - start_time)*1000:.2f}ms")
    
    return {
        "prompts_created": len(population),
        "top_fitness": top_10[0].fitness_scores['fitness'],
        "execution_time_ms": (end_time - start_time) * 1000
    }


if __name__ == "__main__":
    try:
        # Run basic evolution demo
        results = demo_basic_evolution()
        
        # Run benchmark
        benchmark_results = run_simple_benchmark()
        
        # Save results
        full_results = {
            "generation": "1_MAKE_IT_WORK",
            "status": "COMPLETE",
            "demo_results": results,
            "benchmark_results": benchmark_results,
            "timestamp": time.time()
        }
        
        with open("generation_1_simple_working_results.json", "w") as f:
            json.dump(full_results, f, indent=2)
        
        print("\n🎯 GENERATION 1 COMPLETE: BASIC FUNCTIONALITY WORKING")
        print(f"✅ Results saved to generation_1_simple_working_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)