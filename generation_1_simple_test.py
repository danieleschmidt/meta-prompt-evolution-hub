#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK (Simple)
Simple validation test for core evolutionary prompt optimization functionality.
"""

from meta_prompt_evolution.evolution.population import PromptPopulation, Prompt
from meta_prompt_evolution.evaluation.base import TestCase, FitnessFunction
import json
import time


class SimpleFitnessFunction(FitnessFunction):
    """Simple fitness function for testing."""
    
    def evaluate(self, prompt: Prompt, test_cases: list) -> dict:
        """Simple fitness evaluation based on prompt length and keywords."""
        text = prompt.text.lower()
        
        # Simple metrics
        length_score = min(len(text) / 100.0, 1.0)  # Normalize length
        keyword_score = sum(1 for keyword in ['help', 'assist', 'task', 'will'] if keyword in text) / 4.0
        
        fitness = (length_score + keyword_score) / 2.0
        
        return {
            "fitness": fitness,
            "length": len(text),
            "keywords": keyword_score,
            "accuracy": fitness  # Simple accuracy proxy
        }
    
    async def evaluate_async(self, prompt: Prompt, test_cases: list) -> dict:
        """Async version of evaluate method."""
        return self.evaluate(prompt, test_cases)


def run_generation_1_test():
    """Run Generation 1 simple functionality test."""
    print("🚀 Generation 1: MAKE IT WORK (Simple) - Starting Test")
    start_time = time.time()
    
    # Create initial population
    seed_prompts = [
        "You are a helpful assistant. Please {task}",
        "As an AI assistant, I will help you {task}",
        "Let me assist you with {task}",
        "I can help you to {task}",
        "Here's how I'll {task}"
    ]
    
    print(f"📝 Creating population with {len(seed_prompts)} seed prompts")
    population = PromptPopulation.from_seeds(seed_prompts)
    
    # Create simple test cases
    test_cases = [
        TestCase(
            input_data="Explain quantum computing",
            expected_output="Clear explanation",
            metadata={"difficulty": "medium"},
            weight=1.0
        ),
        TestCase(
            input_data="Write a summary",
            expected_output="Concise summary",
            metadata={"difficulty": "easy"},
            weight=1.0
        )
    ]
    
    # Simple fitness evaluation
    fitness_fn = SimpleFitnessFunction()
    
    print("🧮 Evaluating population fitness...")
    for prompt in population:
        prompt.fitness_scores = fitness_fn.evaluate(prompt, test_cases)
    
    # Get top performers
    top_prompts = population.get_top_k(3)
    
    print("\n📊 Top 3 Performing Prompts:")
    for i, prompt in enumerate(top_prompts, 1):
        fitness = prompt.fitness_scores.get("fitness", 0.0)
        print(f"{i}. Fitness: {fitness:.3f} - '{prompt.text[:50]}...'")
    
    # Simple mutation test
    print("\n🧬 Testing simple mutation...")
    original_prompt = top_prompts[0]
    mutated_text = original_prompt.text.replace("help", "support")
    mutated_prompt = Prompt(text=mutated_text, generation=1)
    mutated_prompt.fitness_scores = fitness_fn.evaluate(mutated_prompt, test_cases)
    
    print(f"Original: {original_prompt.fitness_scores.get('fitness', 0.0):.3f}")
    print(f"Mutated:  {mutated_prompt.fitness_scores.get('fitness', 0.0):.3f}")
    
    # Results summary
    execution_time = time.time() - start_time
    best_fitness = max(p.fitness_scores.get("fitness", 0.0) for p in population)
    
    results = {
        "generation": 1,
        "status": "COMPLETE",
        "execution_time": execution_time,
        "population_size": len(population),
        "best_fitness": best_fitness,
        "average_fitness": sum(p.fitness_scores.get("fitness", 0.0) for p in population) / len(population),
        "top_prompts": [
            {
                "id": p.id,
                "text": p.text,
                "fitness": p.fitness_scores.get("fitness", 0.0),
                "metrics": p.fitness_scores
            }
            for p in top_prompts
        ]
    }
    
    print(f"\n✅ Generation 1 Test Complete!")
    print(f"⏱️  Execution Time: {execution_time:.2f}s")
    print(f"🎯 Best Fitness: {best_fitness:.3f}")
    print(f"📈 Average Fitness: {results['average_fitness']:.3f}")
    
    # Save results
    with open("generation_1_simple_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("💾 Results saved to generation_1_simple_results.json")
    
    return results


if __name__ == "__main__":
    results = run_generation_1_test()
    
    # Validate success criteria
    if results["best_fitness"] > 0.3 and results["execution_time"] < 10.0:
        print("\n🎉 Generation 1: MAKE IT WORK - SUCCESS!")
        print("✅ Core functionality working")
        print("✅ Population management operational")
        print("✅ Fitness evaluation functional")
        print("✅ Ready for Generation 2 enhancement")
    else:
        print("\n⚠️  Generation 1 needs optimization")