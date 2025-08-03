"""Main CLI entry point for meta-prompt-hub commands."""

import typer
import json
import time
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel

from ..evolution.hub import EvolutionHub, EvolutionConfig
from ..evolution.population import PromptPopulation, Prompt
from ..evaluation.base import TestCase
from ..evaluation.evaluator import MockLLMProvider, ComprehensiveFitnessFunction

app = typer.Typer(help="Meta-Prompt-Evolution-Hub CLI")
console = Console()


@app.command()
def evolve(
    seeds: str = typer.Option(..., "--seeds", help="Comma-separated seed prompts"),
    generations: int = typer.Option(10, "--generations", help="Number of generations"),
    population_size: int = typer.Option(20, "--population-size", help="Population size"),
    algorithm: str = typer.Option("nsga2", "--algorithm", help="Algorithm: nsga2, map_elites, cma_es"),
    test_file: Optional[str] = typer.Option(None, "--test-file", help="Test cases JSON file"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file for results")
):
    """Evolve prompts using evolutionary algorithms."""
    console.print(Panel(
        f"🧬 Starting Evolution\n"
        f"Algorithm: {algorithm}\n"
        f"Generations: {generations}\n"
        f"Population Size: {population_size}",
        title="Meta-Prompt Evolution Hub"
    ))
    
    # Parse seed prompts
    seed_list = [s.strip() for s in seeds.split(",")]
    population = PromptPopulation.from_seeds(seed_list)
    
    # Expand population to target size with variations
    while len(population) < population_size:
        base_prompt = population.prompts[len(population) % len(seed_list)]
        variation = _create_variation(base_prompt)
        population.inject_prompts([variation])
    
    # Load test cases
    test_cases = _load_test_cases(test_file)
    if not test_cases:
        # Create default test cases for demonstration
        test_cases = _create_default_test_cases()
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        algorithm=algorithm,
        evaluation_parallel=True
    )
    
    # Run evolution
    hub = EvolutionHub(config)
    
    try:
        console.print("🚀 Evolution in progress...")
        evolved_population = hub.evolve(population, test_cases)
        
        # Display results
        _display_results(hub, evolved_population)
        
        # Save results if requested
        if output:
            _save_results(hub, evolved_population, output)
            console.print(f"📁 Results saved to {output}")
        
    except Exception as e:
        console.print(f"❌ Evolution failed: {e}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    prompt: str = typer.Option(..., "--prompt", help="Prompt to benchmark"),
    test_file: Optional[str] = typer.Option(None, "--test-file", help="Test cases file"),
    iterations: int = typer.Option(5, "--iterations", help="Number of benchmark iterations")
):
    """Benchmark a single prompt against test cases."""
    console.print(f"🔍 Benchmarking prompt: {prompt[:60]}...")
    
    # Load test cases
    test_cases = _load_test_cases(test_file)
    if not test_cases:
        test_cases = _create_default_test_cases()
    
    # Create fitness function
    fitness_fn = ComprehensiveFitnessFunction()
    
    # Create prompt object
    test_prompt = Prompt(text=prompt)
    
    # Run benchmark
    results = []
    for i in track(range(iterations), description="Running benchmark..."):
        scores = fitness_fn.evaluate(test_prompt, test_cases)
        results.append(scores)
    
    # Display results
    _display_benchmark_results(results)


@app.command()
def demo():
    """Run a demonstration evolution with sample data."""
    console.print(Panel(
        "🎭 Demo Mode: Evolving prompts for text summarization",
        title="Demo Evolution"
    ))
    
    # Sample seed prompts
    seeds = [
        "Please summarize the following text:",
        "Provide a concise summary of:",
        "Give me the key points from:"
    ]
    
    # Create test cases for summarization
    test_data = [
        {
            "input": "Artificial intelligence is transforming industries worldwide. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. This technology is being applied in healthcare, finance, transportation, and many other sectors.",
            "expected": "AI and machine learning are revolutionizing multiple industries by processing data to find patterns and make predictions.",
            "weight": 1.0
        },
        {
            "input": "Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Immediate action is needed to reduce greenhouse gas emissions.",
            "expected": "Climate change causes rising temperatures, melting ice, rising seas, and extreme weather, requiring urgent emission reductions.",
            "weight": 1.0
        }
    ]
    
    population = PromptPopulation.from_seeds(seeds)
    test_cases = [TestCase(
        input_data=item["input"],
        expected_output=item["expected"],
        weight=item["weight"]
    ) for item in test_data]
    
    # Run evolution
    config = EvolutionConfig(
        population_size=15,
        generations=8,
        algorithm="nsga2"
    )
    
    hub = EvolutionHub(config)
    evolved_population = hub.evolve(population, test_cases)
    
    _display_results(hub, evolved_population)


@app.command()
def status():
    """Show status and capabilities of the evolution system."""
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Check system components
    table.add_row("Evolutionary Algorithms", "✅ Available", "NSGA-II, MAP-Elites, CMA-ES")
    table.add_row("Evaluation System", "✅ Ready", "Multi-threaded evaluation")
    table.add_row("Mock LLM Provider", "✅ Active", "Simulated responses for testing")
    
    try:
        import ray
        table.add_row("Ray Framework", "✅ Available", "Distributed computing ready")
    except ImportError:
        table.add_row("Ray Framework", "⚠️ Not Available", "Install with: pip install ray")
    
    console.print(table)
    
    # Show algorithm details
    algo_table = Table(title="Available Algorithms")
    algo_table.add_column("Algorithm", style="cyan")
    algo_table.add_column("Type", style="magenta")
    algo_table.add_column("Best For", style="green")
    
    algo_table.add_row("NSGA-II", "Multi-objective", "Balanced optimization of multiple metrics")
    algo_table.add_row("MAP-Elites", "Quality-Diversity", "Maintaining diverse, high-quality solutions")
    algo_table.add_row("CMA-ES", "Continuous", "Fine-tuning prompt parameters")
    
    console.print(algo_table)


def _create_variation(base_prompt: Prompt) -> Prompt:
    """Create a variation of a base prompt."""
    import random
    
    words = base_prompt.text.split()
    variation_words = words.copy()
    
    # Simple variations
    modifiers = ["carefully", "clearly", "precisely", "thoroughly", "effectively"]
    connectors = ["and", "then", "also", "furthermore"]
    
    if random.random() < 0.5:
        variation_words.insert(random.randint(0, len(variation_words)), 
                              random.choice(modifiers))
    
    return Prompt(text=" ".join(variation_words))


def _load_test_cases(test_file: Optional[str]) -> List[TestCase]:
    """Load test cases from file."""
    if not test_file:
        return []
    
    try:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        test_cases = []
        for item in data:
            test_case = TestCase(
                input_data=item.get("input", ""),
                expected_output=item.get("expected", ""),
                metadata=item.get("metadata", {}),
                weight=item.get("weight", 1.0)
            )
            test_cases.append(test_case)
        
        return test_cases
    
    except Exception as e:
        console.print(f"⚠️ Could not load test file {test_file}: {e}")
        return []


def _create_default_test_cases() -> List[TestCase]:
    """Create default test cases for demonstration."""
    return [
        TestCase(
            input_data="Explain quantum computing",
            expected_output="Quantum computing uses quantum mechanics principles to process information in ways classical computers cannot.",
            weight=1.0
        ),
        TestCase(
            input_data="What is machine learning?",
            expected_output="Machine learning is a subset of AI that enables computers to learn and improve from data without explicit programming.",
            weight=1.0
        ),
        TestCase(
            input_data="Describe photosynthesis",
            expected_output="Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
            weight=1.0
        )
    ]


def _display_results(hub: EvolutionHub, population: PromptPopulation):
    """Display evolution results."""
    stats = hub.get_evolution_statistics()
    
    # Evolution summary
    summary_table = Table(title="Evolution Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Algorithm", stats.get("algorithm_used", "Unknown"))
    summary_table.add_row("Generations", str(stats.get("total_generations", 0)))
    summary_table.add_row("Final Best Fitness", f"{stats.get('final_best_fitness', 0.0):.3f}")
    summary_table.add_row("Fitness Improvement", f"{stats.get('fitness_improvement', 0.0):.3f}")
    summary_table.add_row("Average Diversity", f"{stats.get('average_diversity', 0.0):.3f}")
    
    console.print(summary_table)
    
    # Top prompts
    top_prompts = population.get_top_k(5)
    
    prompt_table = Table(title="Top 5 Evolved Prompts")
    prompt_table.add_column("Rank", style="cyan")
    prompt_table.add_column("Fitness", style="green")
    prompt_table.add_column("Prompt", style="yellow")
    
    for i, prompt in enumerate(top_prompts, 1):
        fitness = prompt.fitness_scores.get("fitness", 0.0) if prompt.fitness_scores else 0.0
        prompt_text = prompt.text[:60] + "..." if len(prompt.text) > 60 else prompt.text
        prompt_table.add_row(str(i), f"{fitness:.3f}", prompt_text)
    
    console.print(prompt_table)


def _display_benchmark_results(results: List[dict]):
    """Display benchmark results."""
    # Calculate averages
    avg_results = {}
    for key in results[0].keys():
        if isinstance(results[0][key], (int, float)):
            avg_results[key] = sum(r[key] for r in results) / len(results)
    
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Average", style="green")
    table.add_column("Min", style="red")
    table.add_column("Max", style="blue")
    
    for metric, avg_value in avg_results.items():
        min_val = min(r[metric] for r in results)
        max_val = max(r[metric] for r in results)
        
        table.add_row(
            metric.replace("_", " ").title(),
            f"{avg_value:.3f}",
            f"{min_val:.3f}",
            f"{max_val:.3f}"
        )
    
    console.print(table)


def _save_results(hub: EvolutionHub, population: PromptPopulation, output_file: str):
    """Save evolution results to file."""
    results = {
        "evolution_statistics": hub.get_evolution_statistics(),
        "final_population": [
            {
                "id": prompt.id,
                "text": prompt.text,
                "fitness_scores": prompt.fitness_scores,
                "generation": prompt.generation
            }
            for prompt in population.prompts
        ],
        "top_prompts": [
            {
                "rank": i + 1,
                "text": prompt.text,
                "fitness": prompt.fitness_scores.get("fitness", 0.0) if prompt.fitness_scores else 0.0
            }
            for i, prompt in enumerate(population.get_top_k(10))
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()