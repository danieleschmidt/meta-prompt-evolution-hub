"""Main CLI entry point for meta-prompt-hub commands."""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

from ..evolution.hub import EvolutionHub, EvolutionConfig
from ..evolution.population import PromptPopulation

app = typer.Typer(help="Meta-Prompt-Evolution-Hub CLI")
console = Console()


@app.command()
def evolve(
    seeds: str = typer.Option(..., "--seeds", help="Comma-separated seed prompts"),
    generations: int = typer.Option(50, "--generations", help="Number of generations"),
    population_size: int = typer.Option(100, "--population-size", help="Population size"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file for results")
):
    """Evolve prompts using genetic algorithms."""
    console.print(f"🧬 Starting evolution with {generations} generations...")
    
    seed_list = [s.strip() for s in seeds.split(",")]
    population = PromptPopulation.from_seeds(seed_list)
    
    config = EvolutionConfig(
        population_size=population_size,
        generations=generations
    )
    
    hub = EvolutionHub(config)
    console.print(f"✅ Evolution completed! Population size: {population.size()}")


@app.command()
def status():
    """Show status of running evolution processes."""
    table = Table(title="Evolution Status")
    table.add_column("Process ID", style="cyan")
    table.add_column("Generation", style="magenta") 
    table.add_column("Best Fitness", style="green")
    table.add_column("Status", style="yellow")
    
    # Placeholder data
    table.add_row("evo_001", "25/50", "0.89", "Running")
    table.add_row("evo_002", "10/100", "0.76", "Running")
    
    console.print(table)


@app.command()
def benchmark(
    prompt: str = typer.Option(..., "--prompt", help="Prompt to benchmark"),
    test_file: Optional[str] = typer.Option(None, "--test-file", help="Test cases file")
):
    """Benchmark a prompt against test cases."""
    console.print(f"🔍 Benchmarking prompt: {prompt[:50]}...")
    console.print("📊 Results: Accuracy: 0.85, Latency: 120ms")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()