"""Integration tests for CLI functionality."""

import pytest
from typer.testing import CliRunner
from meta_prompt_evolution.cli.main import app

runner = CliRunner()


class TestCLI:
    """Test cases for CLI commands."""
    
    def test_evolve_command_help(self):
        """Test evolve command help."""
        result = runner.invoke(app, ["evolve", "--help"])
        assert result.exit_code == 0
        assert "Evolve prompts using genetic algorithms" in result.stdout
    
    def test_status_command(self):
        """Test status command."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Evolution Status" in result.stdout
        assert "evo_001" in result.stdout
    
    def test_benchmark_command_help(self):
        """Test benchmark command help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Benchmark a prompt against test cases" in result.stdout
    
    def test_benchmark_command_missing_prompt(self):
        """Test benchmark command without required prompt."""
        result = runner.invoke(app, ["benchmark"])
        assert result.exit_code != 0
        assert "Missing option" in result.stdout
    
    def test_benchmark_command_with_prompt(self):
        """Test benchmark command with prompt."""
        result = runner.invoke(app, [
            "benchmark", 
            "--prompt", "Test prompt for benchmarking"
        ])
        assert result.exit_code == 0
        assert "Benchmarking prompt" in result.stdout
        assert "Results:" in result.stdout