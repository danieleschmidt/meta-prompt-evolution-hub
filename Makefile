.PHONY: help install install-dev test test-unit test-integration lint format type-check security clean build docs serve-docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev,test]"
	pre-commit install

test:  ## Run all tests
	pytest tests/ -v --tb=short --cov=meta_prompt_evolution --cov-report=html --cov-report=term

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v --tb=short

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v --tb=short

lint:  ## Run linting checks
	ruff check .
	black --check .
	mypy meta_prompt_evolution/

format:  ## Format code
	black .
	ruff check --fix .

type-check:  ## Run type checking
	mypy meta_prompt_evolution/

security:  ## Run security checks
	bandit -r meta_prompt_evolution/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

clean:  ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -f .coverage
	rm -f bandit-report.json
	rm -f safety-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	python -m build

docs:  ## Generate documentation
	sphinx-build -b html docs/ docs/_build/html

serve-docs:  ## Serve documentation locally
	python -m http.server 8000 --directory docs/_build/html

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

setup-dev:  ## Complete development setup
	python -m venv venv
	. venv/bin/activate && pip install -e ".[dev,test]"
	. venv/bin/activate && pre-commit install
	@echo "Development environment set up. Activate with: source venv/bin/activate"