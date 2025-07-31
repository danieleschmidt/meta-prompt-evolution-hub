# Contributing to Meta-Prompt-Evolution-Hub

Thank you for your interest in contributing to Meta-Prompt-Evolution-Hub! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/yourusername/meta-prompt-evolution-hub/issues) page
- Search existing issues before creating a new one
- Provide detailed reproduction steps and environment information
- Include relevant logs, error messages, and code snippets

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/meta-prompt-evolution-hub.git
cd meta-prompt-evolution-hub
```

2. Set up development environment:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

3. Run tests to verify setup:
```bash
pytest
```

### Development Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure tests pass:
```bash
# Run tests
pytest

# Run linting
ruff check .
black --check .

# Run type checking  
mypy meta_prompt_evolution/
```

3. Commit your changes:
```bash
git add .
git commit -m "feat: add your feature description"
```

4. Push and create a Pull Request:
```bash
git push origin feature/your-feature-name
```

### Pull Request Guidelines

- Follow the [Conventional Commits](https://conventionalcommits.org/) specification
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass
- Keep PRs focused and atomic
- Write clear PR descriptions explaining the changes

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(evolution): add NSGA-II multi-objective optimization`
- `fix(evaluation): handle timeout in distributed evaluator`
- `docs(readme): update installation instructions`

### Testing

- Write unit tests for new functionality in `tests/unit/`
- Write integration tests in `tests/integration/`
- Ensure test coverage remains above 80%
- Use meaningful test names and descriptions

### Documentation

- Update relevant documentation for new features
- Follow Google-style docstrings for Python code
- Update README.md if needed
- Add examples for new functionality

### Performance Considerations

- Profile performance-critical code
- Consider memory usage for large-scale operations
- Test with realistic data sizes
- Document performance characteristics

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Use Ruff for linting
- Use type hints for all public APIs
- Write self-documenting code with clear variable names

### Architecture

- Follow the existing module structure
- Keep modules focused and cohesive
- Use dependency injection for testability
- Follow SOLID principles
- Document architectural decisions

### Error Handling

- Use appropriate exception types
- Provide helpful error messages
- Log errors with appropriate levels
- Handle edge cases gracefully

## Getting Help

- Check the [documentation](https://meta-prompt-evolution-hub.readthedocs.io)
- Ask questions in [GitHub Discussions](https://github.com/yourusername/meta-prompt-evolution-hub/discussions)
- Join our community chat (link to be added)

## Recognition

Contributors will be recognized in:
- The CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to Meta-Prompt-Evolution-Hub!