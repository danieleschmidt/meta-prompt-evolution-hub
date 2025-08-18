# Testing Guide

This document outlines the testing strategy and practices for Meta-Prompt-Evolution-Hub.

## Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── e2e/           # End-to-end tests for complete workflows
├── performance/   # Performance and load tests
├── fixtures/      # Test data and fixtures
├── conftest.py    # Shared test configuration
└── __init__.py
```

## Test Types

### Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 1 second each)
- High coverage requirement (>95%)

### Integration Tests
- Test component interactions
- Use real database/cache instances
- Test API endpoints and data flow
- Moderate execution time (1-10 seconds)

### End-to-End Tests
- Test complete user workflows
- Use full application stack
- Test with real external services (when possible)
- Slower execution (10+ seconds)

### Performance Tests
- Load testing with multiple concurrent users
- Stress testing under high resource usage
- Benchmark critical algorithms and endpoints
- Memory and CPU profiling

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m performance

# Run with coverage
pytest --cov=meta_prompt_evolution --cov-report=html

# Run specific test file
pytest tests/unit/test_hub.py
```

### Using Tox for Multiple Environments
```bash
# Test all Python versions
tox

# Test specific environment
tox -e py311
tox -e lint
tox -e type
tox -e security
```

### Parallel Test Execution
```bash
# Run tests in parallel (faster for integration/e2e tests)
pytest -n auto

# Run specific number of workers
pytest -n 4
```

## Test Configuration

### Pytest Configuration
Located in `pytest.ini` with markers for different test types and coverage requirements.

### Environment Variables
Set these for testing:
```bash
export TESTING=true
export DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db
export REDIS_URL=redis://localhost:6379/1
export OPENAI_API_KEY=test_key  # Use test/mock keys
```

### Test Database
Use separate test database that gets reset between test runs:
```bash
# Setup test database
createdb test_meta_prompt_hub
pytest --create-db  # Creates schema and test data
```

## Writing Tests

### Test Structure Template
```python
import pytest
from meta_prompt_evolution.core.module import ComponentToTest

class TestComponentToTest:
    """Test cases for ComponentToTest."""
    
    def test_basic_functionality(self, sample_fixture):
        """Test basic functionality with clear description."""
        # Arrange
        component = ComponentToTest(config=sample_fixture)
        
        # Act
        result = component.method_to_test()
        
        # Assert
        assert result == expected_value
        assert component.state == expected_state
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        component = ComponentToTest()
        
        with pytest.raises(ValueError, match="Expected error message"):
            component.method_with_invalid_input(None)
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async methods."""
        component = ComponentToTest()
        result = await component.async_method()
        assert result is not None
```

### Fixture Usage
```python
# Using existing fixtures
def test_with_sample_data(sample_prompts, evolution_config):
    # Use fixtures directly
    assert len(sample_prompts) > 0
    assert evolution_config.population_size > 0

# Creating custom fixtures
@pytest.fixture
def custom_test_data():
    return {"key": "value", "number": 42}
```

### Mocking External Dependencies
```python
from unittest.mock import patch, MagicMock

@patch('meta_prompt_evolution.external.llm_client')
def test_with_mocked_llm(mock_client):
    """Test with mocked external service."""
    mock_client.complete.return_value = "mocked response"
    
    # Your test code here
    result = component_using_llm.process()
    
    assert result == "expected result"
    mock_client.complete.assert_called_once()
```

## Test Data Management

### Static Test Data
Store in `tests/fixtures/` directory:
- JSON files for API responses
- CSV files for datasets
- YAML files for configurations

### Dynamic Test Data
Use factories and builders:
```python
class PromptFactory:
    @staticmethod
    def create_prompt(text="Default prompt", **kwargs):
        return Prompt(text=text, **kwargs)
    
    @staticmethod
    def create_population(size=10):
        prompts = [PromptFactory.create_prompt(f"Prompt {i}") for i in range(size)]
        return PromptPopulation(prompts)
```

## Continuous Integration

### GitHub Actions Integration
Tests run automatically on:
- Pull request creation/updates
- Push to main branch
- Scheduled daily runs

### Test Matrix
- Python versions: 3.9, 3.10, 3.11, 3.12
- Operating systems: Ubuntu, macOS, Windows
- Dependencies: minimum, latest versions

### Performance Regression Detection
- Benchmark critical functions
- Compare against baseline performance
- Fail CI if performance degrades significantly

## Best Practices

### Test Organization
1. Group related tests in classes
2. Use descriptive test names
3. One assertion per test (when possible)
4. Test both success and failure cases

### Test Isolation
1. Each test should be independent
2. Reset state between tests
3. Use fixtures for shared setup
4. Clean up resources after tests

### Test Coverage
1. Aim for >95% line coverage on core modules
2. Test edge cases and error conditions
3. Include integration tests for critical paths
4. Document any untestable code

### Performance Testing
1. Set baseline performance benchmarks
2. Test with realistic data sizes
3. Monitor memory usage and leaks
4. Test concurrency and race conditions

## Debugging Tests

### Common Issues
```bash
# Verbose output for debugging
pytest -v -s

# Stop on first failure
pytest -x

# Debug with pdb
pytest --pdb

# Show local variables in failures
pytest -l
```

### Test Database Issues
```bash
# Reset test database
pytest --create-db --reset-db

# Check database connections
pytest -k "database" -v
```

### Async Test Issues
```bash
# Run only async tests
pytest -k "async" -v

# Check for event loop issues
pytest --asyncio-mode=auto
```

## Contributing Test Code

1. Write tests for all new functionality
2. Update tests when modifying existing code
3. Follow existing test patterns and conventions
4. Include both positive and negative test cases
5. Add performance tests for critical paths
6. Document complex test scenarios

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Hypothesis Property-Based Testing](https://hypothesis.readthedocs.io/)