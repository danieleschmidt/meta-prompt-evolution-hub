# Test Fixtures

This directory contains test data fixtures used across the test suite.

## Structure

- **prompts/**: Sample prompt data in various formats
- **datasets/**: Test datasets for evaluation
- **configs/**: Test configuration files
- **responses/**: Mock API responses
- **databases/**: Database fixtures and schemas

## Usage

Fixtures are automatically loaded by pytest through the conftest.py configuration.
Use fixtures in your test functions by including them as parameters:

```python
def test_example(sample_prompts, evolution_config):
    # Test implementation using fixtures
    pass
```

## Adding New Fixtures

1. Create data files in appropriate subdirectories
2. Add fixture functions to conftest.py
3. Document the fixture purpose and usage