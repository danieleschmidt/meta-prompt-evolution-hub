# GitHub Actions Workflows Documentation

This directory contains documentation for GitHub Actions workflows that should be implemented for this repository. Since workflows require manual setup, these templates and documentation provide guidance for implementation.

## Required Workflows

### 1. Continuous Integration (ci.yml)

**Location**: `.github/workflows/ci.yml`

**Purpose**: Run tests, linting, and security checks on every push and pull request.

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, '3.10', 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests
      run: pytest tests/ --cov=meta_prompt_evolution --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Security Scanning (security.yml)

**Location**: `.github/workflows/security.yml`

**Purpose**: Run security scans and dependency vulnerability checks.

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Run Bandit security scan
      run: bandit -r meta_prompt_evolution/ -f json -o bandit-report.json
    
    - name: Run Safety vulnerability check
      run: safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

### 3. Release Automation (release.yml)

**Location**: `.github/workflows/release.yml`

**Purpose**: Automate package publishing to PyPI on tag creation.

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### 4. Dependency Updates (dependabot.yml)

**Location**: `.github/dependabot.yml`

**Purpose**: Automated dependency updates.

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "deps"
      include: "scope"
    reviewers:
      - "danieleschmidt"
    labels:
      - "dependencies"
      - "automated"
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "ci"
    reviewers:
      - "danieleschmidt"
    labels:
      - "github-actions"
```

## Implementation Steps

1. **Create Workflow Directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add CI Workflow**:
   - Copy the CI template above to `.github/workflows/ci.yml`
   - Customize Python versions and test commands as needed

3. **Add Security Workflow**:
   - Copy the security template to `.github/workflows/security.yml`
   - Configure security tool settings

4. **Set up Dependabot**:
   - Copy dependabot config to `.github/dependabot.yml`
   - Update reviewers and labels

5. **Configure Secrets**:
   - Add `PYPI_API_TOKEN` to repository secrets for releases
   - Configure any additional service tokens (Codecov, etc.)

## Branch Protection Rules

Recommended branch protection settings for `main` branch:

- **Require pull request reviews**: 1 required reviewer
- **Require status checks**: All CI checks must pass
- **Require branches be up to date**: Yes
- **Restrict pushes**: Only allow squash merging
- **Delete head branches**: Automatically delete PR branches

## Additional Recommendations

### Code Coverage
- Set up Codecov integration
- Require minimum 80% coverage
- Block PRs that decrease coverage significantly

### Performance Testing
- Add performance benchmarks to CI
- Track performance regression over time
- Use GitHub Actions cache for dependencies

### Documentation
- Auto-generate and deploy documentation
- Link check for documentation
- Spell check for markdown files

### Integration Testing
- Matrix testing across different OS (Ubuntu, macOS, Windows)
- Test against different Python versions
- Integration tests with external services

## Monitoring and Alerts

Set up notifications for:
- Failed CI builds
- Security vulnerabilities
- Dependency updates requiring attention
- Performance regressions

## Rollback Strategy

For release workflows:
- Tag-based releases with rollback capability
- Staged deployment (test PyPI first)
- Manual approval gates for production releases
- Version rollback procedures documented