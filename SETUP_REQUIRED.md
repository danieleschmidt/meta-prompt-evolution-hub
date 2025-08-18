# Manual Setup Required

Due to GitHub App permission limitations, the following setup steps must be completed manually by repository maintainers.

## Required GitHub Actions Workflows

### 1. Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Templates

Copy the following files from `docs/workflows/examples/` to `.github/workflows/`:

- `ci.yml` → `.github/workflows/ci.yml`
- `cd.yml` → `.github/workflows/cd.yml`  
- `security-scan.yml` → `.github/workflows/security-scan.yml`

### 3. Required Repository Secrets

Configure the following secrets in GitHub repository settings:

#### Deployment Secrets
- `AWS_ACCESS_KEY_ID`: AWS access key for EKS deployment
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for EKS deployment
- `PYPI_API_TOKEN`: PyPI token for package publishing

#### Notification Secrets
- `SLACK_WEBHOOK_URL`: Slack webhook for deployment notifications
- `GRAFANA_URL`: Grafana instance URL for deployment annotations
- `GRAFANA_TOKEN`: Grafana API token for annotations

#### Security Scanning Secrets
- `GITLEAKS_LICENSE`: GitLeaks license key (if using pro features)

### 4. Configure Dependabot

Create `.github/dependabot.yml`:

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

### 5. Configure CodeQL

Create `.github/codeql/codeql-config.yml`:

```yaml
name: "Meta-Prompt-Evolution-Hub CodeQL Config"

disable-default-queries: false

queries:
  - name: security-extended
    uses: security-extended
  - name: security-and-quality
    uses: security-and-quality

paths-ignore:
  - tests/
  - docs/
  - "**/*.md"

languages:
  - python
```

### 6. Branch Protection Rules

Configure branch protection for `main` branch in GitHub repository settings:

- ✅ Require pull request reviews before merging
  - Required approving reviews: 1
  - Dismiss stale PR approvals when new commits are pushed
- ✅ Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Status checks that are required:
    - `lint-and-format`
    - `test (ubuntu-latest, 3.11)`
    - `security-scan`
    - `docker-build`
- ✅ Require signed commits
- ✅ Require conversation resolution before merging
- ✅ Restrict pushes that create matching branches

### 7. Repository Settings

#### General Settings
- Default branch: `main`
- Allow merge commits: ❌
- Allow squash merging: ✅
- Allow rebase merging: ❌
- Automatically delete head branches: ✅

#### Security Settings
- Dependency graph: ✅
- Dependabot alerts: ✅
- Dependabot security updates: ✅
- Secret scanning: ✅
- Push protection: ✅

#### Code Security and Analysis
- CodeQL analysis: ✅
- Private vulnerability reporting: ✅

### 8. Environments

Create the following environments in repository settings:

#### Staging Environment
- **Protection rules**: No required reviewers
- **Environment secrets**: None additional required
- **Variables**: 
  - `CLUSTER_NAME`: `meta-prompt-hub-staging`
  - `NAMESPACE`: `staging`

#### Production Environment  
- **Protection rules**: Required reviewers (1)
- **Wait timer**: 5 minutes
- **Environment secrets**: Production-specific secrets if needed
- **Variables**:
  - `CLUSTER_NAME`: `meta-prompt-hub-production`  
  - `NAMESPACE`: `production`

### 9. Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.md`:

```yaml
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment (please complete the following information):**
- OS: [e.g. iOS]
- Python version: [e.g. 3.11]
- Package version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
```

Create `.github/ISSUE_TEMPLATE/feature_request.md`:

```yaml
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: 'enhancement'
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

### 10. Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description

Brief description of changes made.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Performance testing performed (if applicable)

## Security

- [ ] Security implications reviewed
- [ ] No sensitive data exposed
- [ ] Authentication/authorization maintained
- [ ] Input validation implemented

## Documentation

- [ ] Code documentation updated
- [ ] README updated (if applicable)
- [ ] API documentation updated (if applicable)
- [ ] Migration guide provided (for breaking changes)

## Checklist

- [ ] Self-review of code completed
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] No new warnings introduced
- [ ] Backward compatibility maintained (or breaking change documented)

## Related Issues

Closes #(issue number)

## Screenshots (if applicable)

<!-- Add screenshots here if the changes affect the UI -->

## Additional Notes

<!-- Any additional information that reviewers should know -->
```

### 11. CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Global owners
* @danieleschmidt

# Core application code
/meta_prompt_evolution/ @danieleschmidt

# Infrastructure and deployment
/deployment/ @danieleschmidt
/docker-compose*.yml @danieleschmidt
/Dockerfile* @danieleschmidt

# CI/CD and workflows
/.github/ @danieleschmidt

# Documentation
/docs/ @danieleschmidt
README.md @danieleschmidt

# Configuration files
pyproject.toml @danieleschmidt
requirements*.txt @danieleschmidt
```

## Post-Setup Verification

After completing the manual setup:

1. **Test CI Pipeline**: Create a test branch and open a PR to verify workflows run
2. **Test Security Scanning**: Verify security scans execute and report findings
3. **Test Dependabot**: Verify automated dependency PRs are created
4. **Test Branch Protection**: Verify branch protection rules are enforced
5. **Test Deployment**: Verify staging deployment works (when infrastructure is ready)

## Infrastructure Prerequisites

Before enabling deployment workflows, ensure the following infrastructure exists:

- EKS clusters (`meta-prompt-hub-staging`, `meta-prompt-hub-production`)
- S3 bucket for backups (`meta-prompt-hub-backups`)
- IAM roles with appropriate permissions
- Kubernetes namespaces (`staging`, `production`)
- LoadBalancer or Ingress configuration
- SSL certificates for HTTPS endpoints

## Support

If you encounter issues during setup, please:

1. Check the workflow logs in GitHub Actions
2. Verify all secrets are configured correctly
3. Ensure branch protection rules match the workflow requirements
4. Contact the development team for assistance

---

**Note**: This setup documentation should be removed once all manual steps are completed.