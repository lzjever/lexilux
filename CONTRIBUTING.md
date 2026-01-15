# Contributing to Lexilux

Thank you for your interest in contributing to Lexilux!

## Development Setup

### Quick Start

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Set up development environment**:
   ```bash
   make dev-install
   ```

3. **Run tests**:
   ```bash
   make test
   ```

That's it! You're ready to contribute.

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Development Workflow

### Standard Development

For active development where you need to import and use lexilux:

```bash
make dev-install  # Installs package + all dependencies
make test         # Run tests
make lint         # Check code quality
make format       # Format code
```

### CI/CD or Code Review

If you only need development tools (linting, formatting) without installing the package:

```bash
make setup-venv   # Only installs dependencies, not the package
make lint         # Can still run linting
make format-check # Can still check formatting
```

**Note**: Some tests may require the package to be installed.

## Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Run checks**:
   ```bash
   make check  # Runs lint, format-check, and tests
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "Add feature: description"
   ```

5. **Push and create a pull request**

## Code Quality

- **Linting**: `make lint` (uses ruff)
- **Formatting**: `make format` (uses ruff)
- **Type checking**: `mypy` (optional, not enforced in CI)
- **Tests**: `make test` (uses pytest)
- **Pre-commit hooks**: `make pre-commit-install` (recommended)

All checks must pass before submitting a PR.

### Code Style Guidelines

Lexilux follows PEP 8 with these specifics:

- **Line length**: Maximum 100 characters
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings
- **Import order**: stdlib → third-party → local

Example:
```python
from __future__ import annotations

from typing import Any

import requests

from lexilux.exceptions import APIError


def make_request(self, endpoint: str, payload: dict[str, Any]) -> requests.Response:
    """
    Send POST request to API endpoint.

    Args:
        endpoint: API endpoint (e.g., "chat/completions").
        payload: Request body as dict.

    Returns:
        requests.Response object.

    Raises:
        APIError: On request failure.
    """
    ...
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>: <description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat: add connection pooling support
fix: correct timeout handling for streaming requests
docs: update API reference
test: add tests for exception hierarchy
```

### Pre-commit Hooks

Recommended for automatic quality checks:

```bash
# Install hooks
make pre-commit-install

# Run hooks manually
make pre-commit-run

# Update hooks
make pre-commit-update
```

Hooks check:
- ruff lint
- ruff format
- trailing whitespace
- file endings
- YAML syntax

## Testing

### Run all tests:
```bash
make test
```

### Run with coverage:
```bash
make test-cov
# View HTML report: open htmlcov/index.html
```

### Run integration tests (requires external services):
```bash
make test-integration
```

### Coverage Goals

- **Overall**: Minimum 60%
- **Core modules** (Chat, BaseAPIClient): >80%
- **Utility modules**: >70%

### Test Structure

```python
import pytest
from lexilux import Chat


class TestChatClient:
    """Test Chat client functionality."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        chat = Chat(
            base_url="https://api.example.com/v1",
            api_key="test-key",
        )
        assert chat.api_key == "test-key"

    def test_timeout_property_backward_compat(self):
        """Test timeout_s property provides backward compatibility."""
        chat = Chat(base_url="https://api.example.com/v1", timeout_s=30.0)
        assert chat.timeout_s == 30.0
```

## Documentation

### Build documentation:
```bash
make docs
```

Documentation is built using Sphinx. See `docs/` directory for source files.

## Project Structure

- `lexilux/` - Main package code
- `tests/` - Test files
- `docs/` - Documentation source
- `examples/` - Example code
- `pyproject.toml` - Project configuration and dependencies

## Questions?

Feel free to open an issue or start a discussion!

## Pull Request Guidelines

### PR Workflow

1. **Create a branch** from `main` or `develop`:
   ```bash
   git checkout -b feature/your-feature-name
   # Or: git checkout -b fix/your-bug-fix
   ```

2. **Make changes** following the guidelines above

3. **Run checks**:
   ```bash
   make check  # Runs lint, format check, and tests
   ```

4. **Commit changes** with clear messages (see Commit Message Format above)

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request** on GitHub

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally (`make test`)
- [ ] Coverage meets minimum requirements (60%)
- [ ] Linting passes (`make lint`)
- [ ] Formatting passes (`make format`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow format

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed:
- Unit tests added/updated
- Manual testing performed
- All tests pass

## Checklist
- [ ] My code follows style guidelines
- [ ] I have performed self-review
- [ ] I have commented my code where necessary
- [ ] I have updated documentation accordingly
- [ ] My changes generate no new warnings
```

## Reporting Bugs

### Bug Report Template

```markdown
**Description**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Python version: ...
2. Lexilux version: ...
3. Code example:
```python
...
```

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened. Include logs/error messages.

**Environment**
- Python version: ...
- Lexilux version: ...
- OS: ...
- Dependencies: ...

**Additional Context**
Any other relevant information, screenshots, or examples.
```

## Requesting Features

### Feature Request Template

```markdown
**Feature Description**
A clear and concise description of the feature.

**Motivation / Use Case**
Why is this feature needed? What problem does it solve?
Provide real-world examples if possible.

**Proposed Solution**
How should this feature work? Include API examples if applicable.

**Alternatives Considered**
What alternative solutions did you consider? Why were they rejected?

**Additional Context**
Any other relevant information, examples, or references.
```

## License

By contributing to Lexilux, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing! 🎉

