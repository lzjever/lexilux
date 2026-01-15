# Testing Documentation

This document describes the testing strategy and baseline metrics for Lexilux.

## Coverage Baseline

As of the start of the quality improvement initiative (Phase 0), the coverage baseline is:

**Overall Coverage Target**: 60% minimum (enforced in CI)

### Module Coverage Goals

| Module | Current Coverage | Target Coverage | Status |
|--------|------------------|-----------------|--------|
| `lexilux.chat` | TBD | 80% | Pending measurement |
| `lexilux.embed` | TBD | 80% | Pending measurement |
| `lexilux.rerank` | TBD | 80% | Pending measurement |
| `lexilux.tokenizer` | TBD | 75% | Pending measurement |
| `lexilux.usage` | TBD | 90% | Pending measurement |

## Running Tests

### Run all tests
```bash
make test
# or
uv run pytest tests/ -v
```

### Run with coverage
```bash
make test-cov
# or
uv run pytest tests/ --cov=lexilux --cov-report=html --cov-report=term-missing
```

### Run specific test file
```bash
uv run pytest tests/test_chat.py -v
```

### Run specific test function
```bash
uv run pytest tests/test_chat.py::test_chat_basic -v
```

## Test Organization

- **Unit Tests**: `tests/test_*.py` - Test individual components in isolation
- **Integration Tests**: Tests marked with `@pytest.mark.integration` - Test against external services
- **Benchmarks**: `tests/benchmarks/` - Performance regression tests

## CI Testing

The CI pipeline runs:
1. Linting (ruff check)
2. Format checking (ruff format --check)
3. Unit tests with coverage (Python 3.8-3.14)
4. Security scanning (pip-audit, bandit)

## Coverage Requirements

- All new code must have test coverage
- Critical paths (error handling, validation) should have >90% coverage
- Minimum 60% overall coverage required for CI to pass

## Next Steps

1. ✅ Add coverage threshold to pyproject.toml
2. 🔄 Measure actual coverage per module
3. ⏳ Increase coverage to 80% for core modules
4. ⏳ Add integration tests with mock servers
