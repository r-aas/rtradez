# Test Coverage and Automated Testing

This document describes the comprehensive test coverage system implemented for RTradez.

## Overview

RTradez includes a robust testing framework with automated coverage reporting, multiple test categories, and CI/CD integration. The system is designed to ensure code quality and prevent regressions.

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Location**: `tests/test_*.py` 
- **Marker**: `@pytest.mark.unit`
- **Coverage**: Focus on single functions/classes

### Integration Tests  
- **Purpose**: Test component interactions and workflows
- **Location**: Integration test classes in test files
- **Marker**: `@pytest.mark.integration`
- **Coverage**: End-to-end workflows

### Performance Tests
- **Purpose**: Benchmark critical operations
- **Location**: Performance test classes 
- **Marker**: `@pytest.mark.slow`
- **Coverage**: Latency, memory, throughput

## Coverage Configuration

### Configuration Files

The coverage system is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/examples/*", 
    "*/__pycache__/*",
    "*/migrations/*",
    "*/venv/*",
    "*/env/*",
    "*/.venv/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "def __str__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "raise ValueError",
    "if 0:",
    "if False:",
    "if __name__ == .__main__.:",
    "pass",
    "...",
]
show_missing = true
skip_covered = false
precision = 2
fail_under = 80

[tool.coverage.html]
directory = "htmlcov"
title = "RTradez Test Coverage Report"

[tool.coverage.xml]
output = "coverage.xml"
```

### Pytest Configuration

Pytest is configured with coverage integration:

```toml
[tool.pytest.ini_options]
addopts = "-ra -q --strict-markers --strict-config --cov=src/rtradez --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=80"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

## Running Tests

### Command Line Interface

#### Using Make Commands

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only  
make test-integration

# Run tests with coverage reporting
make test-coverage

# Run quick tests (exclude slow tests)
make test-quick

# Generate detailed coverage report
make coverage

# Generate HTML coverage report
make coverage-html

# Generate XML coverage report
make coverage-xml

# Clean coverage files
make clean-coverage
```

#### Using Test Scripts

```bash
# Run all tests
./scripts/run_tests.sh all

# Run unit tests with verbose output
./scripts/run_tests.sh unit --verbose

# Run coverage analysis with custom threshold
./scripts/run_tests.sh coverage --threshold 85 --fail-under

# Run quick tests only
./scripts/run_tests.sh quick
```

#### Using Python/uv Directly

```bash
# Run all tests with coverage
uv run pytest --cov=src/rtradez --cov-report=html

# Run specific test file
uv run pytest tests/test_trading_benchmarks.py -v

# Run tests with specific marker
uv run pytest -m unit -v

# Run tests and fail if coverage below threshold
uv run pytest --cov-fail-under=80
```

### Coverage Analysis Script

The `scripts/test_coverage.py` script provides advanced coverage analysis:

```bash
# Basic coverage analysis
python scripts/test_coverage.py

# Coverage with custom threshold
python scripts/test_coverage.py --threshold 85

# Fail if coverage below threshold
python scripts/test_coverage.py --fail-under --threshold 80

# Generate JSON report
python scripts/test_coverage.py --output coverage_report.json

# Run only unit tests
python scripts/test_coverage.py --unit-only

# Verbose output
python scripts/test_coverage.py --verbose
```

## Coverage Reports

### HTML Report

Interactive HTML coverage report generated at `htmlcov/index.html`:

- Line-by-line coverage highlighting
- Branch coverage visualization  
- Package and module breakdown
- Missing line identification
- Search and filtering capabilities

### XML Report  

Machine-readable XML report for CI/CD integration at `coverage.xml`:

- Compatible with Codecov, CodeClimate
- Supports automated coverage badges
- Integrates with GitHub Actions

### JSON Report

Structured coverage data at `coverage_report.json`:

- Package-level statistics
- Overall coverage metrics
- Badge color recommendations
- Timestamp and metadata

### Terminal Report

Real-time coverage summary in terminal:

```
Name                                    Stmts   Miss Branch BrPart    Cover   Missing
-----------------------------------------------------------------------------------
src/rtradez/trading_benchmarks.py         150     10     45      5   92.15%   45-48
src/rtradez/portfolio/manager.py          200     25     60      8   87.50%   156-162
-----------------------------------------------------------------------------------
TOTAL                                     2150    165    580     48   89.25%
```

## CI/CD Integration

### GitHub Actions

The `.github/workflows/test-coverage.yml` workflow provides:

- Multi-platform testing (Ubuntu, macOS, Windows)
- Automated coverage reporting
- Pull request coverage comments
- Coverage badge generation
- Artifact archiving

### Workflow Features

1. **Coverage Threshold Enforcement**: Fails builds below 80% coverage
2. **Multi-Python Version Support**: Tests against Python 3.12
3. **Parallel Execution**: Unit and integration tests run in parallel
4. **Coverage Comments**: Automatic PR comments with coverage reports
5. **Artifact Storage**: Coverage reports archived for download

### Coverage Comments

Pull requests automatically receive coverage comments:

```markdown
## 🟢 Test Coverage Report

| Metric | Coverage |
|--------|----------|
| **Line Coverage** | 89.25% |
| **Branch Coverage** | 85.30% |
| **Lines Covered** | 1985/2150 |
| **Branches Covered** | 532/580 |

### Package Breakdown
| Package | Coverage |
|---------|----------|
| trading_benchmarks | 92.15% |
| portfolio | 87.50% |
| risk | 91.25% |

[📊 View detailed HTML report](link-to-artifacts)
```

## Coverage Thresholds

### Current Standards

- **Overall Coverage**: 80% minimum
- **Critical Modules**: 90% minimum  
- **New Code**: 95% minimum
- **Branch Coverage**: 75% minimum

### Per-Module Targets

| Module | Target Coverage |
|--------|----------------|
| Trading Benchmarks | 95% |
| Portfolio Management | 90% |
| Risk Management | 95% |
| Data Sources | 85% |
| CLI Interface | 80% |
| Utilities | 80% |

## Best Practices

### Writing Tests

1. **Test Structure**: Use AAA pattern (Arrange, Act, Assert)
2. **Fixture Usage**: Leverage pytest fixtures for setup
3. **Mocking**: Mock external dependencies appropriately  
4. **Markers**: Use appropriate test markers (unit/integration/slow)
5. **Documentation**: Include docstrings for complex test scenarios

### Coverage Optimization

1. **Focus on Logic**: Prioritize business logic coverage
2. **Edge Cases**: Test error conditions and edge cases
3. **Integration Paths**: Ensure critical workflows are covered
4. **Avoid Test Pollution**: Don't test framework code
5. **Meaningful Assertions**: Verify actual behavior, not just execution

### Exclusion Guidelines

Use `# pragma: no cover` sparingly for:

- Debug code
- Platform-specific fallbacks
- Defensive programming (should-never-happen cases)
- Abstract method definitions
- Import guards

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Missing Dependencies**: Run `uv sync --all-extras --dev`
3. **Coverage Not Detected**: Check source path configuration
4. **False Positives**: Review exclusion patterns
5. **Performance Issues**: Use test markers to skip slow tests

### Debug Commands

```bash
# Check test discovery
pytest --collect-only

# Run specific failing test
pytest tests/test_module.py::TestClass::test_method -v

# Debug coverage configuration
coverage debug config

# Check source detection
coverage debug data

# Generate coverage without running tests
coverage combine
coverage report
```

## Performance Considerations

### Test Execution Speed

- **Unit Tests**: < 30 seconds total
- **Integration Tests**: < 5 minutes total  
- **Full Suite**: < 10 minutes total
- **Coverage Analysis**: < 1 minute additional

### Optimization Strategies

1. **Parallel Execution**: Use `pytest-xdist` for parallel runs
2. **Test Markers**: Skip slow tests during development
3. **Fixture Scoping**: Use appropriate fixture scopes
4. **Mock Strategy**: Mock expensive operations
5. **Data Generation**: Use small, focused test data

## Future Enhancements

### Planned Features

1. **Mutation Testing**: Add mutation testing with `mutmut`
2. **Property Testing**: Integrate `hypothesis` for property-based tests
3. **Performance Regression**: Add performance regression detection
4. **Security Testing**: Integrate security vulnerability scanning
5. **Code Quality**: Add code complexity and maintainability metrics

### Integration Roadmap

1. **Q1**: Mutation testing implementation
2. **Q2**: Property-based testing framework
3. **Q3**: Performance regression detection
4. **Q4**: Advanced security scanning

---

For additional help or questions about the testing framework, please refer to the project documentation or open an issue.