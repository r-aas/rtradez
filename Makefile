# RTradez Development Makefile
#
# This Makefile provides convenient commands for development tasks
# including testing, coverage, linting, and documentation.

.PHONY: help install test test-unit test-integration test-coverage test-quick
.PHONY: lint format type-check clean clean-coverage docs
.PHONY: build publish dev-setup ci

# Default target
help:
	@echo "RTradez Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install project dependencies with uv"
	@echo "  dev-setup        Full development environment setup"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage    Run tests with coverage reporting"
	@echo "  test-quick       Run quick tests (exclude slow tests)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and ruff"
	@echo "  type-check       Run mypy type checking"
	@echo ""
	@echo "Coverage:"
	@echo "  coverage         Generate detailed coverage report"
	@echo "  coverage-html    Generate HTML coverage report"
	@echo "  coverage-xml     Generate XML coverage report"
	@echo "  clean-coverage   Clean coverage files"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            Clean build artifacts and cache"
	@echo "  docs             Build documentation"
	@echo "  build            Build package"
	@echo "  ci               Run CI pipeline locally"

# Installation and setup
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync --all-extras --dev

dev-setup: install
	@echo "🛠️  Setting up development environment..."
	@echo "📝 Installing pre-commit hooks..."
	uv run pre-commit install || echo "⚠️  pre-commit not available, skipping hooks"
	@echo "✅ Development environment ready!"

# Testing targets
test:
	@echo "🧪 Running all tests..."
	./scripts/run_tests.sh all

test-unit:
	@echo "🧪 Running unit tests..."
	./scripts/run_tests.sh unit

test-integration:
	@echo "🔗 Running integration tests..."
	./scripts/run_tests.sh integration

test-coverage:
	@echo "📊 Running tests with coverage..."
	./scripts/run_tests.sh coverage --html --xml --json

test-quick:
	@echo "⚡ Running quick tests..."
	./scripts/run_tests.sh quick

# Coverage targets
coverage:
	@echo "📊 Generating detailed coverage report..."
	uv run python scripts/test_coverage.py --verbose --output coverage_report.json

coverage-html:
	@echo "📊 Generating HTML coverage report..."
	uv run pytest --cov=src/rtradez --cov-report=html

coverage-xml:
	@echo "📊 Generating XML coverage report..."
	uv run pytest --cov=src/rtradez --cov-report=xml

clean-coverage:
	@echo "🧹 Cleaning coverage files..."
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f coverage_report.json
	rm -f .coverage

# Code quality targets
lint:
	@echo "🔍 Running linting checks..."
	@echo "  📝 Running ruff..."
	uv run ruff check src/ tests/
	@echo "  🎨 Running black check..."
	uv run black --check src/ tests/
	@echo "  🔍 Running mypy..."
	uv run mypy src/

format:
	@echo "🎨 Formatting code..."
	@echo "  🎨 Running black..."
	uv run black src/ tests/
	@echo "  📝 Running ruff fix..."
	uv run ruff check --fix src/ tests/

type-check:
	@echo "🔍 Running type checking..."
	uv run mypy src/

# Maintenance targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docs:
	@echo "📚 Building documentation..."
	@echo "⚠️  Documentation build not yet configured"

build:
	@echo "📦 Building package..."
	uv build

publish: build
	@echo "🚀 Publishing package..."
	@echo "⚠️  Publishing not yet configured"

# CI pipeline
ci: clean install lint test-coverage
	@echo "✅ CI pipeline completed successfully!"

# Development workflow targets
dev-test: format lint test-unit
	@echo "✅ Development test cycle completed!"

quick-check: format lint test-quick
	@echo "✅ Quick check completed!"