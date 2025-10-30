.PHONY: help install dev-install test lint format type-check clean docs serve-docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package dependencies
	uv sync

dev-install:  ## Install package in development mode with dev dependencies
	uv sync --dev
	uv run pre-commit install

test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov=src/rtradez --cov-report=html --cov-report=term

test-fast:  ## Run tests excluding slow tests
	uv run pytest -m "not slow"

lint:  ## Run linting (ruff)
	uv run ruff check .

lint-fix:  ## Run linting with auto-fix
	uv run ruff check --fix .

format:  ## Format code with black
	uv run black .

format-check:  ## Check code formatting
	uv run black --check .

type-check:  ## Run type checking with mypy
	uv run mypy src/

quality:  ## Run all quality checks (lint, format, type)
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) type-check

quality-fix:  ## Run all quality fixes
	$(MAKE) format
	$(MAKE) lint-fix

ci:  ## Run full CI pipeline (quality + tests)
	$(MAKE) quality
	$(MAKE) test

clean:  ## Clean up cache and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

docs:  ## Build documentation
	cd docs && uv run sphinx-build -b html . _build

serve-docs:  ## Serve documentation locally
	cd docs/_build && python -m http.server 8000

demo:  ## Run quick demo
	uv run examples/quick_start.py

validate:  ## Run strategy validation
	uv run examples/best_strategy_validation.py

cache-clear:  ## Clear RTradez cache
	rm -rf ~/.rtradez/cache/

# Development workflow targets
check: quality test  ## Run quality checks and tests

fix: quality-fix  ## Apply all automatic fixes

release-check:  ## Check if ready for release
	$(MAKE) clean
	$(MAKE) quality
	$(MAKE) test-cov
	@echo "✅ Release checks passed"

# Examples
examples:  ## Run all examples
	@echo "Running sklearn interface demo..."
	uv run examples/sklearn_interface_demo.py
	@echo "Running caching demo..."
	uv run examples/caching_and_optimization_demo.py