# Contributing to RTradez

Welcome to the RTradez team! This guide will help you get up to speed and contributing quickly.

## 🚀 Quick Onboarding (30 minutes)

### 1. Environment Setup (5 minutes)
```bash
# Clone the repository
git clone <repo-url>
cd rtradez

# Install Python dependencies
uv sync --dev

# Verify installation
uv run python -c "import rtradez; print('RTradez ready!')"
```

### 2. Run Tests (5 minutes)
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=rtradez --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 3. Try Examples (10 minutes)
```bash
# Quick strategy demo
uv run examples/quick_start.py

# sklearn interface demo
uv run examples/sklearn_interface_demo.py

# Validation demo (may take longer)
uv run examples/caching_and_optimization_demo.py
```

### 4. Code Quality (5 minutes)
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

### 5. Make Your First Contribution (5 minutes)
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes...
# Add tests...

# Commit with conventional format
git commit -m "feat: add new strategy implementation"

# Push and create PR
git push origin feature/your-feature
```

## 🏗️ Project Architecture

### Core Design Principles
1. **sklearn-compatible interface** - All strategies implement `.fit()`, `.predict()`, `.score()`
2. **Modular design** - Each module has clear responsibilities
3. **Performance first** - Intelligent caching and vectorized operations
4. **Test-driven** - Comprehensive test coverage
5. **Documentation** - Clear examples and API docs

### Module Structure
```
src/rtradez/
├── base.py           # Base classes for all components
├── datasets/         # Data management and preprocessing
├── methods/          # Trading strategies and algorithms
├── metrics/          # Performance evaluation tools
├── utils/            # Caching, optimization, experiments
├── loaders/          # Data loading utilities
├── benchmarks/       # Strategy comparison frameworks
├── plotting/         # Visualization tools
└── callbacks/        # Event handlers and monitoring
```

### Key Concepts
- **BaseStrategy**: All trading strategies inherit from this
- **OptionsDataset**: Standardized data container
- **PerformanceAnalyzer**: Comprehensive metrics calculation
- **RTradezCache**: Intelligent caching system
- **RTradezOptimizer**: Hyperparameter tuning with Optuna

## 🧪 Development Workflow

### Adding a New Strategy

1. **Create strategy class**:
```python
# src/rtradez/methods/strategies.py
class MyNewStrategy(BaseStrategy):
    def __init__(self, param1=0.5, param2=10):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # Parameter optimization logic
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Signal generation logic
        return signals
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        # Performance scoring (default: Sharpe ratio)
        return self._calculate_sharpe_ratio(X, y)
```

2. **Add tests**:
```python
# tests/test_strategies.py
def test_my_new_strategy():
    strategy = MyNewStrategy(param1=0.3, param2=15)
    
    # Test sklearn interface
    assert hasattr(strategy, 'fit')
    assert hasattr(strategy, 'predict')
    assert hasattr(strategy, 'score')
    
    # Test with sample data
    X, y = create_sample_data()
    strategy.fit(X, y)
    signals = strategy.predict(X)
    score = strategy.score(X, y)
    
    assert len(signals) == len(X)
    assert isinstance(score, float)
```

3. **Add example**:
```python
# examples/my_new_strategy_demo.py
from rtradez.methods import MyNewStrategy
from rtradez.datasets import OptionsDataset

# Load data
dataset = OptionsDataset.from_source('yahoo', 'SPY')

# Initialize strategy
strategy = MyNewStrategy(param1=0.4, param2=12)

# Fit and evaluate
strategy.fit(dataset.features, dataset.returns)
score = strategy.score(dataset.features, dataset.returns)
print(f"Strategy score: {score:.3f}")
```

### Code Quality Standards

#### Code Formatting
- Use **Black** for formatting: `uv run black .`
- Line length: 88 characters
- Use type hints for all public methods

#### Linting
- Use **Ruff** for linting: `uv run ruff check .`
- Follow PEP 8 guidelines
- No unused imports or variables

#### Type Checking
- Use **MyPy** for type checking: `uv run mypy src/`
- All public methods must have type hints
- Use `Optional` and `Union` appropriately

#### Documentation
- Use Google-style docstrings
- Include examples in docstrings
- Update README.md for major features

### Testing Strategy

#### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-module functionality
3. **Performance Tests**: Caching and optimization
4. **Example Tests**: Ensure examples run successfully

#### Test Structure
```python
def test_feature_with_valid_input():
    """Test normal operation with valid inputs."""
    pass

def test_feature_with_invalid_input():
    """Test error handling with invalid inputs."""
    with pytest.raises(ValueError):
        pass

def test_feature_edge_cases():
    """Test boundary conditions and edge cases."""
    pass
```

#### Coverage Requirements
- Minimum 90% code coverage
- All public methods must be tested
- Critical paths require multiple test cases

## 🔄 Git Workflow

### Branch Naming
- `feature/strategy-name` - New strategies
- `fix/bug-description` - Bug fixes
- `docs/topic` - Documentation updates
- `refactor/component` - Code refactoring

### Commit Messages
Use conventional commit format:
```
feat: add iron butterfly strategy
fix: resolve caching race condition
docs: update validation guide
test: add comprehensive strategy tests
refactor: simplify base strategy interface
```

### Pull Request Process
1. Create feature branch from `main`
2. Make changes with tests
3. Ensure all tests pass
4. Update documentation
5. Create PR with clear description
6. Request review from team
7. Address feedback
8. Merge after approval

## 📊 Performance Guidelines

### Caching Strategy
- Use `@cached` decorator for expensive operations
- Cache market data, feature calculations, backtest results
- Set appropriate cache expiration times

### Optimization Best Practices
- Vectorize operations with NumPy/Pandas
- Use Optuna for hyperparameter tuning
- Implement early stopping for long optimizations
- Profile code for bottlenecks

### Memory Management
- Use generators for large datasets
- Clean up temporary variables
- Monitor memory usage in tests

## 🐛 Debugging and Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure proper installation
   uv pip install -e .
   ```

2. **Test Failures**:
   ```bash
   # Run specific test
   uv run pytest tests/test_strategies.py::test_iron_condor -v
   ```

3. **Performance Issues**:
   ```bash
   # Profile code
   uv run python -m cProfile examples/strategy_demo.py
   ```

4. **Cache Issues**:
   ```bash
   # Clear cache
   rm -rf ~/.rtradez/cache/
   ```

### Debugging Tools
- Use `pdb` for interactive debugging
- Add logging with appropriate levels
- Use `pytest -s` to see print statements
- Profile with `cProfile` for performance issues

## 📚 Resources

### Documentation
- [Validation Guide](HOW_TO_VALIDATE.md) - Complete strategy validation
- [Quick Reference](QUICK_REFERENCE.txt) - Best strategy summary
- [API Documentation](docs/) - Detailed module docs

### External Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML pipelines
- **optuna**: Hyperparameter optimization
- **mlflow**: Experiment tracking
- **vectorbt**: Backtesting framework

### Team Communication
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Code Reviews: All changes require review

## 🎯 Next Steps

After completing onboarding:

1. **Pick an issue**: Look at GitHub issues labeled "good first issue"
2. **Choose a strategy**: Implement a new options strategy
3. **Improve documentation**: Add examples or clarify docs
4. **Optimize performance**: Profile and improve bottlenecks
5. **Add visualizations**: Create new plotting utilities

Welcome to the team! 🚀