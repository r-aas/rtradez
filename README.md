# RTradez

**Comprehensive Options Trading Dataset Organization & Analysis Framework**

RTradez is a Python framework for organizing, processing, and analyzing options trading datasets with focus on systematic strategy development and backtesting.

## 🚀 Features

- **📊 Dataset Management** - Standardized options data ingestion and storage
- **⚡ Trading Methods** - Implementation of various options strategies with sklearn-like interface
- **📈 Performance Metrics** - Comprehensive evaluation tools and risk metrics
- **🔄 Task Automation** - Streamlined workflow management and backtesting pipelines
- **🛠️ Utilities** - Common trading calculations, Greeks, and pricing models
- **📂 Data Loaders** - Efficient data loading and preprocessing with caching
- **🏆 Benchmarks** - Performance comparison frameworks with cross-validation
- **📊 Plotting** - Advanced visualization tools for strategy analysis
- **⚙️ Callbacks** - Event-driven strategy monitoring and experiment tracking

## 🏗️ Project Structure

```
rtradez/
├── src/rtradez/
│   ├── datasets/     # Data ingestion, storage, and management
│   ├── methods/      # Trading strategies and algorithms
│   ├── metrics/      # Performance evaluation and risk metrics
│   ├── tasks/        # Automated workflow and task management
│   ├── utils/        # Common utilities and calculations
│   ├── loaders/      # Data loading and preprocessing
│   ├── benchmarks/   # Strategy comparison frameworks
│   ├── plotting/     # Visualization and charting tools
│   └── callbacks/    # Event handlers and monitoring
├── tests/           # Test suite
├── examples/        # Usage examples and tutorials
├── docs/           # Documentation
└── data/           # Local data storage
```

## ⚡ Quick Start

```bash
# Install dependencies
uv add pandas numpy scipy matplotlib seaborn plotly yfinance
uv add pytest pytest-cov black ruff

# Install rtradez in development mode
uv pip install -e .

# Run tests
uv run pytest

# Example usage
python -c "import rtradez; print('RTradez is ready!')"
```

## 📊 Core Modules

### 📊 Datasets
- Options chain data management
- Historical volatility surfaces
- Market data standardization
- Data quality validation

### ⚡ Methods
- **Directional Strategies**: Calls, Puts, Spreads
- **Neutral Strategies**: Straddles, Strangles, Iron Condors
- **Volatility Strategies**: Calendar spreads, Ratio spreads
- **Greeks-based Strategies**: Delta hedging, Gamma scalping

### 📈 Metrics
- **P&L Analysis**: Realized/Unrealized gains, drawdowns
- **Risk Metrics**: VaR, CVaR, Sharpe ratio, Sortino ratio
- **Options-specific**: Theta decay, Vega exposure, Gamma risk
- **Portfolio Metrics**: Correlation, diversification, allocation

### 🔄 Tasks
- Data collection automation
- Strategy backtesting pipelines
- Performance reporting
- Risk monitoring alerts

### 🛠️ Utils
- Black-Scholes pricing
- Greeks calculations
- Implied volatility computation
- Time decay modeling

## 🧠 Trading Strategy Framework

RTradez implements a sklearn-compatible interface for all strategies and components:

```python
from rtradez.methods import OptionsStrategy
from rtradez.datasets import OptionsDataset
from rtradez.metrics import PerformanceAnalyzer

# Load data
dataset = OptionsDataset.from_source('yahoo', 'SPY')

# Define strategy with sklearn-like interface
strategy = OptionsStrategy('iron_condor', 
                          profit_target=0.35,
                          stop_loss=2.0,
                          dte_range=(20, 40))

# Fit strategy (parameter optimization)
strategy.fit(dataset.features, dataset.returns)

# Predict signals
signals = strategy.predict(dataset.features)

# Score performance
sharpe_ratio = strategy.score(dataset.features, dataset.returns)

# Analyze performance
analyzer = PerformanceAnalyzer(strategy.results_)
print(f"Sharpe Ratio: {analyzer.sharpe_ratio():.2f}")
print(f"Max Drawdown: {analyzer.max_drawdown():.2%}")
```

## 🏆 Best Strategy Results

Based on comprehensive validation across multiple symbols and time periods:

**🥇 Iron Condor Strategy**
- **Best Market**: IWM (Russell 2000)
- **Expected Returns**: 11.68% annually
- **Sharpe Ratio**: 2.057
- **Overall Score**: 0.307

**Optimized Parameters:**
- Profit Target: 36.2%
- Stop Loss: 3.85x
- Put Strike Distance: 12 points OTM
- Call Strike Distance: 10 points OTM

## 📊 Visualization

```python
from rtradez.plotting import StrategyVisualizer, VolatilitySurface

# Plot strategy performance
viz = StrategyVisualizer(results)
viz.plot_pnl_evolution()
viz.plot_greeks_exposure()
viz.plot_risk_metrics()

# Visualize volatility surface
vol_surface = VolatilitySurface(dataset)
vol_surface.plot_3d()
vol_surface.plot_term_structure()
```

## ⚙️ Advanced Features

### 🚀 Intelligent Caching
- **5566x speedup** for repeated operations
- Automatic cache invalidation
- Disk-based persistence

### 🔬 Experiment Tracking
- MLflow integration
- Parameter logging
- Performance metrics tracking
- Result visualization

### 🎯 Hyperparameter Optimization
- Optuna Bayesian optimization
- TPE sampler with pruning
- Cross-validation integration
- Multi-objective optimization

### 🧪 Validation Framework
```bash
# Run comprehensive strategy validation
uv run examples/best_strategy_validation.py

# Generate analysis reports
uv run examples/validation_summary.py
```

## 🔧 Development

```bash
# Setup development environment
uv sync --dev

# Run linting
uv run ruff check .
uv run black --check .

# Run tests with coverage
uv run pytest --cov=rtradez --cov-report=html

# Build documentation
uv run sphinx-build docs docs/_build
```

## 📚 Documentation

- **[Validation Guide](HOW_TO_VALIDATE.md)** - Complete strategy validation methodology
- **[Quick Reference](QUICK_REFERENCE.txt)** - Best strategy summary
- **[API Reference](docs/)** - Detailed module documentation

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

**Built for systematic options trading with Python** 🐍
