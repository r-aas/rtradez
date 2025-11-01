"""
RTradez - Comprehensive Options Trading Dataset Organization & Analysis Framework

A Python framework for organizing, processing, and analyzing options trading datasets
with focus on systematic strategy development and backtesting.
"""

__version__ = "0.1.0"
__author__ = "Russell Cox"
__email__ = "russell@appliedaisystems.com"

# Core imports for convenience - with graceful error handling
try:
    from . import datasets
except ImportError:
    datasets = None

try:
    from . import methods
except ImportError:
    methods = None

try:
    from . import metrics
except ImportError:
    metrics = None

try:
    from . import tasks
except ImportError:
    tasks = None

try:
    from . import utils
except ImportError:
    utils = None

try:
    from . import loaders
except ImportError:
    loaders = None

try:
    from . import benchmarks
except ImportError:
    benchmarks = None

try:
    from . import plotting
except ImportError:
    plotting = None

try:
    from . import callbacks
except ImportError:
    callbacks = None

try:
    from . import data_sources
except ImportError:
    data_sources = None

try:
    from . import validation
except ImportError:
    validation = None

try:
    from . import research
except ImportError:
    research = None

try:
    from . import risk
except ImportError:
    risk = None

try:
    from . import pipeline
except ImportError:
    pipeline = None

try:
    from . import portfolio
except ImportError:
    portfolio = None

try:
    from . import cli
except ImportError:
    cli = None

__all__ = [
    "datasets",
    "methods", 
    "metrics",
    "tasks",
    "utils",
    "loaders",
    "benchmarks",
    "plotting",
    "callbacks",
    "data_sources",
    "validation", 
    "research",
    "risk", 
    "pipeline",
    "portfolio",
    "cli",
]