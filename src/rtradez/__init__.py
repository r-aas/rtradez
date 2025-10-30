"""
RTradez - Comprehensive Options Trading Dataset Organization & Analysis Framework

A Python framework for organizing, processing, and analyzing options trading datasets
with focus on systematic strategy development and backtesting.
"""

__version__ = "0.1.0"
__author__ = "Russell Cox"
__email__ = "russell@appliedaisystems.com"

# Core imports for convenience
from . import datasets
from . import methods
from . import metrics
from . import tasks
from . import utils
from . import loaders
from . import benchmarks
from . import plotting
from . import callbacks

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
]