"""Forecast-aware iterative retrieval system for Dr-CiK."""

from .loop import IterativeAgentSystem, LoopConfig
from .models import Document, ForecastTask
from .pipeline import MinimalAgentSystem, SystemConfig

__all__ = [
    "Document",
    "ForecastTask",
    "IterativeAgentSystem",
    "LoopConfig",
    "MinimalAgentSystem",
    "SystemConfig",
]
__version__ = "0.2.0"
