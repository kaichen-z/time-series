"""Forecast-aware iterative retrieval system for Dr-CiK."""

from .loop import IterativeAgentSystem, LoopConfig
from .impacts import EvidenceToForecastAgent
from .models import Document, EvidenceImpact, ForecastTask
from .pipeline import MinimalAgentSystem, SystemConfig

__all__ = [
    "Document",
    "EvidenceImpact",
    "EvidenceToForecastAgent",
    "ForecastTask",
    "IterativeAgentSystem",
    "LoopConfig",
    "MinimalAgentSystem",
    "SystemConfig",
]
__version__ = "0.3.0"
