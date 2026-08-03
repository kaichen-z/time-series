"""Minimal forecast-aware retrieval system for Dr-CiK."""

from .models import Document, ForecastTask
from .pipeline import MinimalAgentSystem, SystemConfig

__all__ = ["Document", "ForecastTask", "MinimalAgentSystem", "SystemConfig"]
__version__ = "0.1.0"

