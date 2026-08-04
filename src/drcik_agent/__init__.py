"""Forecast-aware iterative retrieval system for Dr-CiK."""

from .loop import IterativeAgentSystem, LoopConfig
from .impacts import EvidenceToForecastAgent
from .memory import ForecastMemoryBank
from .models import Document, EvidenceImpact, ForecastTask, ForecastWorkspace, RevisionAction
from .pipeline import MinimalAgentSystem, SystemConfig
from .workspace import ForecastWorkspaceExecutor, RevisionPlannerAgent

__all__ = [
    "Document",
    "EvidenceImpact",
    "EvidenceToForecastAgent",
    "ForecastTask",
    "ForecastWorkspace",
    "ForecastWorkspaceExecutor",
    "ForecastMemoryBank",
    "IterativeAgentSystem",
    "LoopConfig",
    "MinimalAgentSystem",
    "RevisionAction",
    "RevisionPlannerAgent",
    "SystemConfig",
]
__version__ = "0.4.0"
