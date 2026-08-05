"""Forecast-aware iterative retrieval system for Dr-CiK."""

from .loop import IterativeAgentSystem, LoopConfig
from .backbones import (
    BackboneUnavailableError,
    StatisticalForecastBackbone,
    TimesFMBackboneConfig,
    TimesFMForecastBackbone,
)
from .context import (
    ForecastUtilityRetriever,
    ImportanceAwareContextAgent,
    RetrievalProcessRewardAgent,
)
from .control import ForecastGapControllerAgent
from .forecast_utility import ForecastUtilityLabel, ForecastUtilityLabeler
from .impacts import EvidenceToForecastAgent
from .memory import ForecastMemoryBank
from .models import Document, EvidenceImpact, ForecastTask, ForecastWorkspace, RevisionAction
from .pipeline import MinimalAgentSystem, SystemConfig
from .reasoning import MacroReasoningAgent, MicroReasoningAgent, RevisionUtilityAgent
from .workspace import ForecastWorkspaceExecutor, RevisionPlannerAgent

__all__ = [
    "Document",
    "BackboneUnavailableError",
    "EvidenceImpact",
    "EvidenceToForecastAgent",
    "ForecastTask",
    "ForecastGapControllerAgent",
    "ForecastUtilityLabel",
    "ForecastUtilityLabeler",
    "ForecastUtilityRetriever",
    "ForecastWorkspace",
    "ForecastWorkspaceExecutor",
    "ForecastMemoryBank",
    "IterativeAgentSystem",
    "ImportanceAwareContextAgent",
    "LoopConfig",
    "MinimalAgentSystem",
    "MacroReasoningAgent",
    "MicroReasoningAgent",
    "RetrievalProcessRewardAgent",
    "RevisionAction",
    "RevisionPlannerAgent",
    "RevisionUtilityAgent",
    "SystemConfig",
    "StatisticalForecastBackbone",
    "TimesFMBackboneConfig",
    "TimesFMForecastBackbone",
]
__version__ = "0.7.0"
