"""Reproductions of Dr-CiK's OpenDR/DRBench agents and its Chronos and Direct-Prompt forecasters.

The pipeline splits cleanly in two, mirroring how the benchmark itself scores:
`agents/` read a noisy document corpus and produce cited evidence, `forecasters/` produce
numbers, and `evaluation.py` scores each independently.
"""

from .models import AgentResult, Document, EvidenceItem, Forecast, ForecastTask, RunResult, TaskView

__all__ = ["AgentResult", "Document", "EvidenceItem", "Forecast", "ForecastTask", "RunResult", "TaskView"]
__version__ = "0.1.0"
