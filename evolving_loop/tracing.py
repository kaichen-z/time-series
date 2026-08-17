"""Structured run logging; the implementation is shared, so it lives in common/."""
from __future__ import annotations

from common.tracing import (  # noqa: F401  (re-exported for existing importers)
    TraceEvent,
    configure,
    emit,
)
