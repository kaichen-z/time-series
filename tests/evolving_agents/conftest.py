"""Shared fixtures and skip markers for the evolving_agents suite."""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_DIR = Path("/raid/home/air/khoutaibi/external/Dr-CiK/sample")
DATA_DIR = Path("/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK")

requires_sample = pytest.mark.skipif(not SAMPLE_DIR.is_dir(), reason="Dr-CiK sample dir not present on this machine")
requires_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="Dr-CiK full dataset not present on this machine")


@pytest.fixture(autouse=True)
def _tracing_off():
    """Keep tracing off by default so tests don't depend on global trace state leaking between them."""
    from evolving_agents.harness.trace import configure_tracing

    configure_tracing("off")
    yield
    configure_tracing("off")
