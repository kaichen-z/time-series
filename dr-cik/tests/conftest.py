"""Shared fixtures for the offline dr-cik test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_DIR = Path("/raid/home/air/khoutaibi/external/Dr-CiK/sample")

requires_sample = pytest.mark.skipif(not SAMPLE_DIR.is_dir(), reason="Dr-CiK sample dir not present on this machine")


@pytest.fixture(scope="session")
def sample_tasks():
    from dr_cik.data import load_sample_tasks

    return load_sample_tasks(SAMPLE_DIR)
