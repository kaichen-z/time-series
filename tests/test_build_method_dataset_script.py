from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_method_dataset_script_exposes_reproducible_release_pipeline() -> None:
    script = ROOT / "scripts/build_method_dataset.sh"

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "write_catalog_manifests" in text
    assert "build-dataset" in text
    assert "forecast_method_dataset_v001.json" in text
    assert "pytest" in text
