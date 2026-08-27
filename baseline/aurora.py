"""Aurora as a sampling baseline; it is natively probabilistic, so no paths are synthesized."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Sequence

from common.tsfm import BackboneUnavailableError, resolve_device

from .forecasters import seasonal_period

# Aurora reads the series in tokens; the card recommends the token length track the period, so
# a task's own seasonality drives it rather than one hardcoded number.
DEFAULT_TOKEN_LEN = 24
MAX_TOKEN_LEN = 96


@dataclass(frozen=True)
class AuroraConfig:
    repo_id: str = "DecisionIntelligence/Aurora"
    device: str = "auto"
    max_context: int = 6656

    def __post_init__(self) -> None:
        if self.max_context <= 0:
            raise ValueError("Aurora context limit must be positive")


class AuroraForecaster:
    """Zero-shot Aurora 0.2B, returning the ensemble its generate() already produces."""

    name = "aurora"

    def __init__(self, config: AuroraConfig | None = None, token_len: int | None = None) -> None:
        self.config = config or AuroraConfig()
        self.token_len = token_len
        self._model: Any | None = None

    def _ensure_model(self) -> tuple[Any, Any]:
        try:
            torch = importlib.import_module("torch")
            aurora = importlib.import_module("aurora")
        except ImportError as error:
            raise BackboneUnavailableError(
                "Aurora is the configured backbone but aurora-model is not installed. "
                "Install it with: pip install aurora-model==0.2.0"
            ) from error
        if self._model is None:
            try:
                self._model = aurora.load_model(
                    repo_id=self.config.repo_id, device=resolve_device(self.config.device)
                )
                # load_model hands back a model still in training mode. Left that way, BatchNorm
                # rejects a single series ("expected more than 1 value per channel") and the
                # generated horizon is silently truncated to fewer steps than asked for.
                self._model.eval()
            except Exception as error:
                raise BackboneUnavailableError(
                    f"Could not load Aurora checkpoint {self.config.repo_id!r}: {error}"
                ) from error
        return self._model, torch

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        model, torch = self._ensure_model()
        context = [float(value) for value in history[-self.config.max_context :]]
        if not context:
            raise ValueError("cannot forecast from an empty history")
        device = resolve_device(self.config.device)
        try:
            inputs = torch.tensor(context, dtype=torch.float32).reshape(1, -1).to(device)
            with torch.no_grad():
                predicted = model.generate(
                    inputs=inputs,
                    max_output_length=horizon,
                    num_samples=samples,
                    inference_token_len=self.token_len or DEFAULT_TOKEN_LEN,
                )
        except Exception as error:
            raise BackboneUnavailableError(f"Aurora inference failed: {error}") from error

        # (batch, samples, horizon); one series in, so the batch dimension is dropped.
        paths = [[float(value) for value in path] for path in predicted[0].tolist()]
        if len(paths) != samples or any(len(path) != horizon for path in paths):
            raise BackboneUnavailableError(
                f"Aurora returned {len(paths)} paths of length "
                f"{len(paths[0]) if paths else 0}, expected {samples} of {horizon}"
            )
        return tuple(tuple(path) for path in paths)


def token_length_for(frequency: str, seasonal_period_field: object) -> int:
    """Token length for one task, tracking its period as the model card recommends."""
    period = seasonal_period(frequency, seasonal_period_field)
    return min(period, MAX_TOKEN_LEN) if period > 1 else DEFAULT_TOKEN_LEN
