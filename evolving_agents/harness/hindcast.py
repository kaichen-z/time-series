"""Carves past windows out of a series so hypotheses can be scored without touching ground truth."""

from __future__ import annotations

import statistics

from ..models import HindcastWindow

MIN_TRAIN_LENGTH = 20


def carve_hindcast_windows(
    history_values: tuple[float, ...],
    horizon: int,
    frequency: str,
    n_windows: int = 3,
    min_train_length: int = MIN_TRAIN_LENGTH,
) -> tuple[HindcastWindow, ...]:
    """Cut up to n_windows non-overlapping tails of length `horizon` off the end of the history.

    Uses only history_values, never the task's real future, so Loop A needs no labels at all.
    """
    if horizon <= 0 or not history_values:
        return ()
    windows: list[HindcastWindow] = []
    for index in range(n_windows):
        end = len(history_values) - index * horizon
        start = end - horizon
        if start - min_train_length < 0:
            break
        windows.append(
            HindcastWindow(
                train_history=history_values[:start],
                held_out_future=history_values[start:end],
                frequency=frequency,
            )
        )
    return tuple(windows)


def scaled_error(truth: tuple[float, ...], prediction: tuple[float, ...]) -> float:
    """Mean absolute error scaled by the mean magnitude of the truth, so series of any size compare."""
    if not truth or len(truth) != len(prediction):
        raise ValueError(f"truth/prediction length mismatch: {len(truth)} vs {len(prediction)}")
    magnitude = statistics.fmean(abs(value) for value in truth)
    error = statistics.fmean(abs(actual - predicted) for actual, predicted in zip(truth, prediction))
    return error / magnitude if magnitude > 1e-8 else error
