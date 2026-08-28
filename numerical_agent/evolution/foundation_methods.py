"""The five zero-shot foundation models, written as ordinary methods.

These are handed to the seed verbatim rather than written by the bootstrap model: an LLM cannot
guess each package's calling convention, and a wrong guess reads as a crashed method rather than
as the mistake it is. They are ordinary methods once seeded, so the evolution loop may rewrite
their preconditions and context limits, or delete them, like anything else.

Every model is loaded once per process by common.tsfm.shared_forecaster. The methods module
cannot cache anything itself: only its functions survive a rewrite.
"""


class NotApplicable(Exception):
    """Raised by a method whose stated preconditions the series does not meet."""


def chronos_bolt_zero_shot(history, horizon, frequency):
    """Use as the general-purpose zero-shot baseline on any series long enough to give the model
    context: a patch-based Chronos variant whose encoder produces all horizon steps at once, so
    it costs one forward pass regardless of horizon. Wins where a classical method would have to
    be fitted per series and there is no obvious parametric form. Caveat: fixed patch aggregation
    smooths narrow spikes, and it sees only the numbers, so it cannot anticipate an event."""
    from common.tsfm import shared_forecaster

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points of context, got {len(history)}")
    return list(shared_forecaster("chronos_bolt").forecast(list(history), horizon))


def chronos_2_zero_shot(history, horizon, frequency):
    """Use where Chronos-Bolt is the right idea but the series is long or the horizon far: the
    Chronos-2 checkpoint is trained for longer context and in-context covariates, and tends to
    hold its level better over long horizons. Caveat: it is the larger of the two Chronos models,
    so it costs more per task, and on short histories the extra capacity buys nothing."""
    from common.tsfm import shared_forecaster

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points of context, got {len(history)}")
    return list(shared_forecaster("chronos_2").forecast(list(history), horizon))


def timesfm_2_5_zero_shot(history, horizon, frequency):
    """Use on long histories where the series has structure a decoder can carry forward: TimesFM
    2.5 has the longest context of the five and a continuous quantile head, so it degrades
    gracefully as the horizon grows. Caveat: its continuous quantile head supports at most 1024
    steps, and it carries no frequency token, so it infers the period from the values alone."""
    from common.tsfm import shared_forecaster

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points of context, got {len(history)}")
    if horizon > 1024:
        raise NotApplicable(f"the continuous quantile head supports 1024 steps, got {horizon}")
    return list(shared_forecaster("timesfm_2_5").forecast(list(history), horizon))


def toto_zero_shot(history, horizon, frequency):
    """Use on observability-shaped series -- bursty, spiky, machine-generated -- which is what
    Toto was pretrained on, and where its sampled median beats a fitted model that has to choose
    one parametric shape. Caveat: it is a sampling forecaster, so the point forecast is a median
    over draws and costs far more compute per task than the encoder models; on a smooth,
    strongly seasonal series that expense buys little."""
    from common.tsfm import shared_forecaster

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points of context, got {len(history)}")
    return list(shared_forecaster("toto").forecast(list(history), horizon))


def moirai_zero_shot(history, horizon, frequency):
    """Use when the series comes from a domain unlike the others in the set: Moirai 2.0 was
    trained on the widest corpus of the five and predicts quantiles directly, so its median is
    robust where a mean-trained model chases outliers. Caveat: its own documentation reports
    decline at long horizons, and it rebuilds its forecast head per call, so it is slower than
    the Chronos models on short tasks."""
    from common.tsfm import shared_forecaster

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points of context, got {len(history)}")
    return list(shared_forecaster("moirai").forecast(list(history), horizon))
