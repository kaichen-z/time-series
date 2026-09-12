def naive_last(history, horizon, frequency):
    """Repeat the final observation across the requested horizon."""
    if len(history) < 1:
        raise NotApplicable("history must contain at least one observation")
    return [float(history[-1])] * horizon
