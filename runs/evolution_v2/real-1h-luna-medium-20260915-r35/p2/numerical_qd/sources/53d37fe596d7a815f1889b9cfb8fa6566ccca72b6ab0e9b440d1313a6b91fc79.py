def recent_mean(history, horizon, frequency):
    """Use a recent local level for short noisy series."""
    if not history:
        return [0.0] * horizon
    window = history[-min(len(history), 8):]
    level = sum(window) / len(window)
    return [level] * horizon
