def damped_trend(history, horizon, frequency):
    """Use a recent level with a damped local trend for evolving series."""
    if not history:
        return [0.0] * horizon
    if len(history) == 1:
        return [float(history[-1])] * horizon
    window_size = min(len(history), 8)
    window = history[-window_size:]
    level = float(window[-1])
    slope = (float(window[-1]) - float(window[0])) / max(1, len(window) - 1)
    forecasts = []
    for step in range(1, horizon + 1):
        damping = 0.85 ** (step - 1)
        forecasts.append(level + slope * damping * step)
    return forecasts
