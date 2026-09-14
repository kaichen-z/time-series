def damped_trend(history, horizon, frequency):
    """Use a recent level with a damped local trend for persistent movement."""
    if not history:
        return [0.0] * horizon
    if len(history) == 1:
        return [float(history[-1])] * horizon
    window_size = min(len(history), 8)
    window = history[-window_size:]
    level = sum(window) / len(window)
    slope = (window[-1] - window[0]) / max(1, len(window) - 1)
    forecasts = []
    for step in range(1, horizon + 1):
        damping = 1.0 / (1.0 + 0.15 * step)
        forecasts.append(level + slope * step * damping)
    return forecasts
