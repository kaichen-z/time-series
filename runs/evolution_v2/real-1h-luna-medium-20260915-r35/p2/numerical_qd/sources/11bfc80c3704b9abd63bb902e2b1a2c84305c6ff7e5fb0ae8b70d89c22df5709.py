def damped_trend(history, horizon, frequency):
    """Use a damped recent trend for short noisy series."""
    if horizon <= 0:
        return []
    if not history:
        return [0.0] * horizon
    level = float(history[-1])
    if len(history) < 2:
        return [level] * horizon
    trend = float(history[-1]) - float(history[-2])
    forecasts = []
    for step in range(1, horizon + 1):
        damping = 1.0 / (1.0 + step)
        forecasts.append(level + trend * damping * step)
    return forecasts
