def damped_trend(history, horizon, frequency):
    """Use a damped recent trend for series with persistent local movement."""
    if not history:
        return [0.0] * horizon
    if len(history) < 2:
        return [history[-1]] * horizon
    window_size = min(len(history), 8)
    window = history[-window_size:]
    differences = [window[index] - window[index - 1] for index in range(1, len(window))]
    slope = sum(differences) / len(differences)
    scale = max(abs(window[-1]), 1.0)
    limit = scale * 0.2
    slope = max(-limit, min(limit, slope))
    forecasts = []
    level = window[-1]
    for step in range(horizon):
        level += slope
        slope *= 0.8
        forecasts.append(level)
    return forecasts
