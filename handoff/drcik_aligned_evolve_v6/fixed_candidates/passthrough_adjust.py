"""Host-pinned decision for raw numerical baselines: preserve the base forecast exactly."""

def adjust(view):
    return list(view["base_forecast"])
