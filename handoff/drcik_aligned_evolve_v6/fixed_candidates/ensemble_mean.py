KEYS = ("toto_2_0", "timesfm_2_5", "moirai_2_0", "chronos_bolt")

def forecast(view):
    members = [view["method_forecasts"][k] for k in KEYS]
    return [sum(x) / len(x) for x in zip(*members)]
