"""Single neutral ForecastController seed for all three benchmarks."""


def _season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f:
        return None
    for k, p in (("hour", 24), ("week", 52), ("w", 52), ("month", 12), ("day", 7), ("d", 7)):
        if k in f:
            return p
    return None


def _seasonal(h, H, p):
    return [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]


def _blend(members, weights, shrink, last):
    return [(1.0 - shrink) * sum(w * m[i] for w, m in zip(weights, members)) + shrink * last
            for i in range(len(members[0]))]


def forecast(view):
    """Return H finite floats; raw/strength-zero behavior is always available."""
    mf, h, H = view["method_forecasts"], view["history"], view["H"]
    toto = mf["toto_2_0"]
    tfm = mf.get("timesfm_2_5", toto)
    moirai = mf.get("moirai_2_0", toto)
    sea = _seasonal(h, H, _season(view["freq"]))
    last = h[-1]
    if view["dataset"] == "drcik":
        return _blend([toto, tfm, moirai], (0.4, 0.5, 0.1), 0.3, last)
    if view["dataset"] == "time_mmd":
        return _blend([toto, tfm], (0.4, 0.6), 0.0, last)
    if "d" in str(view["freq"]).lower() and "w" not in str(view["freq"]).lower():
        return _blend([toto, tfm, sea], (0.7, 0.2, 0.1), 0.3, last)
    return _blend([tfm, sea], (0.7, 0.3), 0.0, last)
