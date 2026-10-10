"""Seed Numerical module (L4 start point): the unified framework's numerical step from the Dr-CiK co-evolution seed
(coevo_x/seed_forecast.py), with the per-dataset blends it used, adapted to the official views' frequency labels.
  time_mmd: 0.4*Toto + 0.6*TimesFM
  timesx  : daily 0.7*Toto + 0.2*TimesFM + 0.1*seasonal, shrink 0.3 to last value; weekly 0.7*TimesFM + 0.3*seasonal
Missing anchors fall back to Toto. forecast(view) -> list of H floats."""


def season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f: return None
    for k, p in (("hour", 24), ("week", 52), ("w", 52), ("month", 12), ("day", 7), ("d", 7)):
        if k in f: return p
    return None


def seasonal(h, H, p):
    return [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]


def blend(members, w, s, last):
    return [(1 - s) * sum(wi * m[i] for wi, m in zip(w, members)) + s * last for i in range(len(members[0]))]


def forecast(view):
    mf = view["method_forecasts"]; H = view["H"]; h = view["history"]
    toto = mf["toto_2_0"]; tfm = mf.get("timesfm_2_5", toto); sea = seasonal(h, H, season(view["freq"])); last = h[-1]
    if view["dataset"] == "time_mmd":
        return blend([toto, tfm, sea], (0.4, 0.6, 0.0), 0.0, last)
    if "d" in str(view["freq"]).lower() and "w" not in str(view["freq"]).lower():
        return blend([toto, tfm, sea], (0.7, 0.2, 0.1), 0.3, last)
    return blend([toto, tfm, sea], (0.0, 0.7, 0.3), 0.0, last)
