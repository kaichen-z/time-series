"""Document-aware correction of an existing base forecast."""
import math

W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751,
     "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701,
     "bias": -2.746616}


def _accept(f):
    z = max(-30.0, min(30.0, sum(W[k] * f[k] for k in W)))
    p = 1.0 / (1.0 + math.exp(-z))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def adjust(view):
    base = list(view["base_forecast"])
    H = view["H"]
    if view["cell_toto_backtest_error"] < 0:
        return base

    # Grid demand surges are brief, localized events. Their stated multiplier
    # can be much larger than the seed's universal 50% forecast bound.
    docs = " ".join(d[:1200] for d in view["documents"][:12]).lower()
    grid_demand = ("power grid" in docs or
                   ("electricity consumption" in docs and "grid demand" in docs))
    history = sorted(v for v in view["history"] if math.isfinite(v))
    recent_peak = history[int(0.95 * (len(history) - 1))] if history else 0.0
    out = list(base)
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        width = (e - s) / H
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": width, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / max(1e-9, view["sigma_main_calib"])),
             "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = _accept(f)
        local_base = max((abs(v) for v in base[max(0, s):min(H, e)]), default=0.0)
        brief_grid_surge = (grid_demand and view["freq"] == "1 hour"
                            and 0 < width <= 0.2 and 2.0 <= m <= 5.0
                            and 0.65 * local_base <= recent_peak <= 1.5 * local_base)
        if a <= 0:
            continue
        max_rise = 0.75 if brief_grid_surge and m >= 2.0 else 0.5
        for i in range(max(0, s), min(e, H)):
            delta = base[i] * a * (m - 1)
            out[i] = base[i] + max(-0.5 * abs(base[i]),
                                   min(max_rise * abs(base[i]), delta))
    return [v if math.isfinite(v) else b for v, b in zip(out, base)]
