"""Document correction for the Dr-CiK forecast."""
import math
import statistics

W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751,
     "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701,
     "bias": -2.746616}


def accept(f):
    z = max(-30.0, min(30.0, sum(W[k] * f[k] for k in W)))
    p = 1 / (1 + math.exp(-z))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def adjust(view):
    base = list(view["base_forecast"])
    H = view["H"]
    cs = view["corrections"]
    history = view["history"]
    if (len(cs) == 1 and cs[0]["start"] == 0 and cs[0]["end"] >= H
            and cs[0]["multiplier"] == 0 and view["cell"].endswith("|flat")
            and len(history) >= 12 and H >= 12 and history[-1] > 0
            and all(x > 0 for x in history[-12:])):
        diffs = [history[i] - history[i-1] for i in range(len(history)-10, len(history))]
        slope = statistics.median(diffs)
        if slope < 0 and sum(d < 0 for d in diffs) >= 8 and max(abs(d-slope) for d in diffs) < 0.4*abs(slope):
            return [min(b, max(0.0, history[-1] + (i+1)*(2.0/3.0)*slope))
                    for i, b in enumerate(base)]
    if view["cell_toto_backtest_error"] < 0.0:
        return base
    out = list(base)
    short_spike = False
    if H <= 24 and len(cs) == 1:
        c = cs[0]
        short_spike = (2.0 <= c["multiplier"] <= 5.0
                       and 1 < c["end"] - c["start"] <= 0.2 * H)
    for c in cs:
        s, e, m = c["start"], c["end"], c["multiplier"]
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / view["sigma_main_calib"]),
             "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = accept(f)
        if a <= 0:
            continue
        mm = 1 + a * (m - 1)
        effective_end = e - 1 if short_spike else e
        for i in range(s, min(effective_end, H)):
            out[i] = base[i] * mm
    upper = max(1.0, cs[0]["multiplier"] - 1.0) if short_spike else 0.5
    out = [b + max(-0.5 * abs(b), min(upper * abs(b), o - b))
           for b, o in zip(base, out)]

    # If the baseline repeats a historical outage, multiplying its near-zero
    # values cannot represent a documented positive event. Interpolate the
    # surrounding active hours over the interior of the outage instead.
    if H <= 24 and len(cs) == 1 and view["cell"].startswith("hour|"):
        c = cs[0]
        s, e, m = c["start"], c["end"], c["multiplier"]
        if (m >= 1.5 and 4 <= e-s <= 8 and s >= 2 and e+1 < H
                and min(base[s-1], base[e]) > 0):
            surround = statistics.median(base[s-2:s] + base[e:e+2])
            inside = statistics.median(base[s:e-1])
            if surround > 0 and inside < 0.2 * surround:
                for i in range(s, e-1):
                    bridge = base[s-1] + (base[e] - base[s-1]) * (i-s+1) / (e-s+1)
                    out[i] = max(out[i], bridge)
                out[e-1] = base[e-1]
    return out
