"""Document adjustment with a wider bound for isolated short surge events."""
import math

W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751,
     "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701,
     "bias": -2.746616}


def accept(f):
    z = sum(W[k] * f[k] for k in W)
    z = max(-30.0, min(30.0, z))
    p = 1 / (1 + math.exp(-z))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def adjust(view):
    base = list(view["base_forecast"])
    H = view["H"]
    out = list(base)
    corrections = view["corrections"]
    for c in corrections:
        s, e, m = c["start"], c["end"], c["multiplier"]
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / view["sigma_main_calib"]),
             "trust": min(3.0, view["cell_toto_backtest_error"]),
             "bias": 1.0}
        a = accept(f)
        if a <= 0:
            continue
        # A single, short, plausible surge has a different risk profile from
        # a sustained trend or one of several overlapping event claims.
        lower = -1.0 if (len(corrections) == 1 and m == 0 and
                          e - s <= H / 2 and view["cell"].startswith("day")) else -0.5
        if lower == -1.0:
            a = 1.0
        upper = max(1.5, m - 2) if (len(corrections) == 1 and
                                     2 <= m <= 6 and
                                     e - s <= H / 2 and
                                     view["cell"].startswith("hour")) else 0.5
        mm = 1 + max(lower, min(upper, a * (m - 1)))
        for i in range(max(0, s), min(e - 1 if lower == -1.0 else e, H)):
            out[i] = base[i] * mm
    return [float(x) if math.isfinite(x) else float(b)
            for x, b in zip(out, base)]
