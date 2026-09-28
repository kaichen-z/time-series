"""Document correction with a level-aware cap for strong local upward events."""
import math

W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751,
     "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701,
     "bias": -2.746616}


def accept(f):
    z = max(-30.0, min(30.0, sum(W[k] * f[k] for k in W)))
    p = 1.0 / (1.0 + math.exp(-z))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def adjust(view):
    base = list(view["base_forecast"])
    H = view["H"]
    out = list(base)
    high_event = [False] * H
    threefold_event = [False] * H
    sigma = max(float(view["sigma_main_calib"]), 1e-9)
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / sigma),
             "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = accept(f)
        if a <= 0:
            continue
        mm = 1.0 + a * (m - 1.0)
        for i in range(max(0, s), min(e, H)):
            out[i] = base[i] * mm
            high_event[i] = m >= 5.0
            threefold_event[i] = 3.0 <= m < 4.0
    result = []
    for i, (b, o) in enumerate(zip(base, out)):
        cap_up = 0.5
        # Large local events in high-level series can legitimately exceed the
        # usual 50% bound. The calibration scale distinguishes these series.
        if o > b and abs(b) / sigma > 100:
            cap_up = 3.0 if high_event[i] else (1.5 if threefold_event[i] else 2.0)
        delta = max(-0.5 * abs(b), min(cap_up * abs(b), o - b))
        result.append(float(b + delta))
    return result
