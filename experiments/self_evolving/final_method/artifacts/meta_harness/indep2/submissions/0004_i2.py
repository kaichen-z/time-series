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
    event_mag = [1.0] * H
    short_shutdown = [False] * H
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
            event_mag[i] = m
            short_shutdown[i] = m == 0.0 and (e - s) <= H // 4
    result = []
    for i, (b, o) in enumerate(zip(base, out)):
        cap_up = 0.5
        # Large local events in high-level series can legitimately exceed the
        # usual 50% bound. The calibration scale distinguishes these series.
        if o > b and abs(b) / sigma > 100:
            cap_up = min(4.0, max(2.0, event_mag[i] - 1.0))
        cap_down = 1.0 if short_shutdown[i] else 0.5
        delta = max(-cap_down * abs(b), min(cap_up * abs(b), o - b))
        result.append(float(b + delta))
    return result
