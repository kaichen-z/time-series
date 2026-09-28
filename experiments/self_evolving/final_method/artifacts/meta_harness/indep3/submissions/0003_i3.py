"""Seed correction function = the current main method (evolved nrd4 team, seed 1), rewritten as plain code.
adjust(view) -> list of H floats.  `view` fields: see TASK.md.  Must not read any other file."""
import math

W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751, "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701, "bias": -2.746616}
STRENGTH, TRUST_GATE = 1.0, 0.0


def accept(f):
    z = sum(W[k] * f[k] for k in W); z = max(-30.0, min(30.0, z)); p = 1 / (1 + math.exp(-z))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def adjust(view):
    base = list(view["base_forecast"]); H = view["H"]
    if view["cell_toto_backtest_error"] < TRUST_GATE:
        return base                                   # Toto trusted in this cell: leave the base untouched
    out = list(base)
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"], "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / view["sigma_main_calib"]), "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = accept(f)
        if a <= 0: continue
        mm = 1 + STRENGTH * (1 + a * (m - 1) - 1)
        for i in range(s, min(e, H)): out[i] = base[i] * mm
    # A brief, positive demand spike can exceed the generic 50% safety cap.
    upper = 0.5
    if H <= 24 and len(view["corrections"]) == 1:
        c = view["corrections"][0]
        if 2.0 <= c["multiplier"] <= 5.0 and c["end"] - c["start"] <= 0.2 * H:
            upper = max(1.0, (2.0 / 3.0) * (c["multiplier"] - 1.0))
    return [b + max(-0.5 * abs(b), min(upper * abs(b), o - b)) for b, o in zip(base, out)]
