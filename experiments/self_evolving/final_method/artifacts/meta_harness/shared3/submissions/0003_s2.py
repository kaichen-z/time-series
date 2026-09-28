"""Document correction validator and local event adjustment."""
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
    if view["cell_toto_backtest_error"] < 0.0:
        return base
    effect = [0.0] * H
    count = [0] * H
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        width = max(0, min(H, e) - max(0, s))
        if not width or not math.isfinite(m):
            continue
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / max(1e-12, view["sigma_main_calib"])),
             "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = accept(f)
        # A short upward event with focused documents should not be rejected
        # solely because its extracted magnitude is modest.
        if (a == 0.0 and view["cell"].startswith("hour|seas")
                and 1.3 < m <= 2.0 and width / H <= 0.2
                and view["docbase"] < 0.48
                and view["cell_toto_backtest_error"] > 0.75):
            a = 1.0
        if not a:
            continue
        delta = a * (m - 1.0)
        for i in range(max(0, s), min(e, H)):
            # Average overlapping signals so document order does not change
            # the result. Conflicting directions then partially cancel.
            effect[i] += delta
            count[i] += 1
    out = []
    for i, b in enumerate(base):
        delta = effect[i] / count[i] if count[i] else 0.0
        cap = 0.5
        # An explicit zero during a short outage is more informative than
        # a generic decrease, but retain some base signal for timing error.
        if count[i] and delta < 0 and any(
            c["multiplier"] == 0 and c["start"] <= i < c["end"]
            and (c["end"] - c["start"]) / H < 0.2
            for c in view["corrections"]):
            cap = 0.65
        out.append(float(b + max(-cap * abs(b), min(0.5 * abs(b), b * delta))))
    return out
