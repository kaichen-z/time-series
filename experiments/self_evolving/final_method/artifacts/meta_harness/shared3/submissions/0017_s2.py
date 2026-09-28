"""Document adjustments with a separate regime for large, localized level shifts."""
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
    sigma = max(1e-9, abs(view["sigma_main_calib"]))
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        s, e = max(0, s), min(H, e)
        if e <= s or not math.isfinite(m):
            continue
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / sigma),
             "trust": min(3.0, view["cell_toto_backtest_error"]),
             "bias": 1.0}
        a = accept(f)

        # A localized positive change can be many normal-deviation units even
        # when its multiplier is modest. The seed's 50% cap hides such events.
        typical_level = sum(abs(x) for x in base[s:e]) / (e - s)
        large_shift = (m >= 1.5 and (e - s) <= 0.2 * H
                       and typical_level / sigma >= 100
                       and view["docbase"] < 0.55
                       and view["cell"].endswith("|seas"))
        # On low-level series, a moderate multiplier can still imply less
        # than one ordinary deviation. Long, documented positive events are
        # safe to apply without the seed validator's all-or-nothing gate.
        modest_shift = (1.5 <= m <= 3.0 and (e - s) >= 0.25 * H
                        and abs(m - 1) * typical_level <= sigma
                        and view["docbase"] < 0.55
                        and view["cell"].endswith("|seas"))
        mild_drop = (0.75 <= m <= 0.9 and s == 0
                     and (e - s) <= 0.4 * H
                     and view["docbase"] < 0.4
                     and view["cell"] == "hour|seas")
        if large_shift:
            a = 1.0
        if modest_shift:
            a = 1.0
        if mild_drop:
            a = max(a, 0.5)
        # When the task backtest itself is poor, modest localized positive
        # evidence can be worth using despite a boilerplate-heavy document.
        if (view["cell"] == "hour|seas"
                and view["task_toto_backtest_error"] >= 2.0
                and view["doc_confidence"] >= 0.7
                and 1.2 <= m <= 1.5
                and (e - s) <= 0.35 * H):
            a = max(a, 0.5)
        # A large relative multiplier on a baseline smaller than one normal
        # deviation is weak evidence of a large absolute event. On a long
        # horizon, keep the base for this isolated spike.
        if (m >= 4.0 and H >= 48 and typical_level <= sigma):
            continue
        if a <= 0:
            continue
        effect = a * (m - 1)
        if large_shift:
            effect = min(3.0, 0.75 * effect + 0.125)
        elif modest_shift:
            effect = min(1.25, effect)
        elif (m == 0.0 and (e - s) <= 0.15 * H
              and view["doc_confidence"] >= 0.9
              and view["docbase"] < 0.5):
            effect = -0.85
        else:
            effect = max(-0.5, min(0.5, effect))
        for i in range(s, e):
            out[i] = base[i] * (1.0 + effect)
    return [x if math.isfinite(x) else b for x, b in zip(out, base)]
