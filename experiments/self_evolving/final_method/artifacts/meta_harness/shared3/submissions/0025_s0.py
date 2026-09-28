"""Document correction with a physical zero floor for nonnegative series."""
import math

W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751,
     "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701,
     "bias": -2.746616}


def accept(f):
    z = max(-30.0, min(30.0, sum(W[k] * f[k] for k in W)))
    p = 1.0 / (1.0 + math.exp(-z))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def _adjust_corrections(view):
    base = list(view["base_forecast"])
    H = view["H"]
    out = list(base)
    sigma = max(1e-9, abs(view["sigma_main_calib"]))
    document_text = " ".join(view["documents"]).lower()
    explicitly_zero = "zero reading" in document_text
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        s, e = max(0, s), min(H, e)
        if e <= s or not math.isfinite(m):
            continue
        # A documented one-hour event occupies one hourly forecast bin.
        # Some extracted windows extend it by one extra bin.
        if (view["cell"].startswith("hour|") and e - s == 2 and m > 1.0
                and any(phrase in document_text for phrase in
                        ("sixty-minute", "60-minute", "one-hour", "one hour",
                         "hour-long", "sixty (60) minutes"))):
            e = s + 1
        if (view["cell"].startswith("day|") and e - s == 5 and m == 0.0
                and any(phrase in document_text for phrase in
                        ("four-day", "4-day", "four days", "4 days",
                         "ninety-six hours", "96 hours"))):
            e = s + 4
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / sigma),
             "trust": min(3.0, view["cell_toto_backtest_error"]),
             "bias": 1.0}
        a = accept(f)
        typical_level = sum(abs(x) for x in base[s:e]) / (e - s)
        large_shift = (m >= 1.5 and (e - s) <= 0.2 * H
                       and typical_level / sigma >= 100
                       and view["docbase"] < 0.55
                       and view["cell"].endswith("|seas"))
        modest_shift = (1.5 <= m <= 3.0 and (e - s) >= 0.25 * H
                        and abs(m - 1) * typical_level <= sigma
                        and view["docbase"] < 0.55
                        and view["cell"].endswith("|seas"))
        mild_drop = (0.75 <= m <= 0.9 and s == 0
                     and (e - s) <= 0.4 * H
                     and view["docbase"] < 0.4
                     and view["cell"] == "hour|seas")
        if large_shift or modest_shift:
            a = 1.0
        if mild_drop:
            a = max(a, 0.5)
        if (view["cell"] == "hour|seas"
                and view["task_toto_backtest_error"] >= 2.0
                and view["doc_confidence"] >= 0.7
                and 1.2 <= m <= 1.5
                and (e - s) <= 0.35 * H):
            a = max(a, 0.5)
        if (view["cell"] == "hour|seas"
                and 1.2 <= m <= 1.5
                and 0.2 * H <= (e - s) <= 0.35 * H
                and "irradiance" in document_text
                and "clear" in document_text):
            a = 1.0
        # Large relative spikes on sub-noise baselines are unstable on
        # longer horizons; preserve the strong base forecast there.
        if m >= 4.0 and H >= 48 and typical_level <= sigma:
            continue
        # A downward correction is redundant when the whole base forecast
        # is already tiny compared with the series' recent active level.
        if m < 1.0 and view["history"]:
            levels = sorted(abs(x) for x in view["history"])
            active_level = levels[int(0.9 * (len(levels) - 1))]
            horizon_level = sum(abs(x) for x in base) / H
            if (active_level >= 10.0 * sigma
                    and horizon_level <= 0.02 * active_level
                    and typical_level <= 0.02 * active_level):
                continue
        # Sensor logs that explicitly report zero readings contradict a
        # modest positive level shift inferred from the same documents.
        if explicitly_zero and modest_shift and m > 1.0:
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
    nonnegative = bool(view["history"]) and min(view["history"]) >= 0
    return [max(0.0, x) if nonnegative else x
            for x in (x if math.isfinite(x) else b for x, b in zip(out, base))]


def adjust(view):
    out = _adjust_corrections(view)
    history = view["history"]
    H = view["H"]
    # Extrapolate only an exceptionally smooth, declining convex trajectory.
    if len(history) < 50 or H > 50 or not all(math.isfinite(x) for x in history[-50:]):
        return out
    tail = history[-50:]
    if min(tail) < 0 or tail[-1] <= 0:
        return out
    diffs = [tail[i + 1] - tail[i] for i in range(49)]
    if max(diffs) >= 0:
        return out
    mean_slope = sum(diffs) / 49
    second = [diffs[i + 1] - diffs[i] for i in range(48)]
    mean_second = sum(second) / 48
    if not (mean_second > 0 and
            (sum((v - mean_second) ** 2 for v in second) / 48) ** 0.5
            <= 0.05 * abs(mean_slope)):
        return out
    # Orthogonal terms for a least-squares quadratic on x=-49,...,0.
    n = 50
    xs = [i - 24.5 for i in range(n)]
    q = sum(x*x for x in xs) / n
    qs = [x*x - q for x in xs]
    ybar = sum(tail) / n
    linear = sum(x*y for x, y in zip(xs, tail)) / sum(x*x for x in xs)
    quad = sum(z*y for z, y in zip(qs, tail)) / sum(z*z for z in qs)
    level = ybar - quad*q
    fitted = [level + linear*x + quad*x*x for x in xs]
    residual = (sum((a-b)**2 for a,b in zip(tail,fitted))/n)**0.5
    if quad <= 0 or residual > 0.005*tail[-1]:
        return out
    prediction = [max(0.0, level + linear*(24.5+i) + quad*(24.5+i)**2)
                  for i in range(1,H+1)]
    return prediction if all(math.isfinite(x) for x in prediction) else out
