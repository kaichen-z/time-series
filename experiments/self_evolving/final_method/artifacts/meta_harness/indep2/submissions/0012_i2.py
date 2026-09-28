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
    shutdown = [False] * H
    sigma = max(float(view["sigma_main_calib"]), 1e-9)
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / sigma),
             "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = accept(f)
        # A short, modest increase can still be useful on a stable, high-level
        # series: the validator's magnitude feature otherwise rejects it.
        if 1.3 <= m <= 2.0 and (e - s) <= H // 4:
            window = base[max(0, s):min(e, H)]
            if window and min(abs(x) for x in window) / sigma > 100:
                a = max(a, 1.0)
        # A modest local uplift can also matter when the base is only tens of
        # calibration units high, provided the event is large in absolute
        # units and Toto's cell backtest is poor. Use half strength here.
        if 1.2 <= m <= 2.0 and (e - s) <= H // 3 and view["cell_toto_backtest_error"] >= 0.8:
            window = base[max(0, s):min(e, H)]
            if window and max(abs((m - 1.0) * x) for x in window) >= 5.0 * sigma:
                a = max(a, 0.5)
        short_shutdown = m == 0.0 and (e - s) <= H // 4
        if short_shutdown:
            a = 1.0
        if a <= 0:
            continue
        mm = 1.0 + a * (m - 1.0)
        for i in range(max(0, s), min(e, H)):
            # A multiplier is poorly identified when its full implied change
            # remains within a few units of the series calibration scale.
            if m != 0.0 and abs((m - 1.0) * base[i]) < 5.0 * sigma:
                continue
            out[i] = base[i] * (1.0 + (mm - 1.0) * (0.5 if short_shutdown and i == min(e, H) - 1 else 1.0))
            high_event[i] = m >= 5.0
            shutdown[i] = short_shutdown
    result = []
    for i, (b, o) in enumerate(zip(base, out)):
        cap_up = 0.5
        # Large local events in high-level series can legitimately exceed the
        # usual 50% bound. The calibration scale distinguishes these series.
        if o > b and abs(b) / sigma > 100:
            cap_up = 3.0 if high_event[i] else 2.0
        down = 1.0 if shutdown[i] else 0.5
        delta = max(-down * abs(b), min(cap_up * abs(b), o - b))
        result.append(float(b + delta))
    # A zero target over the whole horizon can describe a gradual run-down.
    # Apply a smooth taper when the observed series is already steadily declining.
    hist = view.get("history", [])
    recent = hist[-20:]
    declining = len(recent) >= 10 and all(recent[j] >= recent[j + 1] > 0 for j in range(len(recent) - 1))
    if declining:
        for c in view["corrections"]:
            if c["multiplier"] == 0.0 and c["start"] == 0 and c["end"] >= H:
                for i in range(H):
                    phase = i / max(1, H - 1)
                    result[i] = float(base[i] * (1.0 - phase) ** 1.6)
    # A very large multiplier on a near-zero baseline can imply an event in
    # absolute units, while a percentage cap barely moves the prediction.
    # Limit that uplift to the calibration scale and avoid adding it to an
    # already elevated base step within the same event window.
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        if m < 10.0 or e - s > H // 4:
            continue
        window = base[max(0, s):min(e, H)]
        if not window:
            continue
        middle = sorted(window)[len(window) // 2]
        if middle <= 0 or middle > 2.0 * sigma:
            continue
        target = middle + 8.0 * sigma
        for i in range(max(0, s), min(e, H)):
            if 0 < base[i] <= 3.0 * middle:
                result[i] = float(max(result[i], min(m * base[i], target)))
    return result
