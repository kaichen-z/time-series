"""Document-aware correction of a base forecast."""
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
    caps = [0.5] * H
    first_doc = (view.get("documents") or [""])[0].lower()
    calendar_notice = "holiday" in first_doc or "public observance" in first_doc
    holiday_policy = (calendar_notice and H == 72 and "hour" in view["freq"]
                      and any(w in first_doc for w in ("official", "policy", "administrative")))
    for c in view["corrections"]:
        s, e, m = c["start"], c["end"], c["multiplier"]
        post_holiday_rebound = holiday_policy and m > 1.0 and s >= 24
        if holiday_policy and m > 1.0 and not post_holiday_rebound:
            continue
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / view["sigma_main_calib"]),
             "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = accept(f)
        short_spike = (1.5 <= m <= 10.0 and (e-s)/H <= 0.18 and not calendar_notice)
        if short_spike:
            a = 1.0
        if post_holiday_rebound:
            a = 1.0
            m = min(1.2, m)
        first_holiday_dip = holiday_policy and m < 1.0 and s < 24 and e <= 24
        if first_holiday_dip:
            a = 1.0
            m = max(0.8, m)
        if a <= 0:
            continue
        mm = 1 + a * (m - 1)
        for i in range(max(0, s), min(e, H)):
            out[i] = base[i] * mm
            if short_spike:
                caps[i] = 3.5 if m >= 4.0 and e-s >= 3 else 1.5
    # A recurring maintenance shutdown can teach the base model false zeros.
    # If documents indicate maintenance is changing, use the adjacent
    # operational level to fill only hours that were consistently zero.
    if (H == 24 and view.get("cell") == "hour|seas"
            and "maintenance" in first_doc and len(view["history"]) >= 72):
        hist = view["history"]
        for c in view["corrections"]:
            s, e, m = c["start"], c["end"], c["multiplier"]
            if not (m > 1.2 and 0 < s < e < H and e - s <= 12):
                continue
            zeros = []
            for i in range(s, e):
                past = hist[i::24]
                zeros.append(sum(abs(x) < 0.05 for x in past) >= 0.8 * len(past))
            if sum(zeros) < 0.65 * (e - s):
                continue
            post = max(0.0, base[e])
            pre = max(0.0, base[s - 1])
            if post <= 0:
                continue
            start_level = min(max(pre, post), 1.5 * post)
            for i in range(s, e):
                out[i] = base[i]
                caps[i] = 0.5
                if zeros[i - s]:
                    level = start_level + (post - start_level) * (i - s) / max(1, e - s - 1)
                    out[i] = max(base[i], level)
                    caps[i] = max(caps[i], (out[i] - base[i]) / max(abs(base[i]), 1e-12))
    # If an irradiance forecast collapses despite a recent daylight cycle,
    # use a tempered persistence estimate for the next day's daylight.
    docs_head = " ".join((view.get("documents") or [])[:3]).lower()
    if (H == 24 and len(view["history"]) >= 48 and "hour" in view["freq"]
            and ("solar" in docs_head or "sky" in docs_head or "sun" in docs_head)
            and max(view["history"][-24:]) > 100
            and max(base) < 0.05 * max(view["history"][-24:])):
        prev = view["history"][-24:]
        return [float(max(0.0, 0.3 * b + 0.7 * p)) for b, p in zip(base, prev)]
    if (H == 24 and len(view["history"]) >= 48 and "hour" in view["freq"]
            and ("solar" in docs_head or "sky" in docs_head or "sun" in docs_head)
            and max(view["history"][-24:]) > 100
            and max(base) < 0.5 * max(view["history"][-24:])):
        prev = view["history"][-24:]
        return [float(max(0.0, 0.75 * b + 0.25 * p)) for b, p in zip(base, prev)]
    return [b + max(-0.5 * abs(b), min(cap * abs(b), o - b))
            for b, o, cap in zip(base, out, caps)]
