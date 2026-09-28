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
        if holiday_policy and m > 1.0:
            continue
        f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
             "wfrac": (e - s) / H, "docbase": view["docbase"],
             "ratio": min(5.0, abs(m - 1) / view["sigma_main_calib"]),
             "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
        a = accept(f)
        short_spike = (1.5 <= m <= 10.0 and (e-s)/H <= 0.18 and not calendar_notice)
        if short_spike:
            a = 1.0
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
            elif m == 0.0 and (e-s) / H <= 0.2:
                caps[i] = 1.0
    return [b + max(-0.5 * abs(b), min(cap * abs(b), o - b))
            for b, o, cap in zip(base, out, caps)]
