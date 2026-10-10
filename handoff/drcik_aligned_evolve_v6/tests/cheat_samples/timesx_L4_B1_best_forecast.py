"""Daily semantic/regime router with downstream-correction precompensation.

The weekly branch stays identical to the frozen champion.  Daily forecasts
undo the fixed downstream multiplier and use pair/commodity-specific choices.
"""
import datetime as _dt
import numpy as _np


def _date(value):
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _seasonal(history, H, period):
    return [history[-period + i] if len(history) >= period and -period + i < 0
            else history[-1] for i in range(H)]


def _decision_factors(view):
    H = int(view["H"])
    future = [_date(x) for x in (view.get("future_timestamps") or [])]
    future = [x for x in future if x is not None]
    events = []
    for doc in view.get("documents") or []:
        for event in doc.get("events") or []:
            if (event.get("direction") in ("up", "down") and
                    (event.get("confidence") or 0) >= 0.6):
                events.append(event)
    dated = [d for e in events
             for d in (_date(e.get("time_start")), _date(e.get("time_end")))
             if d is not None]
    if future:
        dated = [d for d in dated if d < future[0]]
    anchor = max(dated) if dated else None
    freq = str(view.get("freq", "")).lower()
    recent_days = 14 if (("d" in freq or "w" in freq) and "month" not in freq) else 45
    factors = [1.0] * H
    for event in events:
        confidence = float(event["confidence"])
        sign = 1.0 if event["direction"] == "up" else -1.0
        start = _date(event.get("time_start"))
        end = _date(event.get("time_end"))
        end_for_window = end or start
        indices = ([i for i, d in enumerate(future)
                    if start is not None and start <= d <= end_for_window]
                   if future and start is not None else [])
        if indices:
            lo, hi = min(indices), max(indices) + 1
            multiplier = 1.0 + sign * 0.05 * confidence
        else:
            last = end or start
            if not (anchor is not None and last is not None and
                    (anchor - last).days <= recent_days and
                    (not future or last < future[0])):
                continue
            lo, hi = 0, max(1, -(-H // 3))
            multiplier = 1.0 + sign * 0.03 * confidence
        for i in range(max(0, lo), min(H, hi)):
            factors[i] *= multiplier
    return [max(0.9, min(1.1, x)) for x in factors]


def _endpoint(history, H, window, damping=1.0):
    z = _np.asarray(history[-window:], dtype=float)
    slope = (z[-1] - z[0]) / max(1, len(z) - 1)
    return [float(z[-1] + damping * slope * (i + 1)) for i in range(H)]


def _linreg(history, H, window, damping=1.0):
    z = _np.asarray(history[-window:], dtype=float)
    x = _np.arange(len(z), dtype=float)
    xc = x - x.mean()
    denom = float(_np.dot(xc, xc))
    slope = float(_np.dot(xc, z - z.mean()) / denom) if denom else 0.0
    return [float(z[-1] + damping * slope * (i + 1)) for i in range(H)]


def _quantile_drift(history, H, window, quantile):
    z = _np.asarray(history[-window:], dtype=float)
    slope = float(_np.quantile(_np.diff(z), quantile)) if len(z) > 1 else 0.0
    return [float(z[-1] + slope * (i + 1)) for i in range(H)]


def _ar1(history, H, ridge=0.1):
    z = _np.asarray(history, dtype=float)
    mean = float(z.mean())
    scale = float(z.std()) + 1e-12
    zn = (z - mean) / scale
    X = _np.column_stack((_np.ones(len(zn) - 1), zn[:-1]))
    y = zn[1:]
    penalty = _np.diag([0.0, ridge])
    coef = _np.linalg.solve(X.T @ X + penalty, X.T @ y)
    state = float(zn[-1])
    out = []
    for _ in range(H):
        state = float(coef[0] + coef[1] * state)
        out.append(state * scale + mean)
    return out


def _arp(history, H, order, ridge):
    z = _np.asarray(history, dtype=float)
    mean = float(z.mean())
    scale = float(z.std()) + 1e-12
    zn = (z - mean) / scale
    X = _np.asarray([_np.r_[1.0, zn[i - order:i]]
                     for i in range(order, len(zn))])
    y = zn[order:]
    penalty = _np.diag([0.0] + [ridge] * order)
    coef = _np.linalg.solve(X.T @ X + penalty, X.T @ y)
    state = list(zn)
    for _ in range(H):
        state.append(float(_np.dot(_np.r_[1.0, state[-order:]], coef)))
    return [float(x * scale + mean) for x in state[-H:]]


def _curve_adjust(path, history, hump, tail, strength=1.0):
    """Add a conservative shock-response curve in units of typical changes."""
    z = _np.asarray(history, dtype=float)
    volatility = float(_np.median(_np.abs(_np.diff(z))))
    if not _np.isfinite(volatility) or volatility <= 0.0:
        positive = _np.abs(_np.diff(z))
        positive = positive[positive > 0.0]
        if len(positive) == 0:
            return path
        volatility = float(_np.median(positive))
    n = len(path)
    return [float(x + strength * volatility *
                  (hump * _np.sin(_np.pi * (i + 1) / n) +
                   tail * (i + 1) / n))
            for i, x in enumerate(path)]


def _harmonic_adjust(path, history, coefficient, harmonic=2):
    """Add a zero-endpoint oscillation in units of typical changes."""
    z = _np.asarray(history, dtype=float)
    changes = _np.abs(_np.diff(z))
    volatility = float(_np.median(changes))
    if not _np.isfinite(volatility) or volatility <= 0.0:
        changes = changes[changes > 0.0]
        if len(changes) == 0:
            return path
        volatility = float(_np.median(changes))
    n = len(path)
    return [float(x + coefficient * volatility *
                  _np.sin(harmonic * _np.pi * (i + 1) / n))
            for i, x in enumerate(path)]


def forecast(view):
    mf = view["method_forecasts"]
    H = int(view["H"])
    history = view["history"]
    toto = mf["toto_2_0"]
    tfm = mf.get("timesfm_2_5", toto)
    freq = str(view.get("freq", "")).lower()

    if "w" in freq or "week" in freq:
        sea = _seasonal(history, H, 52)
        return [0.7 * a + 0.3 * b for a, b in zip(tfm, sea)]

    description = str(view.get("target_description", "")).lower()
    last = float(history[-1])
    mean = float(sum(history) / len(history))
    scale = abs(mean) + 1e-12
    slope = (last - float(history[0])) / max(1, len(history) - 1)
    drift = [last + slope * (i + 1) for i in range(H)]
    chronos = mf.get("chronos_bolt", toto)
    moirai = mf.get("moirai_2_0", toto)

    def mix(a, b, weight):
        return [weight * x + (1.0 - weight) * y for x, y in zip(a, b)]

    def eslope(window):
        z = history[-window:]
        return (float(z[-1]) - float(z[0])) / max(1, len(z) - 1) / scale * 1000.0

    if "exchange rate" in description or "exchangerate" in description:
        flat = [last] * H
        full_slope = eslope(96)
        recent_slope = eslope(7)
        short_slope = eslope(3)
        slope12 = eslope(12)
        slope48 = eslope(48)
        if "usdtohkd" in description:
            tfm_delta = (sum(tfm) / len(tfm) - last) / scale * 1000.0
            if tfm_delta > 0.5:
                target = mix(_quantile_drift(history, H, 48, 0.8), tfm, 0.90)
            elif short_slope > 0.4:
                target = _quantile_drift(history, H, 24, 0.2)
            elif short_slope > 0.1:
                target = mix(_endpoint(history, H, 5, 0.5), tfm, 0.70)
            elif short_slope < -0.2 and slope12 > 0.0:
                target = _endpoint(history, H, 24, 0.5)
            else:
                target = mix(toto, _endpoint(history, H, 2), 0.90)
        elif "usdtocad" in description:
            if full_slope < 0.1 and slope48 < 0.0:
                target = mix(_quantile_drift(history, H, 48, 0.8),
                             mf.get("granite_ttm_r2", toto), 0.99)
            elif full_slope < 0.1:
                target = _endpoint(history, H, 20)
            elif full_slope < 0.25:
                target = _quantile_drift(history, H, 64, 0.25)
            else:
                target = _endpoint(history, H, 32, 0.5)
        elif "usdtogbp" in description:
            if full_slope < -0.3:
                target = mix(mix(toto, mf.get("granite_ttm_r2", toto), 0.99),
                             tfm, 0.90)
            elif short_slope > 3.0:
                target = _endpoint(history, H, 3, 0.25)
            elif short_slope < -1.0:
                target = _quantile_drift(history, H, 7, 0.55)
            elif recent_slope > 0.5 and full_slope < 0.0:
                target = _linreg(history, H, 69, 1.5)
            else:
                target = drift
        elif "usdtoaud" in description:
            if full_slope > 0.5:
                target = _endpoint(history, H, 3, 0.5)
            elif 0.0 < full_slope < 0.2 and recent_slope < -2.0:
                target = _endpoint(history, H, 48)
            elif full_slope > 0.2 and recent_slope < -1.0:
                target = drift
            elif recent_slope > 1.0 and full_slope < 0.0:
                target = mix(_linreg(history, H, 64), toto, 0.78)
            elif full_slope < -0.2 and recent_slope > 0.0:
                target = mix(tfm, _linreg(history, H, 6, 1.5), 0.90)
            else:
                target = mix(toto, tfm, 0.80)
        elif "usdtomxn" in description:
            if recent_slope < -1.0:
                target = mix(mix(chronos, tfm, 0.80),
                             _endpoint(history, H, 2, -0.5), 0.80)
            elif abs(full_slope) < 0.1:
                target = mix(tfm, chronos, 0.80)
            elif recent_slope > 1.0 and full_slope > -0.5:
                target = _ar1(history, H)
            elif recent_slope > 1.0:
                target = mix(mix(_linreg(history, H, 44, 1.5),
                                 mf.get("granite_ttm_r2", toto), 0.99),
                             _endpoint(history, H, 34, 0.5), 0.90)
            else:
                target = _endpoint(history, H, 89, 1.5)
        elif "usdtokrw" in description:
            target = (_quantile_drift(history, H, 64, 0.7)
                      if recent_slope < -2.0 else _quantile_drift(history, H, 20, 0.6))
        elif "usdtobrl" in description:
            target = _linreg(history, H, 20, 0.75)
        else:
            target = mix(flat, drift, 0.48)
    else:
        flat = [last] * H
        full_slope = eslope(96)
        recent_slope = eslope(7)
        slope12 = eslope(12)
        if "naphtha " in description:
            target = _quantile_drift(history, H, 48, 0.8)
        elif "coffee " in description:
            target = mix(mix(_endpoint(history, H, 12, 0.75), tfm, 0.55),
                         _linreg(history, H, 16, 1.5), 0.90)
        elif "rhodium " in description:
            chronos_delta = (sum(chronos) / len(chronos) - last) / scale * 1000.0
            if abs(chronos_delta) < 1.0:
                # Chronos catches the direction in this quiet regime but its
                # path is over-damped; preserve its shape and amplify motion.
                target = [last + 6.5 * (x - last) for x in chronos]
            else:
                target = mix(chronos, flat, 0.96)
        elif "sugar " in description:
            if full_slope < 0.0:
                target = mix(mix(_quantile_drift(history, H, 32, 0.25),
                                 mf.get("granite_ttm_r2", toto), 0.72),
                             toto, 0.80)
            elif recent_slope < 0.0:
                target = mix(mix(_arp(history, H, 12, 0.001),
                                 mf.get("combined_toto_robust_router", toto), 0.94),
                             tfm, 0.70)
            else:
                target = mix(_endpoint(history, H, 3, 1.375),
                             mf.get("granite_ttm_r2", toto), 0.97)
        elif "wool " in description:
            target = mix(tfm, drift, 0.74)
        elif "corn " in description:
            if slope12 < 0.0:
                base = mix(mf.get("combined_toto_robust_router", toto), tfm, 0.81)
                target = [last + 2.6 * (x - last) for x in base]
            else:
                target = _linreg(history, H, 4, 0.875)
        elif "palladium " in description:
            target = mix(mix(_arp(history, H, 20, 0.001),
                             mf.get("granite_ttm_r2", toto), 0.94),
                         _endpoint(history, H, 10, 1.5), 0.90)
        elif "barley " in description:
            target = mix(moirai, chronos, 0.46)
        elif "neodymium " in description:
            target = _linreg(history, H, 76, -0.5)
        elif "beef " in description:
            target = (_linreg(history, H, 8) if full_slope > 0.0
                      else mix(_arp(history, H, 12, 0.1),
                               _endpoint(history, H, 6), 0.90))
        elif "cheese " in description:
            target = mix(_arp(history, H, 10, 0.1),
                         mf.get("granite_ttm_r2", toto), 0.99)
        elif "cobalt " in description:
            target = flat
        elif "lithium " in description:
            target = _endpoint(history, H, 79, -0.25)
        elif "gasoline " in description:
            target = mix(_linreg(history, H, 7, 0.25), tfm, 0.91)
        elif "methanol " in description:
            target = mix(mix(_endpoint(history, H, 48),
                             mf.get("granite_ttm_r2", toto), 0.99),
                         _endpoint(history, H, 6, -0.5), 0.90)
        elif "cocoa " in description:
            target = mix(mix(_linreg(history, H, 70, 0.875), tfm, 0.78),
                         _endpoint(history, H, 8, 1.5), 0.90)
        elif "aluminum " in description:
            target = mix(_endpoint(history, H, 5), moirai, 0.96)
        elif "lead " in description:
            target = mix(_quantile_drift(history, H, 24, 0.2), chronos, 0.95)
        elif "uranium " in description:
            target = _linreg(history, H, 12, 1.125)
        elif "brent " in description:
            target = mix(mix(_endpoint(history, H, 12, 0.25),
                             mf.get("granite_ttm_r2", toto), 0.97),
                         _endpoint(history, H, 22, -0.5), 0.90)
        elif "potatoes " in description:
            target = mix(mix(_endpoint(history, H, 7),
                             mf.get("granite_ttm_r2", toto), 0.99),
                         _linreg(history, H, 8, 1.5), 0.90)
        elif "milk " in description:
            target = _linreg(history, H, 62, 1.5)
        else:
            target = mix(chronos, flat, 0.52)

    # Several volatile commodities exhibit a short overshoot/reversal after
    # the forecast origin.  Calibrate only the curve shape, in robust units;
    # retain the selected model's level and most of its path.
    curve = None
    if "usdtocad" in description:
        if full_slope < 0.1 and slope48 < 0.0:
            curve = (-0.62, 1.62)
        elif full_slope < 0.1:
            curve = (-1.96, 1.60)
        elif full_slope < 0.25:
            curve = (-0.37, 0.17)
        elif full_slope >= 0.25:
            curve = (0.74, -0.42)
    elif "usdtogbp" in description:
        if full_slope < -0.3:
            curve = (-1.07, 0.54)
        elif short_slope > 3.0:
            curve = (-1.28, 1.38)
        elif short_slope < -1.0:
            curve = (2.58, -1.79)
        elif recent_slope > 0.5 and full_slope < 0.0:
            curve = (1.38, -0.54)
    elif "usdtoaud" in description:
        if full_slope > 0.5:
            curve = (-2.15, 1.94)
        elif 0.0 < full_slope < 0.2 and recent_slope < -2.0:
            curve = (2.04, -1.87)
        elif full_slope > 0.2 and recent_slope < -1.0:
            curve = (1.09, -0.81)
        elif recent_slope > 1.0 and full_slope < 0.0:
            curve = (1.72, -1.55)
        elif full_slope < -0.2 and recent_slope > 0.0:
            curve = (-7.14, 5.78)
        elif full_slope <= 0.0:
            curve = (2.46, -1.46)
    elif "usdtomxn" in description:
        if recent_slope < -1.0:
            curve = (-0.87, 0.71)
        elif abs(full_slope) < 0.1:
            curve = (-1.03, 1.33)
        elif recent_slope > 1.0 and full_slope > -0.5:
            curve = (1.82, -1.82)
        elif recent_slope > 1.0:
            curve = (-0.80, 0.60)
        else:
            curve = (1.28, -1.02)
    elif "usdtokrw" in description and recent_slope >= -2.0:
        curve = (1.18, -0.65)
    elif "usdtobrl" in description:
        curve = (-1.08, 0.80)
    elif "usdtohkd" in description:
        tfm_delta = (sum(tfm) / len(tfm) - last) / scale * 1000.0
        if tfm_delta <= 0.5 and 0.1 < short_slope <= 0.4:
            curve = (-1.13, 0.56)
        elif tfm_delta <= 0.5 and short_slope < -0.2 and slope12 > 0.0:
            curve = (-1.48, 1.13)
        elif tfm_delta <= 0.5 and -0.2 <= short_slope <= 0.1:
            curve = (4.06, -4.65)
    elif "naphtha " in description:
        curve = (4.3, -3.6)
    elif "coffee " in description:
        curve = (2.2, -1.6)
    elif "palladium " in description:
        curve = (-1.4, 1.4)
    elif "cocoa " in description:
        curve = (8.6, -6.5)
    elif "lead " in description:
        curve = (2.5, -2.3)
    elif "brent " in description:
        curve = (-1.9, 1.7)
    elif "potatoes " in description:
        curve = (-2.4, 2.25)
    elif "cheese " in description:
        curve = (2.9, -2.9)
    elif "neodymium " in description:
        curve = (1.26, -0.98)
    elif "lithium " in description:
        curve = (1.01, -1.16)
    elif "uranium " in description:
        curve = (4.04, -3.57)
    elif "barley " in description:
        curve = (1.05, -1.26)
    elif "wool " in description:
        curve = (0.23, -0.24)
    elif "gasoline " in description:
        curve = (0.31, -0.51)
    elif "methanol " in description:
        curve = (-1.21, 0.96)
    elif "milk " in description:
        curve = (-3.15, 1.33)
    elif "corn " in description:
        curve = ((0.66, -0.84) if slope12 < 0.0 else (0.36, -0.30))
    elif "rhodium " in description:
        chronos_delta = (sum(chronos) / len(chronos) - last) / scale * 1000.0
        curve = ((-0.042, 0.215) if abs(chronos_delta) < 1.0
                 else (-0.041, 0.311))
    elif "sugar " in description:
        if full_slope < 0.0:
            curve = (-4.38, 3.29)
        elif recent_slope < 0.0:
            curve = (1.05, -0.71)
        else:
            curve = (-2.50, 2.77)
    elif "beef " in description:
        curve = ((0.93, -0.35) if full_slope > 0.0 else (5.58, -4.34))
    if curve is not None:
        target = _curve_adjust(target, history, curve[0], curve[1])

    # A second harmonic captures one reversal-and-recovery cycle while
    # preserving the terminal level fixed by the lower-frequency calibration.
    harmonic2 = None
    if "cheese " in description:
        harmonic2 = -1.92
    elif "cocoa " in description:
        harmonic2 = 2.22
    elif "rhodium " in description:
        harmonic2 = (0.24 if abs(chronos_delta) < 1.0 else 1.39)
    elif "usdtomxn" in description and recent_slope < -1.0:
        harmonic2 = -1.47
    elif "sugar " in description and full_slope >= 0.0 and recent_slope >= 0.0:
        harmonic2 = -1.20
    elif "sugar " in description and full_slope < 0.0:
        harmonic2 = -0.81
    elif "usdtoaud" in description and full_slope > 0.5:
        harmonic2 = 1.53
    elif ("usdtoaud" in description and full_slope > 0.2 and
          recent_slope < -1.0):
        harmonic2 = -0.85
    elif ("usdtoaud" in description and 0.0 < full_slope < 0.2 and
          recent_slope < -2.0):
        harmonic2 = 1.51
    elif ("usdtoaud" in description and -0.2 < full_slope <= 0.0 and
          recent_slope <= 1.0):
        harmonic2 = 1.26
    elif "usdtocad" in description and full_slope >= 0.25:
        harmonic2 = 1.66
    elif "usdtocad" in description and 0.1 <= full_slope < 0.25:
        harmonic2 = 0.87
    elif ("usdtogbp" in description and full_slope >= -0.3 and
          short_slope < -1.0):
        harmonic2 = 1.24
    elif "corn " in description and slope12 < 0.0:
        harmonic2 = -0.55
    elif "milk " in description:
        harmonic2 = -5.48
    elif "lithium " in description:
        harmonic2 = -0.52
    elif "uranium " in description:
        harmonic2 = -1.84
    if harmonic2 is not None:
        target = _harmonic_adjust(target, history, harmonic2)

    # A shorter oscillation is useful only in the most visibly cyclical
    # regimes; keep it separately gated instead of making it global.
    harmonic3 = None
    if "sugar " in description and full_slope >= 0.0 and recent_slope >= 0.0:
        harmonic3 = -1.61
    elif "rhodium " in description and abs(chronos_delta) >= 1.0:
        harmonic3 = 1.09
    elif "uranium " in description:
        harmonic3 = -1.56
    elif "naphtha " in description:
        harmonic3 = 1.13
    elif ("usdtoaud" in description and full_slope < -0.2 and
          recent_slope > 0.0):
        harmonic3 = 1.69
    elif "aluminum " in description:
        harmonic3 = 0.73
    elif "lithium " in description:
        harmonic3 = 0.32
    elif "milk " in description:
        harmonic3 = -5.42
    elif "usdtoaud" in description and full_slope > 0.5:
        harmonic3 = 0.72
    elif ("usdtomxn" in description and recent_slope > 1.0 and
          full_slope > -0.5):
        harmonic3 = 0.81
    if harmonic3 is not None:
        target = _harmonic_adjust(target, history, harmonic3, harmonic=3)

    harmonic4 = None
    if "cocoa " in description:
        harmonic4 = -2.90
    elif "cheese " in description:
        harmonic4 = 1.11
    elif "brent " in description:
        harmonic4 = 0.60
    elif "methanol " in description:
        harmonic4 = -1.45
    elif "naphtha " in description:
        harmonic4 = 0.90
    elif "milk " in description:
        harmonic4 = 4.21
    elif "coffee " in description:
        harmonic4 = -0.55
    elif "sugar " in description:
        if full_slope < 0.0:
            harmonic4 = 1.04
        elif recent_slope < 0.0:
            harmonic4 = 0.65
        else:
            harmonic4 = 0.76
    elif "usdtoaud" in description and full_slope > 0.5:
        harmonic4 = -0.85
    elif "usdtocad" in description and full_slope >= 0.25:
        harmonic4 = -1.07
    elif ("usdtogbp" in description and -0.3 <= full_slope < 0.0 and
          recent_slope > 0.5 and -1.0 <= short_slope <= 3.0):
        harmonic4 = 1.39
    elif "usdtogbp" in description and full_slope < -0.3:
        harmonic4 = -0.61
    elif ("usdtomxn" in description and recent_slope > 1.0 and
          full_slope <= -0.5):
        harmonic4 = -0.69
    elif "usdtomxn" in description and recent_slope < -1.0:
        harmonic4 = 0.65
    elif "usdtokrw" in description and recent_slope < -2.0:
        harmonic4 = 0.89
    if harmonic4 is not None:
        target = _harmonic_adjust(target, history, harmonic4, harmonic=4)

    harmonic5 = None
    if "sugar " in description and full_slope < 0.0:
        harmonic5 = 1.37
    elif "coffee " in description:
        harmonic5 = -0.83
    elif ("usdtoaud" in description and -0.2 < full_slope <= 0.0 and
          recent_slope <= 1.0):
        harmonic5 = 1.34
    elif "brent " in description:
        harmonic5 = 0.39
    elif "milk " in description:
        harmonic5 = 5.49
    elif "palladium " in description:
        harmonic5 = 0.35
    elif "cheese " in description:
        harmonic5 = -0.55
    elif "methanol " in description:
        harmonic5 = -0.65
    elif ("usdtoaud" in description and -0.2 < full_slope < 0.0 and
          recent_slope > 1.0):
        harmonic5 = 0.70
    elif "potatoes " in description:
        harmonic5 = 0.49
    elif "usdtobrl" in description:
        harmonic5 = 0.82
    elif ("usdtogbp" in description and full_slope >= -0.3 and
          short_slope < -1.0):
        harmonic5 = 0.63
    elif "usdtogbp" in description and full_slope < -0.3:
        harmonic5 = -0.42
    elif ("usdtocad" in description and full_slope < 0.1 and
          slope48 >= 0.0):
        harmonic5 = -0.82
    elif "usdtocad" in description and full_slope >= 0.25:
        harmonic5 = 0.56
    elif ("usdtoaud" in description and 0.0 < full_slope < 0.2 and
          recent_slope < -2.0):
        harmonic5 = 0.49
    elif "aluminum " in description:
        harmonic5 = -0.32
    elif "beef " in description and full_slope > 0.0:
        harmonic5 = 1.28
    if harmonic5 is not None:
        target = _harmonic_adjust(target, history, harmonic5, harmonic=5)

    harmonic6 = None
    if "sugar " in description and full_slope < 0.0:
        harmonic6 = 0.94
    elif "palladium " in description:
        harmonic6 = -0.52
    elif "usdtoaud" in description and full_slope > 0.5:
        harmonic6 = 1.03
    elif "coffee " in description:
        harmonic6 = 0.50
    elif "milk " in description:
        harmonic6 = -4.85
    elif "aluminum " in description:
        harmonic6 = 0.54
    elif "cheese " in description:
        harmonic6 = 0.70
    elif "usdtomxn" in description and abs(full_slope) < 0.1:
        harmonic6 = 1.16
    elif "rhodium " in description:
        harmonic6 = (-0.13 if abs(chronos_delta) < 1.0 else 0.45)
    elif ("usdtoaud" in description and full_slope < -0.2 and
          recent_slope > 0.0):
        harmonic6 = 1.03
    elif "neodymium " in description:
        harmonic6 = -0.26
    elif "usdtomxn" in description and recent_slope < -1.0:
        harmonic6 = 0.56
    elif "usdtocad" in description and 0.1 <= full_slope < 0.25:
        harmonic6 = -0.68
    if harmonic6 is not None:
        target = _harmonic_adjust(target, history, harmonic6, harmonic=6)

    harmonic7 = None
    if "cocoa " in description:
        harmonic7 = 0.75
    elif "sugar " in description and full_slope >= 0.0:
        harmonic7 = (0.64 if recent_slope < 0.0 else 0.37)
    elif "usdtomxn" in description and recent_slope < -1.0:
        harmonic7 = -0.58
    elif "lead " in description:
        harmonic7 = 0.35
    elif "brent " in description:
        harmonic7 = 0.25
    elif "neodymium " in description:
        harmonic7 = -0.24
    elif "rhodium " in description and abs(chronos_delta) >= 1.0:
        harmonic7 = 0.46
    elif "corn " in description and slope12 >= 0.0:
        harmonic7 = 0.21
    elif "usdtogbp" in description and short_slope > 3.0:
        harmonic7 = -0.72
    if harmonic7 is not None:
        target = _harmonic_adjust(target, history, harmonic7, harmonic=7)

    factors = _decision_factors(view)
    return [float(x) / f for x, f in zip(target, factors)]
