"""Numerical ensemble with compensation for the paired multiplicative adjuster."""
import datetime as _dt


def _date(x):
    try:
        return _dt.date.fromisoformat(str(x)[:10])
    except Exception:
        return None


def _correction_multiplier(view):
    """Reproduce the frozen retriever/adjuster multiplier, independent of values."""
    H = view["H"]
    future = [_date(x) for x in view.get("future_timestamps") or []]
    future = [x for x in future if x]
    events = [(d.get("document_id", ""), e)
              for d in view.get("documents") or []
              for e in d.get("events") or []
              if e.get("direction") in ("up", "down")
              and (e.get("confidence") or 0) >= .6]
    dated = [x for _, e in events
             for x in (_date(e.get("time_start")), _date(e.get("time_end"))) if x]
    if future:
        dated = [x for x in dated if x < future[0]]
    anchor = max(dated) if dated else None
    f = str(view.get("freq", "")).lower()
    days = 14 if any(x in f for x in ("d", "w")) and "month" not in f else 45
    mult = [1.0] * H
    for _, e in events:
        sign = 1 if e["direction"] == "up" else -1
        conf = float(e["confidence"])
        t0, t1 = _date(e.get("time_start")), _date(e.get("time_end"))
        idx = [i for i, x in enumerate(future) if t0 and t0 <= x <= (t1 or t0)]
        if idx:
            start, end, step = min(idx), max(idx) + 1, .05
        else:
            last = t1 or t0
            if not (anchor and last and (anchor - last).days <= days
                    and (not future or last < future[0])):
                continue
            start, end, step = 0, max(1, -(-H // 3)), .03
        m = 1 + sign * step * conf
        for i in range(start, end):
            mult[i] *= m
    return [max(.9, min(1.1, x)) for x in mult]


def _seasonal(history, H, period):
    return [history[-period + i % period] if len(history) >= period else history[-1]
            for i in range(H)]


def forecast(view):
    mf, h, H = view["method_forecasts"], view["history"], view["H"]
    f = str(view["freq"]).lower()
    weekly = "w" in f or "week" in f
    if weekly:
        sea = _seasonal(h, H, 52)
        old = sum(h[-64:-52])
        ratio = sum(h[-12:]) / old if old else 1.0
        ratio = max(.8, min(1.25, ratio))
        sea = [x * (.25 + .75 * ratio) for x in sea]
        target = [.6 * a + .4 * b for a, b in zip(mf["timesfm_2_5"], sea)]
    else:
        sea = _seasonal(h, H, 7)
        toto = mf["toto_2_0"]
        chronos = mf.get("chronos_bolt", toto)
        last = h[-1]
        target = [.42 * a + .21 * b + .07 * c + .3 * last
                  for a, b, c in zip(toto, chronos, sea)]
    mult = _correction_multiplier(view)
    return [x / m for x, m in zip(target, mult)]
