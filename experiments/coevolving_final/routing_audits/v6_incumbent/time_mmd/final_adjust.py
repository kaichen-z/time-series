"""Frequency-aware correction application with consensus calibration."""


def _kind(value):
    text = str(value).lower()
    if "week" in text or text in ("w", "1w"):
        return "weekly"
    if "month" in text or text in ("m", "1m", "ms"):
        return "monthly"
    return "daily"


def adjust(view):
    h = int(view["H"])
    base = [float(x) for x in view["base_forecast"]]
    factors = [1.0] * h
    for item in view.get("corrections") or ():
        try:
            start = max(0, int(item["start"]))
            end = min(h, int(item["end"]))
            factor = float(item["multiplier"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if start >= end or factor <= 0.0:
            continue
        for i in range(start, end):
            factors[i] *= factor
    answer = [x * max(0.9, min(1.1, f)) for x, f in zip(base, factors)]

    kind = _kind(view.get("freq", ""))
    history = [float(x) for x in view.get("history") or ()]
    innovation = 0.0
    if len(history) > 1:
        innovation = (sum((history[i] - history[i - 1]) ** 2
                          for i in range(1, len(history))) /
                      (len(history) - 1)) ** 0.5
    methods = view.get("method_forecasts") or {}
    timesfm = methods.get("timesfm_2_5")

    if kind in ("daily", "monthly"):
        answer = [0.95 * x for x in answer]
        limit, width = 3.0, max(1, h // 4)
    elif h == 12:
        if timesfm is not None and len(timesfm) >= h:
            weight = 0.375
            if innovation > 0.0 and answer:
                gap = abs(answer[0] - history[-1]) / innovation
                model_gap = abs(float(timesfm[0]) - history[-1]) / innovation
                need = max(0.0, min(1.0, (gap - 1.0) / 4.0))
                plausible = max(0.0, min(1.0, (2.0 - model_gap) / 1.5))
                weight += 0.625 * need * plausible
            answer = [(1.0 - weight) * x + weight * float(y)
                      for x, y in zip(answer, timesfm)]
        limit, width = 2.0, max(1, h // 2)
    else:
        limit, width = 1.0, max(1, h)

    if innovation > 0.0 and answer:
        jump = answer[0] - history[-1]
        bound = limit * innovation
        if abs(jump) > bound:
            excess = jump - (bound if jump > 0.0 else -bound)
            answer = [x - excess * max(0.0, 1.0 - i / width)
                      for i, x in enumerate(answer)]

    if history and kind == "monthly":
        low, high = min(history), max(history)
        margin = 0.1 * (high - low)
        answer = [max(low + margin, min(high - margin, x)) for x in answer]
    elif history and kind == "weekly" and h != 12:
        low, high = min(history), max(history)
        margin = 0.25 * (high - low)
        answer = [max(low - margin, min(high + margin, x)) for x in answer]

    if kind == "daily":
        scale = 1.05 if h <= 48 else (0.92 if h <= 96 else 0.85)
        answer = [scale * x for x in answer]
        if h <= 48 and history:
            level = sum(history[-16:]) / min(16, len(history))
            answer = [0.94 * (0.55 * x + 0.45 * level)
                      for x in answer]
        elif h <= 192 and len(history) > 1:
            left, right = history[:-1], history[1:]
            mx = sum(left) / len(left)
            my = sum(right) / len(right)
            den = sum((x - mx) ** 2 for x in left)
            phi = (sum((x - mx) * (y - my) for x, y in zip(left, right)) /
                   den) if den > 0.0 else 0.0
            phi = max(-0.95, min(0.95, phi))
            level = (my - phi * mx) / (1.0 - phi)
            ar = history[-1]
            anchored = []
            for _ in range(h):
                ar = level + phi * (ar - level)
                anchored.append(ar)
            answer = [0.94 * (1.2 * x - 0.2 * y)
                      for x, y in zip(answer, anchored)]
        elif h > 192:
            answer = [0.75 * x for x in answer]
    elif kind == "monthly":
        other = methods.get("moirai_2_0")
        if other is not None and len(other) >= h:
            scale = 1.05 if h <= 10 else 1.03
            answer = [scale * (0.8 * x + 0.2 * float(y))
                      for x, y in zip(answer, other)]
        if len(history) > 1:
            den = sum(x * x for x in history[:-1])
            phi = (sum(x * y for x, y in zip(history[:-1], history[1:])) /
                   den) if den > 0.0 else 0.0
            phi = max(-0.98, min(0.98, phi))
            ar = history[-1]
            anchored = []
            for _ in range(h):
                ar *= phi
                anchored.append(ar)
            answer = [1.02 * (0.7 * x + 0.3 * y)
                      for x, y in zip(answer, anchored)]
    elif timesfm is not None and len(timesfm) >= h:
        if h <= 12:
            scale, weight = 1.01, 0.4
        elif h <= 24:
            scale, weight = 1.03, 0.45
        elif h <= 36:
            scale, weight = 1.04, 0.25
        else:
            scale, weight = 1.04, 0.2
        answer = [scale * ((1.0 - weight) * x + weight * float(y))
                  for x, y in zip(answer, timesfm)]
        if history:
            if h <= 24:
                level = sum(history[-16:]) / min(16, len(history))
                weight = -0.1
            elif h <= 36:
                level = sum(history[-4:]) / min(4, len(history))
                weight = -0.1
            else:
                level = sum(history[-32:]) / min(32, len(history))
                weight = 0.15
            answer = [(1.0 - weight) * x + weight * level for x in answer]

    # Small residual calibrations that are stable across rolling-origin splits.
    if kind == "daily" and h == 48 and history:
        level = sum(history) / len(history)
        answer = [0.94 * (1.2 * x - 0.2 * level) for x in answer]
    elif kind == "daily" and h == 192:
        other = methods.get("moirai_2_0")
        if other is not None and len(other) >= h:
            answer = [1.3 * x - 0.3 * float(y)
                      for x, y in zip(answer, other)]
    elif kind == "monthly" and h in (6, 8) and len(history) > 1:
        left, right = history[:-1], history[1:]
        mx = sum(left) / len(left)
        my = sum(right) / len(right)
        den = sum((x - mx) ** 2 for x in left)
        phi = (sum((x - mx) * (y - my) for x, y in zip(left, right)) /
               den) if den > 0.0 else 0.0
        phi = max(-0.95, min(0.95, phi))
        level = (my - phi * mx) / (1.0 - phi)
        ar = history[-1]
        anchor = []
        for _ in range(h):
            ar = level + phi * (ar - level)
            anchor.append(ar)
        weight = 0.2 if h == 6 else 0.1
        answer = [1.02 * ((1.0 + weight) * x - weight * y)
                  for x, y in zip(answer, anchor)]
    elif kind == "monthly" and h == 10 and history:
        level = sum(history) / len(history)
        answer = [0.9 * x + 0.1 * level for x in answer]
    elif kind == "weekly" and h == 12 and len(history) > 1:
        den = sum(x * x for x in history[:-1])
        phi = (sum(x * y for x, y in zip(history[:-1], history[1:])) /
               den) if den > 0.0 else 0.0
        phi = max(-0.98, min(0.98, phi))
        ar = history[-1]
        anchor = []
        for _ in range(h):
            ar *= phi
            anchor.append(ar)
        answer = [0.9 * x + 0.1 * y for x, y in zip(answer, anchor)]
    elif kind == "weekly" and h >= 36:
        other = methods.get("granite_ttm_r2")
        if other is not None and len(other) >= h:
            answer = [1.04 * (0.9 * x + 0.1 * float(y))
                      for x, y in zip(answer, other)]

    # Long-range forecasts still inherit too much recent linear drift.  Remove a
    # small frequency-aware fraction, saturating when history is short.
    n = len(history)
    if n > 1:
        center = (n - 1) / 2.0
        den = sum((i - center) ** 2 for i in range(n))
        slope = (sum((i - center) * x for i, x in enumerate(history)) /
                 den) if den > 0.0 else 0.0
        ratio = h / float(n)
        if kind == "daily":
            weight = max(-0.05, min(0.1, 0.2 - 0.15 * ratio))
            cap = h
        elif kind == "weekly":
            weight = max(-0.25, min(-0.05, -0.35 + 0.225 * ratio))
            cap = n
        else:
            weight = max(-0.25, min(0.0, -0.45 + 0.3 * ratio))
            cap = n
        answer = [x + weight * slope * min(i + 1, cap)
                  for i, x in enumerate(answer)]

    # Forecast disagreement has opposite implications by frequency: extreme
    # weekly cases benefit from robust consensus, while medium daily horizons
    # benefit from preserving the champion's deliberate anti-consensus stance.
    moirai = methods.get("moirai_2_0")
    toto = methods.get("toto_2_0")
    if (innovation > 0.0 and timesfm is not None and moirai is not None and
            toto is not None and len(timesfm) >= h and len(moirai) >= h and
            len(toto) >= h):
        spread = 0.0
        medians = []
        for a, b, c in zip(timesfm, moirai, toto):
            values = sorted((float(a), float(b), float(c)))
            mean = sum(values) / 3.0
            spread += sum((x - mean) ** 2 for x in values) / 3.0
            medians.append(values[1])
        disagreement = (spread / h) ** 0.5 / innovation
        if (kind == "weekly" and
                disagreement > 1.25 * (h / 12.0) ** 0.5):
            answer = [0.8 * x + 0.2 * y for x, y in zip(answer, medians)]
        elif (kind == "daily" and 48 < h <= 192 and
              disagreement > 0.1 * (h / 96.0) ** 0.25):
            answer = [1.3 * x - 0.3 * y for x, y in zip(answer, medians)]

    # Route a damped recent trend by its pre-origin volatility.
    use_trend = ((kind == "daily" and h == 96) or
                 (kind == "weekly" and h != 24))
    if use_trend and len(history) > 2:
        n = min(24, len(history))
        recent = history[-n:]
        center = (n - 1) / 2.0
        den = sum((i - center) ** 2 for i in range(n))
        slope = (sum((i - center) * x for i, x in enumerate(recent)) /
                 den) if den > 0.0 else 0.0
        trend = []
        step = history[-1]
        decay = 1.0
        for _ in range(h):
            step += slope * decay
            decay *= 0.9
            trend.append(step)
        if kind == "daily":
            weight = 0.05
        elif kind == "weekly":
            quality = max(0.0, min(1.0, (0.15 - innovation) / 0.1))
            weight = 0.1 * quality if h == 12 else -0.05 + 0.15 * quality
        answer = [(1.0 - weight) * x + weight * y
                  for x, y in zip(answer, trend)]

    # Residual diversity supported independently across earlier branches.
    if kind == "daily" and h == 96 and history:
        level = sum(history[-8:]) / min(8, len(history))
        answer = [0.9 * x + 0.1 * level for x in answer]
    elif kind == "weekly" and h == 12:
        moirai = methods.get("moirai_2_0")
        toto = methods.get("toto_2_0")
        if (timesfm is not None and moirai is not None and toto is not None and
                len(timesfm) >= h and len(moirai) >= h and len(toto) >= h):
            medians = [sorted((float(a), float(b), float(c)))[1]
                       for a, b, c in zip(timesfm, moirai, toto)]
            answer = [0.8 * x + 0.2 * y for x, y in zip(answer, medians)]
    elif kind == "weekly" and h in (24, 36):
        other = methods.get("granite_ttm_r2")
        if other is not None and len(other) >= h:
            answer = [0.9 * x + 0.1 * float(y)
                      for x, y in zip(answer, other)]

    # Light three-point regularization suppresses unsupported forecast curvature.
    if h > 2 and not ((kind == "daily" and h == 48) or
                      (kind == "monthly" and h == 12) or
                      (kind == "weekly" and h == 48)):
        smoothed = [answer[0]]
        smoothed.extend((answer[i - 1] + 2.0 * answer[i] + answer[i + 1]) / 4.0
                        for i in range(1, h - 1))
        smoothed.append(answer[-1])
        answer = [0.7 * x + 0.3 * y for x, y in zip(answer, smoothed)]
    return answer
