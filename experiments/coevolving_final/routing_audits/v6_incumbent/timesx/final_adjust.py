"""Decision: restore the paired base, with a guarded weekly annual analogue."""


def _multipliers(view):
    H = view["H"]
    mult = [1.0] * H
    for c in view.get("corrections") or []:
        try:
            lo = max(0, int(c["start"]))
            hi = min(H, int(c["end"]))
            m = float(c["multiplier"])
        except (KeyError, TypeError, ValueError):
            continue
        if lo >= hi:
            continue
        for i in range(lo, hi):
            mult[i] *= m
    return [max(.9, min(1.1, m)) for m in mult]


def _restore(view):
    return [x * m for x, m in zip(view["base_forecast"], _multipliers(view))]


def _analogue(h, origin, i, ratio, side):
    nearby = [h[origin - 51 + i], h[origin - 52 + i],
              h[origin - 53 + i]]
    if side < 0:
        x = sorted(nearby)[1]
    else:
        x = side * nearby[0] + (1 - 2 * side) * nearby[1]
        x += side * nearby[2]
    return x * (.25 + .75 * ratio)


def _ratio(h, origin):
    old = sum(h[origin - 64:origin - 52])
    ratio = sum(h[origin - 12:origin]) / old if old else 1.0
    return max(.8, min(1.25, ratio))


def _cv(h, side, origins):
    score = 0.0
    for origin in origins:
        ratio = _ratio(h, origin)
        actual = h[origin:origin + 12]
        scale = max(1e-12, sum(abs(x) for x in actual) / 12)
        errors = [(_analogue(h, origin, i, ratio, side) - y) / scale
                  for i, y in enumerate(actual)]
        score += sum(abs(x) for x in errors) / 12
        score += (sum(x * x for x in errors) / 12) ** .5
    return score


def _errors(h, side):
    seasonal = persistence = 0.0
    for origin in (64, 72, 84):
        ratio = _ratio(h, origin)
        scale = max(1e-12, sum(abs(x) for x in h[origin:origin + 12]))
        for i, y in enumerate(h[origin:origin + 12]):
            seasonal += abs(_analogue(h, origin, i, ratio, side) - y) / scale
            persistence += abs(h[origin - 1] - y) / scale
    return seasonal, persistence


def _weekly(view, fallback):
    h = view["history"]
    if len(h) < 96:
        return fallback
    raw = _cv(h, 0, (64, 72, 84))
    smooth = _cv(h, .25, (64, 72, 84))
    side = 0 if raw < .95 * smooth else .25
    robust = _cv(h, -1, (64, 72, 84))
    if robust < .95 * _cv(h, side, (64, 72, 84)):
        side = -1
    es, ep = _errors(h, side)
    weight = (.6 if es < .4 * ep else
              (.55 if es < .7 * ep else (.25 if es > 1.1 * ep else .4)))
    ratio = _ratio(h, len(h))
    seasonal = [_analogue(h, len(h), i, ratio, side)
                for i in range(view["H"])]
    models = view["method_forecasts"]
    timesfm = models.get("timesfm_2_5")
    if not timesfm:
        return fallback
    out = [(1 - weight) * x + weight * y for x, y in zip(timesfm, seasonal)]
    names = ("timesfm_2_5", "toto_2_0", "chronos_bolt", "moirai_2_0")
    clipped = []
    guard = .875
    for i, x in enumerate(out):
        values = [models[name][i] for name in names if models.get(name)]
        lo, hi = min(values), max(values)
        width = hi - lo
        clipped.append(max(lo - guard * width, min(hi + guard * width, x)))
    return clipped


def adjust(view):
    restored = _restore(view)
    freq = str(view.get("freq", "")).lower()
    return _weekly(view, restored) if ("w" in freq or "week" in freq) else restored
