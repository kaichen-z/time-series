"""Single neutral ForecastController seed for all three benchmarks."""


def _season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f:
        return None
    for k, p in (("hour", 24), ("week", 52), ("w", 52), ("month", 12), ("day", 7), ("d", 7)):
        if k in f:
            return p
    return None


def _seasonal(h, H, p):
    return [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]


def _blend(members, weights, shrink, last):
    return [(1.0 - shrink) * sum(w * m[i] for w, m in zip(weights, members)) + shrink * last
            for i in range(len(members[0]))]


def forecast(view):
    """Return H finite floats; raw/strength-zero behavior is always available."""
    mf, h, H = view["method_forecasts"], view["history"], view["H"]
    toto = mf["toto_2_0"]
    tfm = mf.get("timesfm_2_5", toto)
    moirai = mf.get("moirai_2_0", toto)
    sea = _seasonal(h, H, _season(view["freq"]))
    last = h[-1]
    if view["dataset"] == "drcik":
        return _blend([toto, tfm, moirai], (0.4, 0.5, 0.1), 0.3, last)
    if view["dataset"] == "time_mmd":
        return _blend([toto, tfm], (0.4, 0.6), 0.0, last)
    if "d" in str(view["freq"]).lower() and "w" not in str(view["freq"]).lower():
        return _blend([toto, tfm, sea], (0.7, 0.2, 0.1), 0.3, last)
    return _blend([tfm, sea], (0.7, 0.3), 0.0, last)


# Evidence correction is deliberately separate from the existing raw controller.
# Missing or conflicting evidence has zero strength; all corrections are bounded.
_evidence_base_forecast = forecast


def forecast(view):
    import math
    from statistics import median

    def number(value, default=None):
        if isinstance(value, bool):
            return default
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return value if math.isfinite(value) else default

    horizon = max(0, int(view.get('H', 0)))
    raw = list(_evidence_base_forecast(view))
    history = [number(x) for x in view.get('history', [])]
    history = [x for x in history if x is not None]
    last = history[-1] if history else 0.0
    baseline = [number(raw[i], last) if i < len(raw) else last
                for i in range(horizon)]
    if not horizon or len(history) < 3:
        return baseline

    recent = history[-max(horizon, 16):]
    center = median(recent)
    scale = max(median([abs(x - center) for x in recent]),
                median([abs(b - a) for a, b in zip(recent, recent[1:])]))
    if not math.isfinite(scale) or scale <= 0:
        return baseline

    # Reliability evidence only interpolates toward the supplied alternative.
    # Its influence increases smoothly with the fraction of unreliable history.
    methods = view.get('method_forecasts', {})
    alternative = methods.get('timesfm_2_5_reliability')
    fraction = number(view.get('reliability_fraction'))
    weight = 0.0
    if (view.get('reliability_available') and fraction is not None
            and 0 <= fraction <= 1 and isinstance(alternative, (list, tuple))
            and len(alternative) == horizon):
        weight = 0.15 * (1 - fraction)
    corrected = list(baseline)
    if weight:
        for i in range(horizon):
            target = number(alternative[i])
            if target is not None:
                delta = weight * (target - baseline[i])
                corrected[i] += max(-0.15 * scale, min(0.15 * scale, delta))

    # Only explicit structured directions with explicit high confidence count.
    # Averaging, rather than summing, prevents repeated events increasing impact.
    documents = view.get('documents', [])
    if isinstance(documents, dict):
        documents = [documents]
    if not isinstance(documents, (list, tuple)):
        return corrected
    votes = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        events = document.get('events', [])
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, (list, tuple)):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            confidence = number(event.get('confidence'))
            if confidence is None or not 0.75 < confidence <= 1:
                continue
            direction = event.get('direction')
            if isinstance(direction, str):
                direction = direction.strip().lower()
                if direction in ('up', 'increase', 'positive'):
                    direction = 1.0
                elif direction in ('down', 'decrease', 'negative'):
                    direction = -1.0
                else:
                    continue
            else:
                direction = number(direction)
                if direction not in (-1.0, 1.0):
                    continue
            votes.append(direction * (confidence - 0.75) / 0.25)
    if not votes:
        return corrected
    # Any directional conflict makes the document operator abstain.
    if min(votes) < 0 < max(votes):
        return corrected
    evidence = sum(votes) / len(votes)
    # Do not double-count a direction already expressed by the forecast.
    move = median(corrected) - last
    if evidence * move >= 0:
        return corrected
    shift = evidence * min(0.03 * scale, abs(move))
    return [float(x + shift) for x in corrected]


# Preserve the controller's document and reliability routing. Before routing,
# temper only a strongly isolated TSFM path, using agreement among the other
# models and the observed history scale. Ordinary disagreements remain raw.
_forecast_before_robust_routing = forecast


def forecast(view):
    import math
    import statistics

    horizon = int(view['H'])
    methods = view.get('method_forecasts', {})
    history = view.get('history', [])
    usable = {}
    for name, values in methods.items():
        if name == 'timesfm_2_5_reliability':
            continue
        try:
            path = [float(value) for value in values]
        except (TypeError, ValueError, OverflowError):
            continue
        if len(path) == horizon and all(math.isfinite(x) for x in path):
            usable[name] = path
    try:
        recent = [float(value) for value in history[-max(32, horizon):]]
    except (TypeError, ValueError, OverflowError):
        recent = []
    recent = [value for value in recent if math.isfinite(value)]
    if horizon <= 0 or len(usable) < 3 or len(recent) < 8:
        return _forecast_before_robust_routing(view)

    median = statistics.median
    center = [median(path[t] for path in usable.values())
              for t in range(horizon)]
    distance = {name: sum(abs(x - y) for x, y in zip(path, center)) / horizon
                for name, path in usable.items()}
    level = median(recent)
    history_scale = median(abs(x - level) for x in recent)
    step_scale = median(abs(y - x) for x, y in zip(recent, recent[1:]))
    # A volatile or trending history widens the acceptance region. Requiring
    # separation from typical model disagreement protects plausible alternatives.
    scale = max(history_scale, step_scale, median(distance.values()))
    if scale == 0:
        # Exact agreement on a constant series supplies no positive scale with
        # which to judge the remaining path; leave the original route intact.
        return _forecast_before_robust_routing(view)
    threshold = 4 * scale
    adjusted = dict(methods)
    changed = False
    for name, path in usable.items():
        excess = distance[name] - threshold
        if excess > 0:
            strength = 0.05 * excess / distance[name]
            adjusted[name] = [(1 - strength) * x + strength * y
                              for x, y in zip(path, center)]
            changed = True
    if not changed:
        return _forecast_before_robust_routing(view)
    routed_view = dict(view)
    routed_view['method_forecasts'] = adjusted
    return _forecast_before_robust_routing(routed_view)


# A bounded consensus safeguard: only route away from an isolated excursion
# when the recent history also supports the central model forecasts.
_routing_base_forecast = forecast


def forecast(view):
    import math
    from statistics import median

    base = _routing_base_forecast(view)
    horizon = int(view.get('H', len(base)))
    if horizon <= 0:
        return []
    try:
        strength = float(view.get('routing_strength', 1.0))
    except (TypeError, ValueError):
        strength = 1.0
    if not math.isfinite(strength) or strength <= 0:
        return base
    strength = min(strength, 1.0)
    if view.get('operator') in ('none', 'raw'):
        return base
    history = view.get('history', [])
    try:
        recent = [float(value) for value in history[-96:]]
    except (TypeError, ValueError, OverflowError):
        return base
    if len(recent) < 24 or not all(math.isfinite(v) for v in recent):
        return base
    supplied = view.get('method_forecasts', {})
    if not isinstance(supplied, dict):
        return base
    models = []
    for name, values in supplied.items():
        # Reliability forecasts are a separate operator, not independent votes.
        if 'reliability' in str(name).lower():
            continue
        try:
            path = [float(v) for v in values]
        except (TypeError, ValueError, OverflowError):
            continue
        if len(path) == horizon and all(math.isfinite(v) for v in path):
            if path not in models:
                models.append(path)
    if len(models) < 3:
        return base
    center = median(recent)
    scale = median(abs(v - center) for v in recent)
    step = median(abs(b - a) for a, b in zip(recent, recent[1:]))
    scale = max(scale, step)
    if scale <= 0:
        return base
    half = len(recent) // 2
    drift = abs(median(recent[half:]) - median(recent[:half])) / scale
    # Abstain in changing regimes, where consensus can miss a real transition.
    stability = max(0.0, 1.0 - drift)
    if stability == 0:
        return base
    result = list(base)
    for index in range(horizon):
        votes = [path[index] for path in models]
        consensus = median(votes)
        spread = median(abs(v - consensus) for v in votes)
        radius = max(3.0 * spread, 2.0 * scale)
        offset = base[index] - consensus
        if abs(offset) <= radius or abs(consensus - center) > 2.0 * scale:
            continue
        target = consensus + math.copysign(radius, offset)
        # Small convex adjustment preserves the existing dataset routing and
        # document/reliability operators while limiting isolated model tails.
        weight = 0.05 * strength * stability
        result[index] = float(base[index] + weight * (target - base[index]))
    return result


# Apply only a small reliability blend corroborated by structured event direction.
_forecast_before_evidence = forecast


def forecast(view):
    base = list(_forecast_before_evidence(view))
    horizon = max(0, int(view.get('H', 0)))
    history = view.get('history') or []
    import math

    def number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            return None

    past = [number(value) for value in history]
    past = [value for value in past if value is not None]
    last = past[-1] if past else 0.0
    base = [number(value) for value in base[:horizon]]
    base = [last if value is None else value for value in base]
    base += [base[-1] if base else last] * (horizon - len(base))
    if not horizon or not view.get('reliability_available'):
        return base
    fraction = number(view.get('reliability_fraction'))
    if fraction is None or fraction <= 0:
        return base
    reliable = (view.get('method_forecasts') or {}).get('timesfm_2_5_reliability')
    if not isinstance(reliable, (list, tuple)) or len(reliable) < horizon:
        return base
    reliable = [number(value) for value in reliable[:horizon]]
    if any(value is None for value in reliable):
        return base

    documents = view.get('documents') or []
    if isinstance(documents, dict):
        documents = [documents]
    votes = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        events = document.get('events') or []
        if isinstance(events, dict):
            events = [events]
        for event in events:
            if not isinstance(event, dict):
                continue
            direction = event.get('direction')
            if isinstance(direction, str):
                direction = direction.strip().lower()
                if direction in ('up', 'increase', 'positive'):
                    direction = 1.0
                elif direction in ('down', 'decrease', 'negative'):
                    direction = -1.0
                else:
                    direction = None
            direction = number(direction)
            if direction is not None:
                votes.append(max(-1.0, min(1.0, direction)))
    if not votes:
        return base
    # Averaging prevents document volume from amplifying the intervention;
    # contradictory and neutral events reduce it continuously to zero.
    consensus = sum(votes) / len(votes)
    innovation = sum((other - value) / horizon
                     for value, other in zip(base, reliable))
    if not math.isfinite(innovation) or consensus * innovation <= 0:
        return base
    recent = past[-max(horizon, 2):]
    changes = sorted(abs(right - left) for left, right in zip(recent, recent[1:]))
    scale = changes[len(changes) // 2] if changes else 0.0
    if not math.isfinite(scale) or scale <= 0:
        return base
    # One common weight preserves the reliability trajectory and bounds every
    # point's movement by a small fraction of typical historical variation.
    distance = max(abs(other - value) for value, other in zip(base, reliable))
    if not math.isfinite(distance) or distance <= 0:
        return base
    weight = min(0.05 * min(1.0, fraction) * abs(consensus),
                 0.02 * scale / distance)
    return [float(value + weight * (other - value))
            for value, other in zip(base, reliable)]


# Limit persistence shrinkage to unsupported excursions; exact raw paths pass
# through unchanged. The wrapper retains the existing controller's gates.
def _with_excursion_guard(controller):
    def guarded(view):
        result = controller(view)
        if view.get('strength') == 0 or view.get('operator') in ('none', 'raw'):
            return result
        methods = view.get('method_forecasts') or {}
        if any(result == values for values in methods.values()):
            return result
        history = view.get('history') or []
        if len(history) < 8 or not result:
            return result
        import math
        recent = history[-max(8, len(result) * 2):]
        if not all(isinstance(x, (int, float)) and math.isfinite(x)
                   for x in recent + result):
            return result
        steps = [abs(b - a) for a, b in zip(recent, recent[1:])]
        travel = sum(steps)
        if travel == 0:
            return result
        # Persistent movement makes last-value shrinkage unreliable. Fade the
        # guard continuously to zero as directional efficiency approaches one.
        efficiency = abs(recent[-1] - recent[0]) / travel
        strength = 0.05 * (1.0 - efficiency) ** 2
        scale = max(max(recent) - min(recent), travel / len(steps))
        last = recent[-1]
        guarded_result = []
        for value in result:
            distance = abs(value - last)
            excess = max(0.0, distance - 3.0 * scale)
            weight = strength * excess / distance if distance else 0.0
            guarded_result.append(float((1.0 - weight) * value + weight * last))
        return guarded_result
    return guarded


forecast = _with_excursion_guard(forecast)


_previous_forecast = forecast


def forecast(view):
    """Retain routing, softly reducing unsupported ensemble outliers."""
    import math
    from statistics import median

    result = _previous_forecast(view)
    horizon = int(view.get('H', len(result)))
    history = view.get('history', [])
    methods = view.get('method_forecasts', {})
    if horizon <= 0 or len(history) < 16:
        return result

    def finite_number(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    recent = history[-64:]
    if not all(finite_number(value) for value in recent):
        return result
    paths = [path for name, path in methods.items()
             if 'reliability' not in str(name).lower()
             and isinstance(path, (list, tuple)) and len(path) >= horizon
             and all(finite_number(value) for value in path[:horizon])]
    if len(paths) < 3:
        return result
    center = median(recent)
    level_scale = median(abs(value - center) for value in recent)
    differences = [b - a for a, b in zip(recent, recent[1:])]
    drift = median(differences)
    noise = median(abs(value - drift) for value in differences)
    scale = max(level_scale, noise)
    if scale <= 0:
        return result
    # Trend and long-horizon uncertainty widen the agreement envelope.
    # A majority must agree before an isolated route loses any weight.
    routed = list(result)
    for step in range(min(horizon, len(routed))):
        value = routed[step]
        if not finite_number(value):
            continue
        votes = [path[step] for path in paths]
        consensus = median(votes)
        spread = median(abs(vote - consensus) for vote in votes)
        uncertainty = scale + noise * math.sqrt(step + 1) + abs(drift) * (step + 1)
        envelope = 4 * max(uncertainty, spread)
        distance = abs(value - consensus)
        excess = max(0.0, distance - envelope)
        if excess > 0:
            strength = 0.1 * excess / distance
            routed[step] = float((1 - strength) * value + strength * consensus)
    return routed
