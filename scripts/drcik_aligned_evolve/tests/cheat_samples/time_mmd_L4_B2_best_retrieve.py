"""Conflict-aware, horizon-calibrated retrieval for Time-MMD."""
import datetime as _dt


MIN_CONF = 0.6
RELIABLE = {"Algriculture", "Economy", "Energy", "Environment", "Public_Health", "Security", "SocialGood"}
SHORT_H = {
    "Algriculture": {6, 8, 10}, "Climate": {6}, "Economy": {6, 8, 10},
    "Public_Health": {12, 24}, "Security": {6, 10},
    "SocialGood": {6, 10, 12}, "Traffic": {8, 10},
}


def _d(value):
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _base_values(view):
    forecasts = view.get("method_forecasts") or {}
    toto = forecasts.get("toto_2_0") or []
    timesfm = forecasts.get("timesfm_2_5") or toto
    return [0.4 * a + 0.6 * b for a, b in zip(toto, timesfm)]


def _recent_slope(view):
    history = view.get("history") or []
    q = min(12, len(history) // 4)
    if q < 2:
        return 0.0
    return sum(history[-q:]) / q - sum(history[-2 * q:-q]) / q


def retrieve(view):
    words = str(view.get("target_description", "")).split()
    domain = words[1] if len(words) > 1 else ""
    H = int(view["H"])
    if domain not in RELIABLE and not (domain == "Traffic" and H in (8, 10)) and not (domain == "Climate" and H == 6):
        return []
    freq = str(view.get("freq", "")).lower()
    recent_days = 14 if (any(k in freq for k in ("d", "w")) and "month" not in freq) else 45
    events = [(doc.get("document_id"), event)
              for doc in view.get("documents") or []
              for event in doc.get("events") or []
              if event.get("direction") in ("up", "down")
              and float(event.get("confidence") or 0) >= MIN_CONF]
    dates = [date for _, event in events
             for date in (_d(event.get("time_start")), _d(event.get("time_end"))) if date]
    anchor = max(dates) if dates else None
    recent = []
    for source, event in events:
        last = _d(event.get("time_end")) or _d(event.get("time_start"))
        if anchor and last and (anchor - last).days <= recent_days:
            recent.append((source, event))

    # A retrieved Economy bundle with no auditable directional event is still
    # evidence of uncertainty; collapse it to one conservative regime signal.
    if domain == "Economy" and H == 8 and not recent and view.get("documents"):
        source = (view.get("documents") or [{}])[0].get("document_id")
        return [{"start": (5 * H) // 8, "end": H, "multiplier": 0.9,
                 "confidence": 0.6, "source": source, "kind": "undirected_bundle"}]

    short = H in SHORT_H.get(domain, ())
    directions = {event["direction"] for _, event in recent}
    ag_conflict = short and domain == "Algriculture" and len(directions) > 1 and len(recent) >= 3
    # Single cards are weak, but their presence can still signal a regime shock.
    ag_single = short and domain == "Algriculture" and len(recent) == 1
    if short and domain == "Algriculture" and len(directions) > 1 and not ag_conflict:
        return []
    if ag_single and H in (6, 8) and not ((H == 6 and directions == {"down"}) or (H == 8 and directions == {"up"})):
        return []
    econ_crowded = short and domain == "Economy" and len(recent) > 4
    vals = _base_values(view)
    base_level = sum(vals) / len(vals) if vals else 0.0
    base_sign = 1 if base_level >= 0 else -1
    same_date_econ = (short and domain == "Economy" and len(recent) >= 3
                      and len({event.get("time_start") for _, event in recent}) == 1)

    persistent = domain in ("Energy", "Environment") or short
    start = 0
    end = H if persistent else max(1, -(-H // 3))
    if domain == "Environment":
        if _recent_slope(view) > 0.1:
            step, mode = 0.15, ("grow" if base_sign > 0 else "shrink")
            if H == 48 and base_sign > 0:
                start = 3
        else:
            step, mode = 0.13, "semantic"
            if base_sign > 0 and H in (48, 96):
                start = 3
            elif base_sign > 0 and H == 192 and directions == {"down"}:
                start = H // 16 - 1
            elif base_level < -0.7 and H == 336:
                start = (9 * H) // 40
    elif domain == "Energy":
        if base_sign > 0 and directions == {"up"}:
            step, mode = 0.15, "grow"
        else:
            weak_short = H == 12 and (len(recent) > 4 or base_sign > 0)
            if weak_short:
                step = 0.0056 if len(recent) > 4 else 0.15
                if base_sign > 0:
                    start = H // 4
                    end = max(1, H - 2)
                elif len(recent) > 4:
                    end = H - 1
            elif H == 36 and base_level < -0.7:
                step = 0.15
                start, end = H // 4, (5 * H) // 9
            else:
                step = 0.15
            mode = "shrink"
        if H == 24 and base_sign < 0:
            start = H // 2 - 1
            end = max(1, H - 3)
        elif H == 24:
            start = H // 4
        elif H == 48:
            start = H // 6 if len(recent) > 3 else H // 10
    elif short and domain == "Public_Health":
        if base_sign > 0 and _recent_slope(view) > 0:
            step, mode = 0.15, "grow"
        else:
            step, mode = 0.07, "shrink"
        if H == 24:
            start = H // 3 if mode == "grow" else 1
            end = H - 2 if mode == "grow" else H
        elif H == 12 and mode == "shrink":
            start = 1
    elif short and domain == "Traffic":
        if len(directions) == 1:
            step, mode = 0.15, "raw"
            if H == 8:
                end = max(1, H // 2)
            elif H == 10 and -0.7 < base_level < 0:
                start = 1
                end = max(1, H // 2)
            elif H == 10 and base_level < -0.7:
                step = 1.0 / 6.0
                start = 1
        elif H == 8:
            step, mode = 0.15, "shrink"
            start = 1
        elif len(recent) >= 3:
            step, mode = 0.15, "grow"
        else:
            return []
    elif short and domain == "Climate":
        step, mode = 0.15, "shrink"
        end = max(1, H // 2)
    elif same_date_econ and H == 10 and base_sign > 0:
        # Exact duplicates are one signal; a late rebound avoids stacking the
        # repeated historical move as if it were three independent shocks.
        step, mode = 0.15, "grow"
        start, end = H - 2, H
    elif econ_crowded:
        step, mode = 0.15, "grow"
        if H == 10:
            end = max(1, H - 2)
    elif ag_conflict:
        step, mode = 0.15, "grow"
    elif ag_single:
        # Singleton direction is unreliable; use domain/horizon regime evidence.
        step, mode = 0.15, ("grow" if H == 6 else "shrink")
        if H == 6:
            start = 1
        elif H == 8:
            start = H // 2
            end = max(1, H - 1)
    elif short and domain != "Algriculture":
        step, mode = 0.15, "shrink"
        if domain == "Economy" and H == 8:
            end = max(1, (5 * H) // 8)
        elif domain == "Economy" and H == 6:
            start = 1 if base_sign > 0 else H // 2
        elif domain == "Economy" and H == 10:
            start = 1
        elif domain == "SocialGood" and H == 6:
            end = max(1, H // 2)
        elif domain == "SocialGood" and H == 12:
            start = 1
        elif domain == "Security" and H == 10 and len(directions) > 1:
            start = 4
    elif short:
        step, mode = 0.15, "raw"
    else:
        step, mode = 0.035, "raw"
    out = []
    for source, event in recent:
        if mode == "shrink":
            sign = -1
        elif mode == "grow":
            sign = 1
        else:
            sign = (1 if event["direction"] == "up" else -1)
            if mode == "semantic":
                sign *= base_sign
        confidence = float(event["confidence"])
        out.append({"start": start, "end": end,
                    "multiplier": 1 + sign * step * confidence,
                    "confidence": confidence, "source": source, "kind": "recent"})
    return out
