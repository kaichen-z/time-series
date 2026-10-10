"""Seed Retrieval module (L4 start point): turn the audited Haiku event cards of the task's documents into future
corrections using only information available at the forecast origin.

Two leakage-free mappings:
  1. dated-window events: if the view has future_timestamps (TimesX) and an event's [time_start, time_end] overlaps
     them (e.g. a known holiday), the correction covers exactly those forecast steps;
  2. recent events: the documents are written before the forecast origin, so "recent" is measured relative to the
     latest event date found in the task's own documents (no future timestamp needed; works on Time-MMD, whose views
     carry no timestamps). Events within RECENT_DAYS of that date carry their direction into the first third of the
     horizon.
Only events with direction up/down and confidence >= 0.6 are used; multipliers are small and confidence-scaled.
retrieve(view) -> list of dict(start, end, multiplier, confidence, source, kind) with 0 <= start < end <= H."""
import datetime as _dt

MIN_CONF, STEP_DATED, STEP_RECENT = 0.6, 0.05, 0.03


def _d(s):
    try: return _dt.date.fromisoformat(str(s)[:10])
    except Exception: return None


def _window(fut, t0, t1):
    t1 = t1 or t0
    idx = [i for i, d in enumerate(fut) if t0 <= d <= t1]
    return (min(idx), max(idx) + 1) if idx else None


def retrieve(view):
    H = view["H"]; fut = [_d(t) for t in view.get("future_timestamps") or []]; fut = [d for d in fut if d]
    events = [(doc["document_id"], e) for doc in view.get("documents") or [] for e in doc.get("events") or []
              if e.get("direction") in ("up", "down") and (e.get("confidence") or 0) >= MIN_CONF]
    dated = [d for _, e in events for d in (_d(e.get("time_start")), _d(e.get("time_end"))) if d]
    if fut: dated = [d for d in dated if d < fut[0]]  # recency anchor only from pre-origin dates
    anchor = max(dated) if dated else None
    recent_days = 14 if any(k in str(view["freq"]).lower() for k in ("d", "w")) and "month" not in str(view["freq"]).lower() else 45
    out = []
    for src, e in events:
        sign = 1 if e["direction"] == "up" else -1; conf = float(e["confidence"]); t0, t1 = _d(e.get("time_start")), _d(e.get("time_end"))
        w = _window(fut, t0, t1) if (fut and t0) else None
        if w:
            out.append(dict(start=w[0], end=w[1], multiplier=1 + sign * STEP_DATED * conf, confidence=conf, source=src, kind="dated_window"))
            continue
        last = t1 or t0
        if anchor and last and (anchor - last).days <= recent_days and (not fut or last < fut[0]):
            out.append(dict(start=0, end=max(1, -(-H // 3)), multiplier=1 + sign * STEP_RECENT * conf, confidence=conf, source=src, kind="recent"))
    return out
