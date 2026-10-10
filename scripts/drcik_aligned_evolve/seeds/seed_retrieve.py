"""Seed Retrieval module (L4 start point): turn the audited Haiku event cards of the task's documents into future
corrections. An event becomes a correction only if it has a direction (up/down), confidence >= 0.6 and a date range
that overlaps the forecast window (view["future_timestamps"]); the multiplier is a small, confidence-scaled move.
retrieve(view) -> list of dict(start, end, multiplier, confidence, source) with 0 <= start < end <= H."""


def _window(view, t0, t1):
    fut = [str(t)[:10] for t in view.get("future_timestamps") or []]
    if not fut or not t0: return None
    t1 = t1 or t0
    idx = [i for i, d in enumerate(fut) if t0 <= d <= t1]
    if not idx:  # event before the window but still ongoing at its start
        if t0 < fut[0] <= t1: idx = [0]
        else: return None
    return min(idx), max(idx) + 1


def retrieve(view):
    out = []
    for doc in view.get("documents") or []:
        for e in doc.get("events") or []:
            if e.get("direction") not in ("up", "down") or (e.get("confidence") or 0) < 0.6: continue
            w = _window(view, e.get("time_start"), e.get("time_end"))
            if not w: continue
            step = 0.05 * float(e["confidence"])
            out.append(dict(start=w[0], end=w[1], multiplier=1 + step if e["direction"] == "up" else 1 - step,
                            confidence=float(e["confidence"]), source=doc["document_id"]))
    return out
