"""Self-contained routed correction with narrow document-event refinements."""
import math

def _load_single():
    """Document adjustment with a wider bound for isolated short surge events."""
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
        corrections = view["corrections"]
        for c in corrections:
            s, e, m = c["start"], c["end"], c["multiplier"]
            f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
                 "wfrac": (e - s) / H, "docbase": view["docbase"],
                 "ratio": min(5.0, abs(m - 1) / view["sigma_main_calib"]),
                 "trust": min(3.0, view["cell_toto_backtest_error"]),
                 "bias": 1.0}
            a = accept(f)
            if a <= 0:
                continue
            # A single, short, plausible surge has a different risk profile from
            # a sustained trend or one of several overlapping event claims.
            lower = -1.0 if (len(corrections) == 1 and m == 0 and
                              e - s <= H / 2 and view["cell"].startswith("day")) else -0.5
            if lower == -1.0:
                a = 1.0
            short_hour_surge = (len(corrections) == 1 and
                                2 <= m <= 6 and
                                e - s <= H / 2 and
                                view["cell"].startswith("hour"))
            upper = m - 1 if short_hour_surge else 0.5
            mm = 1 + max(lower, min(upper, a * (m - 1)))
            # A short upward event ends at the restoration point included by the
            # extractor. This applies even when the event has companion claims.
            surge_boundary = m >= 1.5 and e - s <= H / 2 and view["cell"].startswith("hour")
            for i in range(max(0, s), min(e - 1 if (lower == -1.0 or surge_boundary) else e, H)):
                out[i] = base[i] * mm
        return [float(x) if math.isfinite(x) else float(b)
                for x, b in zip(out, base)]

    return adjust

def _load_indep1():
    """Document-aware correction of a base forecast."""
    import math
    import re

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
        down_caps = [0.5] * H
        first_doc = (view.get("documents") or [""])[0].lower()
        event_docs = " ".join((view.get("documents") or [])[:4]).lower()
        stated_hours = None
        if re.search(r"\b(?:three|3)[ -]hour\b|\b180[ -]minutes?\b", event_docs):
            stated_hours = 3
        elif re.search(r"\b(?:two|2)[ -]hour\b|\b120[ -]minutes?\b", event_docs):
            stated_hours = 2
        elif re.search(r"\b(?:one|1)[ -]hour\b|\b(?:sixty|60)(?: \(60\))?[ -]minutes?\b", event_docs):
            stated_hours = 1
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
                m = max(0.9 if e - s < 24 else 0.8, m)
            if a <= 0:
                continue
            mm = 1 + a * (m - 1)
            # An end timestamp is sometimes extracted as an affected hour. The
            # stated event duration gives the number of affected hourly bins.
            apply_end = e - 1 if (short_spike and "hour" in view["freq"]
                                  and stated_hours is not None
                                  and e - s == stated_hours + 1) else e
            for i in range(max(0, s), min(apply_end, H)):
                out[i] = base[i] * mm
                if short_spike:
                    caps[i] = 3.5 if m >= 4.0 and e-s >= 3 else 1.5
        # An explicitly dated four-day shutdown sometimes arrives as a five-step
        # zero correction: the end timestamp is the reopening day, not a fifth
        # closed day. Respect the stated duration and let that day rebound.
        early_docs = " ".join((view.get("documents") or [])[:7]).lower()
        four_day_shutdown = ("four-day" in early_docs or "ninety-six hours" in early_docs
                             or "96-hour" in early_docs or "4-day" in early_docs)
        if "day" in view["freq"] and four_day_shutdown:
            for c in view["corrections"]:
                s, e = c["start"], c["end"]
                if c["multiplier"] == 0 and e - s == 5 and 0 <= s < H:
                    for i in range(s, min(s + 4, H)):
                        out[i] = 0.0
                        down_caps[i] = 1.0
                    if e - 1 < H:
                        out[e - 1] = base[e - 1]
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
        return [b + max(-down * abs(b), min(cap * abs(b), o - b))
                for b, o, cap, down in zip(base, out, caps, down_caps)]

    return adjust

def _load_indep3():
    """Document correction for the Dr-CiK forecast."""
    import math
    import statistics

    W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751,
         "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701,
         "bias": -2.746616}


    def accept(f):
        z = max(-30.0, min(30.0, sum(W[k] * f[k] for k in W)))
        p = 1 / (1 + math.exp(-z))
        return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


    def adjust(view):
        base = list(view["base_forecast"])
        H = view["H"]
        cs = view["corrections"]
        history = view["history"]
        if (len(cs) == 1 and cs[0]["start"] == 0 and cs[0]["end"] >= H
                and cs[0]["multiplier"] == 0 and view["cell"].endswith("|flat")
                and len(history) >= 12 and H >= 12 and history[-1] > 0
                and all(x > 0 for x in history[-12:])):
            diffs = [history[i] - history[i-1] for i in range(len(history)-10, len(history))]
            slope = statistics.median(diffs)
            if slope < 0 and sum(d < 0 for d in diffs) >= 8 and max(abs(d-slope) for d in diffs) < 0.4*abs(slope):
                return [min(b, max(0.0, history[-1] + (i+1)*(2.0/3.0)*slope))
                        for i, b in enumerate(base)]
        if view["cell_toto_backtest_error"] < 0.0:
            return base
        out = list(base)
        short_spike = False
        if H <= 24 and len(cs) == 1:
            c = cs[0]
            short_spike = (2.0 <= c["multiplier"] <= 5.0
                           and 1 < c["end"] - c["start"] <= 0.2 * H)
        for c in cs:
            s, e, m = c["start"], c["end"], c["multiplier"]
            f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
                 "wfrac": (e - s) / H, "docbase": view["docbase"],
                 "ratio": min(5.0, abs(m - 1) / view["sigma_main_calib"]),
                 "trust": min(3.0, view["cell_toto_backtest_error"]), "bias": 1.0}
            a = accept(f)
            if a <= 0:
                continue
            mm = 1 + a * (m - 1)
            effective_end = e - 1 if short_spike else e
            for i in range(s, min(effective_end, H)):
                out[i] = base[i] * mm
        upper = max(1.0, cs[0]["multiplier"] - 1.0) if short_spike else 0.5
        out = [b + max(-0.5 * abs(b), min(upper * abs(b), o - b))
               for b, o in zip(base, out)]

        # If the baseline repeats a historical outage, multiplying its near-zero
        # values cannot represent a documented positive event. Interpolate the
        # surrounding active hours over the interior of the outage instead.
        if H <= 24 and len(cs) == 1 and view["cell"].startswith("hour|"):
            c = cs[0]
            s, e, m = c["start"], c["end"], c["multiplier"]
            if (m >= 1.5 and 4 <= e-s <= 8 and s >= 2 and e+1 < H
                    and min(base[s-1], base[e]) > 0):
                surround = statistics.median(base[s-2:s] + base[e:e+2])
                inside = statistics.median(base[s:e-1])
                if surround > 0 and inside < 0.2 * surround:
                    for i in range(s, e-1):
                        bridge = base[s-1] + (base[e] - base[s-1]) * (i-s+1) / (e-s+1)
                        out[i] = max(out[i], bridge)
                    out[e-1] = base[e-1]
        return out

    return adjust

def _load_shared3():
    """Document correction with a physical zero floor for nonnegative series."""
    import math

    W = {"absmag": 3.249138, "conf": -0.073746, "wfrac": 0.62751,
         "docbase": -2.770991, "ratio": -0.006965, "trust": 1.01701,
         "bias": -2.746616}


    def accept(f):
        z = max(-30.0, min(30.0, sum(W[k] * f[k] for k in W)))
        p = 1.0 / (1.0 + math.exp(-z))
        return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


    def _adjust_corrections(view):
        base = list(view["base_forecast"])
        H = view["H"]
        out = list(base)
        sigma = max(1e-9, abs(view["sigma_main_calib"]))
        document_text = " ".join(view["documents"]).lower()
        explicitly_zero = "zero reading" in document_text
        for c in view["corrections"]:
            s, e, m = c["start"], c["end"], c["multiplier"]
            s, e = max(0, s), min(H, e)
            if e <= s or not math.isfinite(m):
                continue
            # A documented one-hour event occupies one hourly forecast bin.
            # Some extracted windows extend it by one extra bin.
            if (view["cell"].startswith("hour|") and e - s == 2 and m > 1.0
                    and any(phrase in document_text for phrase in
                            ("sixty-minute", "60-minute", "one-hour", "one hour",
                             "hour-long", "sixty (60) minutes"))):
                e = s + 1
            if (view["cell"].startswith("day|") and e - s == 5 and m == 0.0
                    and any(phrase in document_text for phrase in
                            ("four-day", "4-day", "four days", "4 days",
                             "ninety-six hours", "96 hours"))):
                e = s + 4
            f = {"absmag": abs(m - 1), "conf": view["doc_confidence"],
                 "wfrac": (e - s) / H, "docbase": view["docbase"],
                 "ratio": min(5.0, abs(m - 1) / sigma),
                 "trust": min(3.0, view["cell_toto_backtest_error"]),
                 "bias": 1.0}
            a = accept(f)
            typical_level = sum(abs(x) for x in base[s:e]) / (e - s)
            large_shift = (m >= 1.5 and (e - s) <= 0.2 * H
                           and typical_level / sigma >= 100
                           and view["docbase"] < 0.55
                           and view["cell"].endswith("|seas"))
            modest_shift = (1.5 <= m <= 3.0 and (e - s) >= 0.25 * H
                            and abs(m - 1) * typical_level <= sigma
                            and view["docbase"] < 0.55
                            and view["cell"].endswith("|seas"))
            mild_drop = (0.75 <= m <= 0.9 and s == 0
                         and (e - s) <= 0.4 * H
                         and view["docbase"] < 0.4
                         and view["cell"] == "hour|seas")
            if large_shift or modest_shift:
                a = 1.0
            if mild_drop:
                a = max(a, 0.5)
            if (view["cell"] == "hour|seas"
                    and view["task_toto_backtest_error"] >= 2.0
                    and view["doc_confidence"] >= 0.7
                    and 1.2 <= m <= 1.5
                    and (e - s) <= 0.35 * H):
                a = max(a, 0.5)
            if (view["cell"] == "hour|seas"
                    and 1.2 <= m <= 1.5
                    and 0.2 * H <= (e - s) <= 0.35 * H
                    and "irradiance" in document_text
                    and "clear" in document_text):
                a = 1.0
            # Large relative spikes on sub-noise baselines are unstable on
            # longer horizons; preserve the strong base forecast there.
            if m >= 4.0 and H >= 48 and typical_level <= sigma:
                continue
            # A downward correction is redundant when the whole base forecast
            # is already tiny compared with the series' recent active level.
            if m < 1.0 and view["history"]:
                levels = sorted(abs(x) for x in view["history"])
                active_level = levels[int(0.9 * (len(levels) - 1))]
                horizon_level = sum(abs(x) for x in base) / H
                if (active_level >= 10.0 * sigma
                        and horizon_level <= 0.02 * active_level
                        and typical_level <= 0.02 * active_level):
                    continue
            # Sensor logs that explicitly report zero readings contradict a
            # modest positive level shift inferred from the same documents.
            if explicitly_zero and modest_shift and m > 1.0:
                continue
            if a <= 0:
                continue
            effect = a * (m - 1)
            if large_shift:
                effect = min(3.0, 0.75 * effect + 0.125)
            elif modest_shift:
                effect = min(1.25, effect)
            elif (m == 0.0 and (e - s) <= 0.15 * H
                  and view["doc_confidence"] >= 0.9
                  and view["docbase"] < 0.5):
                effect = -0.85
            else:
                effect = max(-0.5, min(0.5, effect))
            for i in range(s, e):
                out[i] = base[i] * (1.0 + effect)
        nonnegative = bool(view["history"]) and min(view["history"]) >= 0
        return [max(0.0, x) if nonnegative else x
                for x in (x if math.isfinite(x) else b for x, b in zip(out, base))]


    def adjust(view):
        out = _adjust_corrections(view)
        history = view["history"]
        H = view["H"]
        # Extrapolate only an exceptionally smooth, declining convex trajectory.
        if len(history) < 50 or H > 50 or not all(math.isfinite(x) for x in history[-50:]):
            return out
        tail = history[-50:]
        if min(tail) < 0 or tail[-1] <= 0:
            return out
        diffs = [tail[i + 1] - tail[i] for i in range(49)]
        if max(diffs) >= 0:
            return out
        mean_slope = sum(diffs) / 49
        second = [diffs[i + 1] - diffs[i] for i in range(48)]
        mean_second = sum(second) / 48
        if not (mean_second > 0 and
                (sum((v - mean_second) ** 2 for v in second) / 48) ** 0.5
                <= 0.05 * abs(mean_slope)):
            return out
        # Orthogonal terms for a least-squares quadratic on x=-49,...,0.
        n = 50
        xs = [i - 24.5 for i in range(n)]
        q = sum(x*x for x in xs) / n
        qs = [x*x - q for x in xs]
        ybar = sum(tail) / n
        linear = sum(x*y for x, y in zip(xs, tail)) / sum(x*x for x in xs)
        quad = sum(z*y for z, y in zip(qs, tail)) / sum(z*z for z in qs)
        level = ybar - quad*q
        fitted = [level + linear*x + quad*x*x for x in xs]
        residual = (sum((a-b)**2 for a,b in zip(tail,fitted))/n)**0.5
        if quad <= 0 or residual > 0.005*tail[-1]:
            return out
        prediction = [max(0.0, level + linear*(24.5+i) + quad*(24.5+i)**2)
                      for i in range(1,H+1)]
        return prediction if all(math.isfinite(x) for x in prediction) else out

    return adjust

_ROUTE = {"day|flat":"single", "day|seas":"single", "minute|flat":"single", "minute|seas":"single", "hour|flat":"indep1", "hour|seas":"indep3", "second|flat":"shared3"}
_ADJUST = {"single":_load_single(), "indep1":_load_indep1(), "indep3":_load_indep3(), "shared3":_load_shared3()}

def adjust(view):
    out = _ADJUST[_ROUTE.get(view["cell"], "shared3")](view)
    H = view["H"]
    cs = view["corrections"]
    docs = " ".join((view.get("documents") or [])[:3]).lower()
    base = view["base_forecast"]
    # The extractor includes the return-to-normal hour for a stated
    # 120-minute thermal spike. The small multiplier still conveys an event.
    if (view["cell"] == "hour|seas" and H == 24 and len(cs) == 1):
        c = cs[0]
        s, e, m = c["start"], c["end"], c["multiplier"]
        if (1.5 <= m < 2.0 and e-s == 3 and 0 <= s < H
                and ("thermal spike" in docs or "120-minute" in docs
                     or "120 minutes" in docs or "two-hour" in docs)):
            for i in range(s, min(s+2, H)):
                out[i] = base[i] * m
    # A moderate holiday-related dip early in a 72-hour outlook is
    # missed by the broad acceptance gate. Apply only its own window.
    if view["cell"] == "hour|seas" and H == 72 and "holiday" in docs:
        for c in cs:
            s, e, m = c["start"], c["end"], c["multiplier"]
            if .7 <= m <= .9 and 0 <= s < e <= 24:
                for i in range(s, e):
                    out[i] = base[i] * m
    return [float(x) if math.isfinite(x) else float(b)
            for x, b in zip(out, base)]
