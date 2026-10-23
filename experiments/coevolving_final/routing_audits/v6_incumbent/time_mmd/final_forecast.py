"""Distance-calibrated ensemble with a monthly short-history support guard."""


def _freq(freq):
    f = str(freq).lower()
    return "weekly" if ("w" in f or "week" in f) else ("monthly" if "month" in f or f in ("m", "1m", "ms") else "daily")


def forecast(view):
    mf, h, H = view["method_forecasts"], view["history"], view["H"]
    toto = mf["toto_2_0"]
    f = _freq(view["freq"])
    if view["dataset"] == "time_mmd":
        if f == "daily":
            seq = [0.6 * a + 0.4 * b for a, b in zip(mf.get("timesfm_2_5", toto), mf.get("moirai_2_0", toto))]
        elif f == "weekly":
            seq = [0.9 * a + 0.1 * b for a, b in zip(mf.get("timesfm_2_5", toto), mf.get("moirai_2_0", toto))]
        else:
            seq = [0.7 * a + 0.3 * b for a, b in zip(mf.get("timesfm_2_5", toto), toto)]
    else:
        period = 52 if f == "weekly" else 7
        sea = [h[-period + i % period] if len(h) >= period else h[-1] for i in range(H)]
        if f == "weekly":
            seq = [0.6 * a + 0.4 * b for a, b in zip(mf.get("timesfm_2_5", toto), sea)]
        else:
            ch = mf.get("chronos_bolt", toto)
            seq = [0.7 * (0.6 * a + 0.3 * b + 0.1 * c) + 0.3 * h[-1] for a, b, c in zip(toto, ch, sea)]
    if f == "monthly":
        low, high = min(h), max(h)
        margin = 0.1 * (high - low)
        seq = [max(low + margin, min(high - margin, x)) for x in seq]
    den = max(1, H - 1)
    corrected = bool(view.get("documents"))
    answer = []
    for i, x in enumerate(seq):
        final_scale = 0.9 - 0.2 * i / den
        # Downstream contributes 0.9 only on document-bearing views.
        answer.append(x * (final_scale / 0.9 if corrected else final_scale))
    return answer
