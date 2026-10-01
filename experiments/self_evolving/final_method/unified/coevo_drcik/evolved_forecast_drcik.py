"""Seed base forecaster = the current unified framework's numerical step (per-dataset choices made on each train set).
- drcik : Dr-CiK-evolved program 0.854*Toto(repaired history if accepted) + 0.046*ARIMA + 0.100*TimesFM, shrink 0.2.
- tmmd  : 0.4*Toto + 0.6*TimesFM.
- timesx: daily 0.7*Toto + 0.2*TimesFM + 0.1*seasonal, shrink 0.3; weekly 0.7*TimesFM + 0.3*seasonal.
- cold start (variable never seen in train, view["seen_variable"] is False): choose (Toto, TimesFM, seasonal) weights
  and shrink by the task's own history backtests (view["backtests"]).
forecast(view) -> list of H floats."""
import itertools, math


def season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f: return None
    for k, p in (("hour", 24), ("week", 52), ("w", 52), ("month", 12), ("day", 7), ("daily", 7), ("d", 7)):
        if k in f: return p
    return None


def seasonal(h, H, p):
    return [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]


def blend(members, w, s, last):
    return [(1 - s) * sum(wi * m[i] for wi, m in zip(w, members)) + s * last for i in range(len(members[0]))]


def jerr(f, y):
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5, mae / sc) + min(5, rmse / sc)


GRID = [(tuple(x / 10 for x in w), s) for w in itertools.product(range(11), repeat=3) if sum(w) == 10 for s in (0.0, 0.1, 0.2, 0.3)]


def forecast(view):
    mf = view["method_forecasts"]; H = view["H"]; h = view["history"]; p = season(view["freq"])
    toto = view.get("toto_forecast_repaired_history") or mf["toto_2_0"]; tfm = mf.get("timesfm_2_5", mf["toto_2_0"])
    sea = seasonal(h, H, p); last = h[-1]
    if not view.get("seen_variable", True) and view.get("backtests"):
        def score(cfg):
            w, s = cfg; e = 0.0
            for b in view["backtests"]:
                hh = h[:b["cut"]]
                e += jerr(blend([b["toto_2_0"], b["timesfm_2_5"], seasonal(hh, H, p)], w, s, hh[-1]), b["target"])
            return e
        w, s = min(GRID, key=score)
        return blend([toto, tfm, sea], w, s, last)
    ds = view["dataset"]
    if ds == "drcik":
        W = dict(toto=0.9904652444807432, arima=0.0531, tfm=0.11595169383119369); tot = sum(W.values())
        ar = mf.get("arima_auto", mf["toto_2_0"])
        f = [(W["toto"] * a + W["arima"] * b + W["tfm"] * c) / tot for a, b, c in zip(toto, ar, tfm)]
        base = [0.7984699167961635 * x + 0.20153008320833654 * last for x in f]
        repaired = view.get("toto_forecast_repaired_history")
        if repaired is not None:
            raw_toto = mf["toto_2_0"]
            repaired_weight = 0.7984699167961635 * W["toto"] / tot
            base = [x + 0.2 * repaired_weight * (r - t)
                    for x, r, t in zip(base, repaired, raw_toto)]
        backtests = view.get("backtests") or []
        if backtests:
            advantage = sum(jerr(b["toto_2_0"], b["target"]) -
                            jerr(b["timesfm_2_5"], b["target"])
                            for b in backtests) / len(backtests)
            if advantage > 0.1:
                base = [0.8 * x + 0.2 * y for x, y in zip(base, tfm)]
        # Blend a little of the independently fitted level for seasonal
        # series, with the reference model chosen by sampling frequency.
        if view.get("cell") == "day|seas":
            raw_toto = mf["toto_2_0"]
            base = [0.8 * x + 0.2 * y for x, y in zip(base, raw_toto)]
        elif view.get("cell") == "minute|seas":
            base = [0.9 * x + 0.1 * y for x, y in zip(base, ar)]
        if view.get("cell") == "day|flat" and "stl_ets" in mf:
            base = [0.80 * x + 0.20 * y for x, y in zip(base, mf["stl_ets"])]
        elif view.get("cell") == "day|seas" and "polynomial_trend_regression" in mf:
            base = [0.95 * x + 0.05 * y for x, y in zip(base, mf["polynomial_trend_regression"])]
        if view.get("cell") == "hour|seas" and "combined_moirai_croston_router" in mf:
            base = [0.95 * x + 0.05 * y for x, y in zip(base, mf["combined_moirai_croston_router"])]
        return base
    if ds == "tmmd":
        return blend([toto, tfm, sea], (0.4, 0.6, 0.0), 0.0, last)
    if str(view["freq"]).startswith("daily"):
        return blend([toto, tfm, sea], (0.7, 0.2, 0.1), 0.3, last)
    return blend([toto, tfm, sea], (0.0, 0.7, 0.3), 0.0, last)
