"""Protocol v6 frozen REFERENCE forecaster (host-side baseline of the evolution score).
Pure numerical blend selected ONLY on official Train with forward-chaining (rolling-origin) CV, one weight vector
and shrink-to-last per dataset x frequency (TABLE).  Missing members fall back to Toto.  Must not change during a
run: its SHA-256 is recorded in stage.json and LOCK.json.  forecast(view) -> list of H floats."""

MEMBERS = ("toto_2_0", "timesfm_2_5", "moirai_2_0", "chronos_bolt", "seasonal")
TABLE = {
 "time_mmd": {
  "daily": {
   "shrink": 0.0,
   "weights": {
    "chronos_bolt": 0.0,
    "moirai_2_0": 0.4,
    "seasonal": 0.0,
    "timesfm_2_5": 0.6,
    "toto_2_0": 0.0
   }
  },
  "monthly": {
   "shrink": 0.0,
   "weights": {
    "chronos_bolt": 0.0,
    "moirai_2_0": 0.0,
    "seasonal": 0.0,
    "timesfm_2_5": 0.7,
    "toto_2_0": 0.3
   }
  },
  "weekly": {
   "shrink": 0.0,
   "weights": {
    "chronos_bolt": 0.0,
    "moirai_2_0": 0.1,
    "seasonal": 0.0,
    "timesfm_2_5": 0.9,
    "toto_2_0": 0.0
   }
  }
 },
 "timesx": {
  "daily": {
   "shrink": 0.3,
   "weights": {
    "chronos_bolt": 0.3,
    "moirai_2_0": 0.0,
    "seasonal": 0.1,
    "timesfm_2_5": 0.0,
    "toto_2_0": 0.6
   }
  },
  "weekly": {
   "shrink": 0.0,
   "weights": {
    "chronos_bolt": 0.0,
    "moirai_2_0": 0.0,
    "seasonal": 0.4,
    "timesfm_2_5": 0.6,
    "toto_2_0": 0.0
   }
  }
 }
}


def freq_key(freq):
    f = str(freq).lower()
    return "weekly" if ("w" in f or "week" in f) else ("monthly" if "month" in f or f in ("m", "1m", "ms") else "daily")


def seasonal(h, H, p):
    return [h[-p + (i % p)] if p and len(h) >= p else h[-1] for i in range(H)]


def forecast(view):
    mf = view["method_forecasts"]; H = view["H"]; h = view["history"]; toto = mf["toto_2_0"]
    fk = freq_key(view["freq"]); cfg = TABLE[view["dataset"]][fk]
    sea = seasonal(h, H, {"daily": 7, "weekly": 52, "monthly": 12}[fk]); last = h[-1]
    members = [sea if m == "seasonal" else mf.get(m, toto) for m in MEMBERS]
    w = [cfg["weights"][m] for m in MEMBERS]; s = cfg["shrink"]
    return [(1 - s) * sum(wi * mem[i] for wi, mem in zip(w, members)) + s * last for i in range(H)]
