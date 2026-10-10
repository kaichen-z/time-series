#!/usr/bin/env python3
"""Train-only seed selection for protocol v6 (no model calls, no Dev/Test labels).

For each dataset and frequency, choose the numerical seed
    forecast = (1 - s) * sum_k w_k * member_k + s * last_value
over members {toto_2_0, timesfm_2_5, moirai_2_0, chronos_bolt, seasonal-naive} with weights on a simplex grid
(step --step) and shrink s in {0, 0.1, 0.2, 0.3}, minimising mean joint error (sMAE + sRMSE, each capped at 5) on
the official TRAIN tasks (pack F0) only, scored by forward-chaining (rolling-origin) CV: the Train timeline is
cut into K+1 blocks by forecast origin (TimesX: calendar date of the target window; Time-MMD: each series' own row
index, recovered from the official CSVs with the loader's task-id scheme, blocks formed per series by the origin's
relative position in that series' Train range).  Each configuration's score is the mean over blocks 1..K (equal
weights) of its mean joint error on that block; block 0 is never scored.  The blends have no fitted parameters, so
no per-fold refitting is needed.  Selected = lowest score; exact ties -> lower grid index.  For groups larger than
--max-per-group a fixed-seed (0) subsample is drawn per (frequency, horizon) before blocking; the sampled task ids
are written to the output.  Missing members fall back to Toto, exactly as in the seed module.
usage: select_seed_cv.py --pack PACK_DIR --dataset {timesx,time_mmd} --out OUT.json [--step 0.1] [--folds 4]
                         [--max-per-group 6000] [--repo TS_REPO --official-root ROOT  (Time-MMD time positions)]"""
import argparse, itertools, json, math, sys
from pathlib import Path

import numpy as np

P = argparse.ArgumentParser()
P.add_argument("--pack", type=Path, required=True); P.add_argument("--dataset", required=True)
P.add_argument("--out", type=Path, required=True); P.add_argument("--step", type=float, default=0.1)
P.add_argument("--folds", type=int, default=4); P.add_argument("--max-per-group", type=int, default=6000)
P.add_argument("--repo", type=Path); P.add_argument("--official-root", type=Path)
A = P.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import viewstore  # noqa: E402

MEMBERS = ("toto_2_0", "timesfm_2_5", "moirai_2_0", "chronos_bolt", "seasonal")
SHRINK = (0.0, 0.1, 0.2, 0.3)


def freq_key(freq):
    f = str(freq).lower()
    return "weekly" if ("w" in f or "week" in f) else ("monthly" if "month" in f or f in ("m", "1m", "ms") else "daily")


def period(fk): return {"daily": 7, "weekly": 52, "monthly": 12}[fk]


def seasonal(h, H, p):
    return [h[-p + (i % p)] if p and len(h) >= p else h[-1] for i in range(H)]


def grid(step):
    n = int(round(1 / step)); out = []
    for c in itertools.product(range(n + 1), repeat=len(MEMBERS) - 1):
        if sum(c) <= n: out.append(tuple(x / n for x in c) + ((n - sum(c)) / n,))
    return np.array(out, dtype=np.float64)


S = viewstore.Store(A.pack / "shared/store")


def time_positions(task_ids):
    """task -> (series_key, origin_pos, target_end_pos, rel) ; rel in [0,1] orders origins within the series' Train range."""
    pos = {}
    if A.dataset == "timesx":
        import datetime as dt
        d = lambda x: dt.date.fromisoformat(str(x)[:10]).toordinal()
        o = {t: d(S.meta[t]["future_timestamps"][0]) for t in task_ids}; lo, hi = min(o.values()), max(o.values())
        for t in task_ids:
            pos[t] = ("all", o[t], d(S.meta[t]["future_timestamps"][-1]), (o[t] - lo) / max(1, hi - lo))
        return pos
    import csv, hashlib
    man = json.load(open(A.repo / "handoff/official_ts_llm/alignment_manifest.json"))["time_mmd"]
    oid = lambda key: "official_" + hashlib.sha256(f"time_mmd\0{key}".encode()).hexdigest()[:24]
    want = set(task_ids)
    for rec in man["data_files"]:
        rp = Path(rec["path"]); rp = Path(*rp.parts[2:]) if rp.parts[:2] == ("repos", "MM-TSFlib") else rp
        with open(A.official_root / "repos/MM-TSFlib" / rp, newline="", encoding="utf-8-sig") as fh: n = sum(1 for _ in csv.DictReader(fh))
        n_train = int(n * 0.7); name = rp.name.lower()
        cad = "weekly" if "week" in name else "daily" if "day" in name else "monthly"
        lb = man["protocol"][cad]["lookback"]; domain = rp.parts[-2]
        for h in man["protocol"][cad]["horizons"]:
            first, last = lb, n_train - h
            for origin in range(first, last + 1):
                t = oid(f"time_mmd:{domain}:h{h}:train:t{origin}")
                if t in want: pos[t] = (domain, origin, origin + h - 1, (origin - first) / max(1, n_train - lb))
    missing = want - set(pos); assert not missing, f"{len(missing)} Train tasks without a time position"
    return pos
f0 = json.load(open(A.pack / "private/eval_F0_feedback.json"))
ids = f0["task_ids"]
# group by (frequency, horizon) for vectorisation; selection is per frequency
groups = {}
for t in ids:
    m = S.meta[t]; groups.setdefault((freq_key(m["freq"]), m["H"]), []).append(t)
W = grid(A.step); C = [(i, s) for i in range(len(W)) for s in SHRINK]
rng = np.random.default_rng(0)
result = dict(dataset=A.dataset, members=list(MEMBERS), shrink_grid=list(SHRINK), weight_step=A.step, folds=A.folds,
              objective="mean joint error (sMAE+sRMSE, each capped at 5), official Train only", groups={})


def errors_for(tasks):
    """per-task joint error for every config: array [n_tasks, n_configs]."""
    H = S.meta[tasks[0]]["H"]; n = len(tasks)
    M = np.zeros((n, len(MEMBERS), H), np.float32); Y = np.zeros((n, H), np.float32); L = np.zeros((n, 1), np.float32)
    for j, t in enumerate(tasks):
        v = S.view(t); mf = v["method_forecasts"]; toto = mf["toto_2_0"]; h = v["history"]
        for k, name in enumerate(MEMBERS):
            M[j, k] = seasonal(h, H, period(freq_key(v["freq"]))) if name == "seasonal" else mf.get(name, toto)
        Y[j] = f0_truth[t]; L[j, 0] = h[-1]
    scale = np.abs(Y).mean(1) + 1e-12
    E = np.zeros((n, len(C)))
    blends = np.einsum("ck,nkh->nch", W.astype(np.float32), M)  # [n, n_weights, H]
    for ci, (wi, s) in enumerate(C):
        F = (1 - s) * blends[:, wi, :] + s * L
        d = F - Y
        E[:, ci] = np.minimum(5, np.abs(d).mean(1) / scale) + np.minimum(5, np.sqrt((d * d).mean(1)) / scale)
    return E


f0_truth = f0["truth"]
POS = time_positions(ids)
by_freq = {}
for (fk, H), ts in sorted(groups.items()):
    if len(ts) > A.max_per_group:  # deterministic stratified subsample, keeps order (blocks stay contiguous)
        keep = np.sort(rng.choice(len(ts), A.max_per_group, replace=False)); ts = [ts[i] for i in keep]
    by_freq.setdefault(fk, []).append((H, ts, errors_for(ts)))
    print(f"{fk} H={H}: {len(ts)} tasks scored", flush=True)

ci_toto = C.index((int(np.where((W == np.eye(len(MEMBERS))[0]).all(1))[0][0]), 0.0))
ci_tfm = C.index((int(np.where((W == np.eye(len(MEMBERS))[1]).all(1))[0][0]), 0.0))
for fk, parts in by_freq.items():
    E = np.vstack([e for _, _, e in parts]); n = len(E)
    tasks = [t for _, ts, _ in parts for t in ts]
    # forward-chaining blocks by relative origin position (K+1 blocks; block 0 is train-only)
    rel = np.array([POS[t][3] for t in tasks]); blk = np.minimum((rel * (A.folds + 1)).astype(int), A.folds)
    ser = [POS[t][0] for t in tasks]; org = np.array([POS[t][1] for t in tasks]); end = np.array([POS[t][2] for t in tasks])
    folds_te = []
    for k in range(1, A.folds + 1):
        te = blk == k
        if te.any(): folds_te.append(te)
    # PRE-REGISTERED RULE: each configuration is scored by its forward-chained out-of-sample error = mean over folds
    # (equal fold weights) of the mean joint error on that fold's test block; the configuration with the LOWEST score
    # is selected; exact ties -> lower configuration index (fixed grid order).  The blends have no fitted parameters,
    # so no per-fold refitting is involved; block 0 (earliest) is never scored.
    fold_mean = np.vstack([E[te].mean(0) for te in folds_te])            # [n_folds, n_configs]
    score = fold_mean.mean(0); best = int(np.argmin(score)); wi, s = C[best]
    cv = {"selected": fold_mean[:, best].tolist(), "toto": fold_mean[:, ci_toto].tolist(), "timesfm": fold_mean[:, ci_tfm].tolist(),
          "n_test": [int(te.sum()) for te in folds_te]}
    result["groups"][fk] = dict(
        n_tasks=int(n), weights={m: float(w) for m, w in zip(MEMBERS, W[wi])}, shrink=float(s),
        train_mean_joint_error=dict(selected=float(E[:, best].mean()), toto=float(E[:, ci_toto].mean()),
                                    timesfm=float(E[:, ci_tfm].mean())),
        sampled_task_ids=tasks,
        cv_mean_joint_error={k: float(np.mean(cv[k])) for k in ("selected", "toto", "timesfm")}, cv_folds=cv,
        selection_rule="argmin over grid of mean-over-folds out-of-sample mean joint error; ties -> lower grid index",
        best_single_member={m: float(score[C.index((int(np.where((W == np.eye(len(MEMBERS))[j]).all(1))[0][0]), 0.0))])
                            for j, m in enumerate(MEMBERS)})
    g = result["groups"][fk]
    print(f"{fk}: weights {g['weights']} shrink {s} | CV selected {g['cv_mean_joint_error']['selected']:.4f} "
          f"toto {g['cv_mean_joint_error']['toto']:.4f} timesfm {g['cv_mean_joint_error']['timesfm']:.4f}", flush=True)
A.out.write_text(json.dumps(result, indent=1)); print("wrote", A.out)
