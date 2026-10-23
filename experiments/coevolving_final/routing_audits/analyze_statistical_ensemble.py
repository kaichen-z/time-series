#!/usr/bin/env python3
"""Train-only grouped-CV audit of statistical ensembles and routing upper bounds.

The script never opens Dev/Test. Folds are assigned by a stable hash of each
row's document bundle so windows sharing the same task/entity evidence stay in
one fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from evaluate_controller import pack_rows
from evaluate_v6_incumbent import incumbent_modules, metrics, run_pipeline


MODELS = (
    "toto_2_0", "timesfm_2_5", "moirai_2_0", "chronos_bolt",
    "granite_ttm_r2", "combined_chronos_damped_trend",
    "combined_granite_regime_profile", "combined_moirai_croston_router",
    "combined_timesfm_seasonal", "combined_toto_robust_router",
)


def stable_blob(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entity_key(view) -> str:
    # Documents are invariant within the same benchmark task/entity while
    # histories change across rolling windows.
    return hashlib.sha256(stable_blob(view.get("documents", [])).encode()).hexdigest()


def fold_of(view) -> int:
    return int(entity_key(view)[:8], 16) % 3


def stratum(view) -> str:
    return f'{view.get("freq", "unknown")}|H={int(view["H"])}'


def safe_forecasts(view):
    out = []
    fallback = np.asarray(view["method_forecasts"]["toto_2_0"], dtype=float)
    for name in MODELS:
        value = np.asarray(view["method_forecasts"].get(name, fallback), dtype=float)
        if value.shape != fallback.shape or not np.isfinite(value).all():
            value = fallback
        out.append(value)
    return np.stack(out, axis=1)  # H x M


def simplex_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = x.shape[1]
    objective = lambda w: float(np.mean((x @ w - y) ** 2))
    result = minimize(objective, np.full(m, 1.0 / m), method="SLSQP",
                      bounds=[(0.0, 1.0)] * m,
                      constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
                      options={"maxiter": 300, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        return np.full(m, 1.0 / m)
    w = np.maximum(result.x, 0.0)
    return w / w.sum()


def summarize(predictions, truths):
    smaes, srmses = [], []
    for p, y in zip(predictions, truths):
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        scale = float(np.mean(np.abs(y))) + 1e-12
        diff = p - y
        smaes.append(min(5.0, float(np.mean(np.abs(diff))) / scale))
        srmses.append(min(5.0, float(np.sqrt(np.mean(diff * diff))) / scale))
    smae = float(np.mean(smaes))
    srmse = float(np.mean(srmses))
    return {"sMAE": smae, "sRMSE": srmse, "joint": smae + srmse}


def evaluate(dataset: str):
    incumbent = incumbent_modules(dataset)
    rows = []
    for view, truth in pack_rows(dataset):
        truth = np.asarray(truth, dtype=float)
        history = np.asarray(view["history"], dtype=float)
        history_scale = float(np.mean(np.abs(history))) + 1e-12
        rows.append({
            "view": view, "truth": truth, "x": safe_forecasts(view),
            "inc": np.asarray(run_pipeline(incumbent, view), dtype=float),
            "scale": history_scale, "fold": fold_of(view),
            "entity": entity_key(view), "stratum": stratum(view),
        })

    unique_entities = len({r["entity"] for r in rows})
    fold_entities = {str(k): len({r["entity"] for r in rows if r["fold"] == k}) for k in range(3)}
    methods = defaultdict(list)
    truths = []
    fold_receipts = []

    for heldout in range(3):
        train = [r for r in rows if r["fold"] != heldout]
        valid = [r for r in rows if r["fold"] == heldout]
        if not train or not valid:
            raise RuntimeError(f"empty fold {heldout}")

        train_x = np.concatenate([r["x"] / r["scale"] for r in train], axis=0)
        train_y = np.concatenate([r["truth"] / r["scale"] for r in train], axis=0)
        simplex = simplex_fit(train_x, train_y)

        # Unconstrained ridge is a useful statistical comparator. Alpha is
        # selected only on the training portion through deterministic
        # entity-hash subfolds.
        best_alpha, best_loss = None, float("inf")
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            losses = []
            for inner in range(3):
                a = [r for r in train if int(r["entity"][8:16], 16) % 3 != inner]
                b = [r for r in train if int(r["entity"][8:16], 16) % 3 == inner]
                if not a or not b:
                    continue
                ax = np.concatenate([r["x"] / r["scale"] for r in a])
                ay = np.concatenate([r["truth"] / r["scale"] for r in a])
                bx = np.concatenate([r["x"] / r["scale"] for r in b])
                by = np.concatenate([r["truth"] / r["scale"] for r in b])
                model = Ridge(alpha=alpha, fit_intercept=True).fit(ax, ay)
                losses.append(float(np.mean((model.predict(bx) - by) ** 2)))
            loss = statistics.mean(losses) if losses else float("inf")
            if loss < best_loss:
                best_alpha, best_loss = alpha, loss
        ridge = Ridge(alpha=best_alpha, fit_intercept=True).fit(train_x, train_y)

        # Exponential weighting uses per-row standardized joint losses.
        model_losses = []
        for j in range(len(MODELS)):
            losses = []
            for r in train:
                scale = float(np.mean(np.abs(r["truth"]))) + 1e-12
                diff = r["x"][:, j] - r["truth"]
                losses.append(min(5.0, float(np.mean(np.abs(diff))) / scale) +
                              min(5.0, float(np.sqrt(np.mean(diff * diff))) / scale))
            model_losses.append(float(np.mean(losses)))
        loss_arr = np.asarray(model_losses)
        temp = max(float(np.std(loss_arr)), 1e-6)
        expw = np.exp(-(loss_arr - loss_arr.min()) / temp)
        expw /= expw.sum()

        # Bias calibration of incumbent, by observable frequency/horizon and
        # relative horizon bucket. Back off to a global bucket when sparse.
        residuals = defaultdict(list)
        for r in train:
            h = len(r["truth"])
            for t, (yy, pp) in enumerate(zip(r["truth"], r["inc"])):
                bucket = min(3, (4 * t) // h)
                residuals[(r["stratum"], bucket)].append((yy - pp) / r["scale"])
                residuals[("*", bucket)].append((yy - pp) / r["scale"])
        bias = {k: float(np.median(v)) for k, v in residuals.items()}

        fold_receipts.append({
            "fold": heldout, "n_train": len(train), "n_valid": len(valid),
            "ridge_alpha": best_alpha,
            "simplex_weights": dict(zip(MODELS, map(float, simplex))),
            "exp_weights": dict(zip(MODELS, map(float, expw))),
        })

        for r in valid:
            truths.append(r["truth"])
            methods["v6_incumbent"].append(r["inc"])
            for j, name in enumerate(MODELS):
                methods[f"raw::{name}"].append(r["x"][:, j])
            methods["cv_simplex"].append(r["x"] @ simplex)
            pred_ridge_norm = ridge.predict(r["x"] / r["scale"])
            methods["cv_ridge"].append(pred_ridge_norm * r["scale"])
            methods["cv_exponential"].append(r["x"] @ expw)
            h = len(r["truth"])
            corrected = r["inc"].copy()
            for t in range(h):
                bucket = min(3, (4 * t) // h)
                delta = bias.get((r["stratum"], bucket), bias.get(("*", bucket), 0.0))
                corrected[t] += delta * r["scale"]
            methods["cv_v6_bias_calibration"].append(corrected)

    scores = {name: summarize(preds, truths) for name, preds in methods.items()}
    base = scores["v6_incumbent"]["joint"]
    for score in scores.values():
        score["gain_vs_v6"] = base - score["joint"]
        score["standardized_gain_vs_v6"] = (base - score["joint"]) / base

    # Two deliberately optimistic ceilings, clearly kept separate from CV.
    per_row_oracle = []
    for r in rows:
        candidates = [r["inc"]] + [r["x"][:, j] for j in range(len(MODELS))]
        scale = float(np.mean(np.abs(r["truth"]))) + 1e-12
        def row_joint(p):
            diff = np.asarray(p) - r["truth"]
            return min(5.0, float(np.mean(np.abs(diff))) / scale) + min(
                5.0, float(np.sqrt(np.mean(diff * diff))) / scale)
        per_row_oracle.append(min(candidates, key=row_joint))
    oracle_score = summarize(per_row_oracle, [r["truth"] for r in rows])
    oracle_score["gain_vs_v6"] = base - oracle_score["joint"]
    oracle_score["standardized_gain_vs_v6"] = (base - oracle_score["joint"]) / base

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["joint"])
    return {
        "dataset": dataset, "protocol": "Train-only 3-fold document/entity-hash grouped CV",
        "n_rows": len(rows), "n_entities": unique_entities, "fold_entities": fold_entities,
        "models": list(MODELS), "fold_receipts": fold_receipts,
        "ranked_scores": [{"method": k, **v} for k, v in ranked],
        "per_row_oracle_upper_bound": oracle_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("time_mmd", "timesx", "all"), default="all")
    parser.add_argument("--out", type=Path, default=Path("statistical_ensemble_audit.json"))
    args = parser.parse_args()
    datasets = ("time_mmd", "timesx") if args.dataset == "all" else (args.dataset,)
    result = {d: evaluate(d) for d in datasets}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
