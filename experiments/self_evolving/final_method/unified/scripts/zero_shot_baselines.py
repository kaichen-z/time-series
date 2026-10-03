import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "runs/zero_shot_baselines"
TASKS = OUT / "tasks.json"
MODELS = ("toto", "timesfm", "chronos2", "moirai")
TMMD = {
    "Agriculture": (6, "monthly"),
    "Climate": (12, "weekly"),
    "Economy": (6, "monthly"),
    "Energy": (12, "weekly"),
    "Environment": (48, "daily"),
    "Health_AFR": (12, "weekly"),
    "Health_US": (12, "weekly"),
    "Security": (6, "monthly"),
    "SocialGood": (6, "monthly"),
    "Traffic": (6, "monthly"),
}


def clean(values):
    out = []
    last = 0.0
    for value in values:
        if value is not None and math.isfinite(value):
            last = float(value)
        out.append(last)
    return out


def build_tmmd():
    manifest = json.load(open(ROOT / "splits/timemmd_531_181_182_v1.json"))
    out = []
    for domain, (horizon, frequency) in TMMD.items():
        path = ROOT / "external/Time-MMD/numerical" / domain / f"{domain}.csv"
        rows = list(csv.DictReader(open(path)))
        date = "date" if "date" in rows[0] else "date.1"
        series = sorted((row[date], float(row["OT"]) if row["OT"] else math.nan) for row in rows)
        dates = [item[0] for item in series]
        values = [item[1] for item in series]
        for split in ("dev", "test"):
            for tid in manifest[split]:
                prefix = f"tmmd_{domain}_"
                if not tid.startswith(prefix):
                    continue
                start = dates.index(tid[len(prefix):])
                truth = values[start:start + horizon]
                if len(truth) != horizon or not all(math.isfinite(value) for value in truth):
                    raise ValueError(f"invalid Time-MMD horizon: {tid}")
                out.append(dict(key="tmmd/" + tid, tid=tid, dataset="tmmd", split=split,
                                history=clean(values[:start]), truth=truth, H=horizon, freq=frequency))
    return out


def build():
    tasks = []
    drcik = ROOT / ".scratch/self_evolving/nrd_cache_full2.json"
    if drcik.exists():
        for row in json.load(open(drcik)):
            if row["part"] in ("dev", "public_test"):
                tasks.append(dict(key="drcik/" + row["tid"], tid=row["tid"], dataset="drcik",
                                  split={"dev": "dev", "public_test": "test"}[row["part"]],
                                  history=clean(row["history"]), truth=row["truth"], H=row["H"], freq=row["freq"]))

    tasks.extend(build_tmmd())

    timesx = ROOT / "work/timesx/tasks.json"
    if timesx.exists():
        for row in json.load(open(timesx)):
            if row["part"] in ("test_id", "test_ood"):
                tasks.append(dict(key="timesx/" + row["tid"], tid=row["tid"], dataset="timesx",
                                  split={"test_id": "test_id", "test_ood": "test_ood"}[row["part"]],
                                  history=clean(row["history"]), truth=row["truth"], H=len(row["truth"]), freq=row["freq"]))

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(tasks, open(TASKS, "w"))
    counts = defaultdict(int)
    for task in tasks:
        counts[(task["dataset"], task["split"])] += 1
    print("tasks", {f"{dataset}/{split}": count for (dataset, split), count in sorted(counts.items())}, flush=True)


def toto(tasks):
    import torch
    from toto2 import Toto2Model
    model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to("cuda").eval()
    out = {}
    for i, task in enumerate(tasks):
        pad = (-len(task["history"])) % 32
        target = torch.tensor([0.0] * pad + task["history"], dtype=torch.float32, device="cuda").reshape(1, 1, -1)
        mask = torch.ones_like(target, dtype=torch.bool)
        if pad:
            mask[..., :pad] = False
        with torch.no_grad():
            quantiles = model.forecast({"target": target, "target_mask": mask,
                                        "series_ids": torch.zeros((1, 1), dtype=torch.long, device="cuda")},
                                       horizon=task["H"], decode_block_size=None, has_missing_values=False)
        quantiles = [q.reshape(-1).tolist() if hasattr(q, "reshape") else q
                     for q in (quantiles if isinstance(quantiles, (list, tuple)) else list(quantiles))]
        out[task["key"]] = quantiles[4]
        if i % 200 == 0:
            print(i, len(tasks), flush=True)
    return out


def timesfm(tasks):
    import timesfm as tfm
    model = tfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(tfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True,
                                     use_continuous_quantile_head=True, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))
    groups = defaultdict(list)
    for task in tasks:
        if task["H"] <= 256:
            groups[task["H"]].append(task)
    out = {}
    for horizon, group in groups.items():
        for i in range(0, len(group), 64):
            batch = group[i:i + 64]
            forecasts, _ = model.forecast(horizon=horizon, inputs=[task["history"][-1024:] for task in batch])
            for task, forecast in zip(batch, np.asarray(forecasts, float)):
                out[task["key"]] = forecast[:horizon].tolist()
    return out


def chronos2(tasks):
    from chronos import Chronos2Pipeline
    model = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    out = {}
    groups = defaultdict(list)
    for task in tasks:
        groups[task["H"]].append(task)
    for horizon, group in groups.items():
        for i in range(0, len(group), 32):
            batch = group[i:i + 32]
            quantiles, _ = model.predict_quantiles([np.asarray(task["history"], dtype=np.float32) for task in batch],
                                                   prediction_length=horizon, quantile_levels=[0.5])
            for task, forecast in zip(batch, quantiles):
                out[task["key"]] = np.asarray(forecast).reshape(-1)[:horizon].tolist()
    return out


def moirai(tasks):
    from numerical_agent.main import _runtime_registry
    from numerical_agent.evolution.portfolio import forecast_tsfm, read_policy_file
    args = argparse.Namespace(tsfm_runtimes="", model_cache_dir=None,
                              tsfm_workers_config=str(ROOT / ".scratch/self_evolving/tsfm_workers.json"),
                              acknowledged_model_licenses="CC-BY-NC-4.0")
    runtimes = _runtime_registry(args)
    portfolio = read_policy_file(ROOT / "runs/method_evolution/v001/policies.py")
    policy = {p.name: p for p in portfolio.tsfm}["moirai_2_0"]
    out = {}
    for i, task in enumerate(tasks):
        try:
            out[task["key"]] = list(forecast_tsfm(policy, history=tuple(task["history"]), horizon=task["H"],
                                                   frequency=task["freq"], runtimes=runtimes))
        except Exception as error:
            print(task["key"], type(error).__name__, str(error)[:200], flush=True)
        if i % 200 == 0:
            print(i, len(tasks), flush=True)
    return out


def worker(name):
    tasks = json.load(open(TASKS))
    output = OUT / f"{name}.json"
    forecasts = json.load(open(output)) if output.exists() else {}
    pending = [task for task in tasks if task["key"] not in forecasts]
    forecasts.update(globals()[name](pending))
    json.dump(forecasts, open(OUT / f"{name}.json", "w"))
    print(name, len(forecasts), "new", len(pending), flush=True)


def score(models=MODELS):
    sys.path.insert(0, str(ROOT))
    from common.metrics import drcik_point_metrics
    tasks = json.load(open(TASKS))
    forecasts = {name: json.load(open(OUT / f"{name}.json")) for name in models}
    rows = []
    for dataset, split in sorted({(task["dataset"], task["split"]) for task in tasks}):
        group = [task for task in tasks if task["dataset"] == dataset and task["split"] == split]
        for name in models:
            available = [task for task in group if task["key"] in forecasts[name]]
            metrics = [drcik_point_metrics(task["truth"], forecasts[name][task["key"]], cap=5.0) for task in available]
            smae = float(np.mean([metric["smae"] for metric in metrics])) if metrics else None
            srmse = float(np.mean([metric["srmse"] for metric in metrics])) if metrics else None
            rows.append(dict(dataset=dataset, split=split, model=name, tasks=len(available), expected=len(group),
                             smae=smae, srmse=srmse))
            values = "–       –" if smae is None else f"{smae:.3f}   {srmse:.3f}"
            print(f"{dataset:7s} {split:8s} {name:8s} {len(available):4d}/{len(group):4d}  {values}")
    json.dump(rows, open(OUT / "metrics.json", "w"), indent=2)


def run(gpus, models=MODELS):
    build()
    tasks = json.load(open(TASKS))
    if not tasks:
        raise SystemExit("no evaluation tasks found")
    keys = {task["key"] for task in tasks}
    interpreters = {
        "toto": ROOT / ".venv/toto2/bin/python",
        "timesfm": ROOT / ".venv/timesfm/bin/python",
        "chronos2": ROOT / ".venv/timesfm/bin/python",
        "moirai": Path(sys.executable),
    }
    jobs = []
    for name, gpu in zip(models, gpus):
        output = OUT / f"{name}.json"
        if output.exists() and keys <= set(json.load(open(output))):
            print("reuse", output)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT), MPLCONFIGDIR="/tmp/matplotlib-zero-shot")
        log = open(OUT / f"{name}.log", "w")
        process = subprocess.Popen([str(interpreters[name]), str(Path(__file__).resolve()), "worker", name],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((name, process, log))
    failed = []
    for name, process, log in jobs:
        status = process.wait()
        log.close()
        if status:
            failed.append(name)
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))
    score(models)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        score(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else MODELS)
    else:
        gpu_ids = (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")
        models = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else MODELS
        if len(gpu_ids) != len(models) or any(model not in MODELS for model in models):
            raise SystemExit("usage: zero_shot_baselines.py <gpu,...> [toto,timesfm,chronos2,moirai]")
        run(gpu_ids, models)
