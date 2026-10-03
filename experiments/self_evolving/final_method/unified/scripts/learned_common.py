"""Shared loading, batching, prediction, and scoring for learned baselines."""
import json, math, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[5]
TASKS = ROOT / "work/learned_baselines/tasks.jsonl"
OUT = ROOT / "runs/learned_baselines"
TEST = {"drcik": ("test",), "tmmd": ("test",), "timesx": ("test_id", "test_ood")}


def load(dataset, parts):
    if not TASKS.exists():
        raise SystemExit(f"run learned_data.py first: {TASKS}")
    rows = [json.loads(line) for line in open(TASKS)]
    return [row for row in rows if row["dataset"] == dataset and row["part"] in parts]


def seed(value=7):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def scale(task, context=512, horizon=256):
    history = np.asarray(task["history"][-context:], dtype=np.float32)
    truth = np.asarray(task["truth"][:horizon], dtype=np.float32)
    center = float(np.mean(history)); spread = float(np.std(history))
    if not math.isfinite(spread) or spread < 1e-5: spread = max(abs(center), 1.0)
    x = np.zeros(context, np.float32); x[-len(history):] = (history - center) / spread
    mask = np.zeros(context, np.float32); mask[-len(history):] = 1
    y = np.zeros(horizon, np.float32); y[:len(truth)] = (truth - center) / spread
    ym = np.zeros(horizon, np.float32); ym[:len(truth)] = 1
    return x, mask, y, ym, center, spread


class Tasks(Dataset):
    def __init__(self, rows, tokenizer=None, context=512, horizon=256):
        self.rows, self.tokenizer, self.context, self.horizon = rows, tokenizer, context, horizon

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x, mask, y, ym, center, spread = scale(row, self.context, self.horizon)
        out = dict(x=torch.tensor(x), mask=torch.tensor(mask), y=torch.tensor(y), ym=torch.tensor(ym),
                   center=torch.tensor(center), spread=torch.tensor(spread), tid=row["tid"])
        if self.tokenizer:
            text = self.tokenizer(row.get("text", "")[:6000], max_length=256, truncation=True,
                                  padding="max_length", return_tensors="pt")
            out["ids"], out["attention"] = text["input_ids"][0], text["attention_mask"][0]
        return out


def loss(pred, batch):
    return (torch.abs(pred - batch["y"]) * batch["ym"]).sum() / batch["ym"].sum().clamp_min(1)


def fit(model, train, dev, path, epochs=20, batch=16, lr=1e-3):
    from torch.utils.data import DataLoader
    seed(); path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    best = float("inf")
    for epoch in range(epochs):
        model.train(); total = 0.0
        for data in DataLoader(train, batch_size=batch, shuffle=True, num_workers=2):
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            value = loss(model(data), data); opt.zero_grad(); value.backward(); opt.step(); total += value.item()
        model.eval(); values = []
        with torch.no_grad():
            for data in DataLoader(dev, batch_size=batch, num_workers=2):
                data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
                pred = model(data) * data["spread"][:, None] + data["center"][:, None]
                truth = data["y"] * data["spread"][:, None] + data["center"][:, None]
                for p, y, mask in zip(pred, truth, data["ym"]):
                    n = int(mask.sum().item()); scale = y[:n].abs().mean().clamp_min(1e-8)
                    values.append(min(5.0, float((p[:n] - y[:n]).abs().mean() / scale)))
        value = float(np.mean(values)); print(epoch + 1, round(total, 4), "dev_smae", round(value, 4), flush=True)
        if value < best:
            best = value; torch.save(model.trainable_state(), path)
    model.load_trainable_state(torch.load(path, map_location="cpu"))


def predict(model, tasks, rows, output, batch=16):
    from torch.utils.data import DataLoader
    output.parent.mkdir(parents=True, exist_ok=True)
    forecasts = json.load(open(output)) if output.exists() else {}
    pending = [row for row in rows if row["tid"] not in forecasts]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.no_grad():
        for data in DataLoader(Tasks(pending, tasks.tokenizer, tasks.context, tasks.horizon), batch_size=batch):
            tids = data.pop("tid")
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            pred = model(data) * data["spread"][:, None] + data["center"][:, None]
            for tid, values in zip(tids, pred.cpu().tolist()):
                horizon = len(next(row["truth"] for row in pending if row["tid"] == tid))
                values = values[:horizon]
                if len(values) != horizon or not all(math.isfinite(x) for x in values): raise ValueError(tid)
                forecasts[tid] = values
            json.dump(forecasts, open(output, "w"))
    print(output, len(forecasts), "new", len(pending))


def score(name, dataset):
    import sys
    sys.path.insert(0, str(ROOT))
    from common.metrics import drcik_point_metrics
    rows, result = load(dataset, TEST[dataset]), []
    forecasts = json.load(open(OUT / name / dataset / "predictions.json"))
    for part in TEST[dataset]:
        group = [row for row in rows if row["part"] == part and row["tid"] in forecasts]
        metrics = [drcik_point_metrics(row["truth"], forecasts[row["tid"]], cap=5.0) for row in group]
        result.append(dict(dataset=dataset, split=part, model=name, tasks=len(group),
                           expected=len([row for row in rows if row["part"] == part]),
                           smae=round(float(np.mean([x["smae"] for x in metrics])), 3) if metrics else None,
                           srmse=round(float(np.mean([x["srmse"] for x in metrics])), 3) if metrics else None))
    json.dump(result, open(OUT / name / dataset / "metrics.json", "w"), indent=2)
    print(*result, sep="\n")
