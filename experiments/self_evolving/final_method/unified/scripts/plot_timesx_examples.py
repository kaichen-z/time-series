import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
RUN = ROOT / "runs/zero_shot_baselines"
sys.path.insert(0, str(ROOT))
from common.metrics import drcik_point_metrics


tasks = [task for task in json.load(open(RUN / "tasks.json")) if task["dataset"] == "timesx"]
models = {
    "Toto-2.0": json.load(open(RUN / "toto.json")),
    "TimesFM-2.5": json.load(open(RUN / "timesfm.json")),
    "Chronos-2": json.load(open(RUN / "chronos2.json")),
}


def joint(task, forecast):
    metrics = drcik_point_metrics(task["truth"], forecast, cap=5.0)
    return metrics["smae"] + metrics["srmse"]


for split in ("test_id", "test_ood"):
    group = [task for task in tasks if task["split"] == split]
    for name, forecasts in models.items():
        values = [drcik_point_metrics(task["truth"], forecasts[task["key"]], cap=5.0) for task in group]
        print(f"{split:8s} {name:12s} sMAE {np.mean([v['smae'] for v in values]):.3f} "
              f"sRMSE {np.mean([v['srmse'] for v in values]):.3f}")

selected = []
for split in ("test_id", "test_ood"):
    group = [task for task in tasks if task["split"] == split]
    ranked = sorted(group, key=lambda task: joint(task, models["Toto-2.0"][task["key"]])
                    - joint(task, models["TimesFM-2.5"][task["key"]]))
    selected.extend([(ranked[-1], "best"), (ranked[len(ranked) // 2], "median"), (ranked[0], "worst")])

figure, axes = plt.subplots(3, 2, figsize=(15, 11))
colors = {"Toto-2.0": "tab:blue", "TimesFM-2.5": "tab:orange", "Chronos-2": "tab:green"}
for axis, (task, rank) in zip(axes.flat, selected):
    history = task["history"][-48:]
    hx = np.arange(len(history))
    fx = np.arange(len(history), len(history) + task["H"])
    axis.plot(hx, history, color="black", linewidth=1.4, label="History")
    axis.plot(fx, task["truth"], color="tab:red", linewidth=2.4, label="Truth")
    errors = []
    for name, forecasts in models.items():
        forecast = forecasts[task["key"]]
        error = joint(task, forecast)
        errors.append(f"{name.split('-')[0]} {error:.2f}")
        axis.plot(fx, forecast, linestyle="--", color=colors[name], label=name)
    axis.axvline(len(history) - 0.5, color="gray", linewidth=1)
    axis.set_title(f"{task['split']} · {rank} change · {task['tid']}\n" + " | ".join(errors), fontsize=9)
    axis.grid(alpha=0.2)

handles, labels = axes.flat[0].get_legend_handles_labels()
figure.suptitle("TimesX zero-shot examples: joint error = sMAE + sRMSE", y=0.995)
figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4)
figure.tight_layout(rect=(0, 0, 1, 0.93))
output = RUN / "timesx_examples.png"
figure.savefig(output, dpi=180)
print(output)
