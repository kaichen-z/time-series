#!/usr/bin/env python3
"""Three-role synchronous Codex evolution of one shared controller."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from evaluate_controller import evaluate


HERE = Path(__file__).resolve().parent
ROLES = {
    "routing": "Improve robust TSFM weighting and routing from dataset, frequency, and history regime.",
    "evidence": "Improve use of document events and reliability evidence with conservative bounded actions.",
    "safety": "Improve shrinkage, gating, and worst-group safety while preserving raw/strength-zero fallbacks.",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(base: Path, score: dict, root: Path, round_no: int, role: str) -> Path:
    work = root / f"round_{round_no:02d}" / role
    work.mkdir(parents=True, exist_ok=False)
    candidate = work / "candidate.py"
    shutil.copy(base, candidate)
    feedback = {k: {x: v[x] for x in ("n", "seed_joint", "candidate_joint", "standardized_gain", "worst")}
                for k, v in score.items() if isinstance(v, dict)}
    prompt = f"""You are the {role} role improving one shared, blinded ForecastController.

Edit ONLY candidate.py. It must keep a pure function forecast(view)->list[float] of exact horizon length, using only standard Python and the supplied view. Do not read any files, do not run searches, and do not use task IDs, timestamps, lookup tables, hashes, encoded constants, or external state. General rules only.

The same controller serves datasets drcik, time_mmd, and timesx. Every view has dataset, freq, H, history, method_forecasts, and documents with structured events. Dr-CiK additionally exposes reliability_available, reliability_fraction, and a neutral extra forecast key timesfm_2_5_reliability. Legal bounded operator families are TSFM mixing, none/raw, reliability selection or blending, document-direction level/tilt, shrink-to-last/seasonal, and conservative volatility calibration. Raw behavior and strength zero must remain possible. The host objective gives each dataset equal weight and rejects material regression on any dataset.

Your focus: {ROLES[role]}
Current host-only aggregate feedback (lower candidate_joint is better; you cannot access labels or evaluator):
{json.dumps(feedback, indent=2)}

Make one principled revision likely to generalize. Keep code under 20 KB and under 300 numeric literals. Do not merely narrate: directly edit candidate.py, then stop.
"""
    with open(work / "codex.log", "w") as log:
        result = subprocess.run([
            "codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
            "--ephemeral", "-C", str(work), "-o", str(work / "final.md"), "-"
        ], input=prompt, text=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode != 0:
        (work / "FAILED").write_text(str(result.returncode))
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=Path, default=HERE / "seed_controller.py")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    champion = args.out / "champion.py"
    shutil.copy(args.start, champion)
    score = evaluate(champion)
    ledger = [{"round": 0, "role": "seed", "sha256": sha(champion), "score": score}]
    (args.out / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for round_no in range(1, args.rounds + 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {role: pool.submit(run_one, champion, score, args.out, round_no, role) for role in ROLES}
            proposals = {role: future.result() for role, future in futures.items()}
        results = []
        for role, path in proposals.items():
            try:
                result = evaluate(path)
                feasible = bool(result["feasible"])
            except Exception as exc:
                result, feasible = {"error": repr(exc)}, False
            results.append((role, path, result, feasible))
            ledger.append({"round": round_no, "role": role, "sha256": sha(path), "score": result, "feasible": feasible})
        eligible = [x for x in results if x[3] and x[2]["objective"] > score["objective"] + 1e-6]
        if eligible:
            role, path, score, _ = max(eligible, key=lambda x: (x[2]["objective"], x[0]))
            shutil.copy(path, champion)
            ledger.append({"round": round_no, "accepted": role, "sha256": sha(champion), "score": score})
        else:
            ledger.append({"round": round_no, "accepted": None, "sha256": sha(champion), "score": score})
        (args.out / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
        print(json.dumps({"round": round_no, "objective": score["objective"], "accepted": ledger[-1]["accepted"]}), flush=True)


if __name__ == "__main__":
    main()
