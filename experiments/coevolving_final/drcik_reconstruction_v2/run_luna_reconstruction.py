from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import threading
import time
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TL2_DIR = REPO / "experiments/self_evolving/final_method/scripts"
TASKS = Path(os.environ.get("TS_DRCIK_TASKS", "data/Dr-CiK_public/tasks"))
SPLIT = REPO / "splits/drcik_public_80_20_99_v3.json"
INSTR = Path(__file__).resolve().parent / "configs/instr_evolved.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tl2():
    sys.path.insert(0, str(TL2_DIR))
    spec = importlib.util.spec_from_file_location("reconstruction_tl2", TL2_DIR / "tl2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.TASKS = TASKS
    return module


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "luna_timelines_v2.json")
    args = ap.parse_args()

    tl2 = load_tl2()
    instruction = json.loads(INSTR.read_text())["instr"]
    split = json.loads(SPLIT.read_text())["partitions"]
    ordered = []
    part_of = {}
    for part in ("train", "dev", "public_test"):
        for tid in split[part]["task_ids"]:
            ordered.append(tid)
            part_of[tid] = part
    assert len(ordered) == 199 and len(set(ordered)) == 199

    cache = json.loads(args.out.read_text()) if args.out.exists() else {}
    errors_path = args.out.with_name("luna_errors_v2.json")
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else {}
    lock = threading.Lock()

    def run(tid: str):
        last = None
        for attempt in range(1, args.attempts + 1):
            started = time.time()
            try:
                result = tl2.codex(tl2.prompt(instruction, tid), tl2.SCHEMA, args.model, 1500)
                return tid, result["intervals"], {
                    "attempt": attempt,
                    "seconds": round(time.time() - started, 3),
                }
            except Exception as exc:  # retain exact failure class/message, then retry
                last = {"attempt": attempt, "type": type(exc).__name__, "message": str(exc)[-1000:]}
        raise RuntimeError(json.dumps({"task": tid, "last_error": last}))

    todo = [tid for tid in ordered if tid not in cache]
    started_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, tid): tid for tid in todo}
        for fut in concurrent.futures.as_completed(futures):
            tid = futures[fut]
            try:
                task, intervals, meta = fut.result()
                with lock:
                    cache[task] = intervals
                    errors.pop(task, None)
                    atomic_json(args.out, cache)
                    atomic_json(errors_path, errors)
                    print(json.dumps({"task": task, "part": part_of[task], "intervals": len(intervals), **meta}), flush=True)
            except Exception as exc:
                with lock:
                    errors[tid] = {"type": type(exc).__name__, "message": str(exc)[-2000:]}
                    atomic_json(errors_path, errors)
                    print(json.dumps({"task": tid, "failed": True, "error": str(exc)[-500:]}), flush=True)

    missing = [tid for tid in ordered if tid not in cache]
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} tasks missing; see {errors_path}")
    canonical = {tid: cache[tid] for tid in ordered}
    atomic_json(args.out, canonical)
    receipt = {
        "schema": "drcik-luna-timelines-reconstruction-v2",
        "status": "new reconstruction, not original SHA 7315f12e cache",
        "model": args.model,
        "workers": args.workers,
        "attempts": args.attempts,
        "task_counts": {part: len(split[part]["task_ids"]) for part in ("train", "dev", "public_test")},
        "empty_counts": {
            part: sum(not canonical[tid] for tid in split[part]["task_ids"])
            for part in ("train", "dev", "public_test")
        },
        "instruction_sha256": sha(INSTR),
        "split_sha256": sha(SPLIT),
        "tasks_root": str(TASKS),
        "official_dataset_revision": "00fbe820ff7a221e4aca71883219ef27f8223050",
        "task_set_content_sha256": "8efea4bfc2e552248faefd8965ac24ad749f4332a0714051b2620aa7692af58f",
        "output_sha256": sha(args.out),
        "codex_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
        "python": platform.python_version(),
        "elapsed_seconds": round(time.time() - started_all, 3),
    }
    atomic_json(args.out.with_name("luna_reconstruction_receipt_v2.json"), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
