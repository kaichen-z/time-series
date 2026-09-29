"""Regenerate the document-derived future-correction cards (`cordp_cards_{part}.json`) with any LLM
available through the `codex` CLI (default: gpt-6-astra). Same prompt, inputs and output format as the
original cards (scripts/cordp_train.py, which used Claude haiku); only the model changes.
No labels are used: the LLM sees the task documents, a history summary and the Toto base forecast.

Run from the repo root (needs: Dr-CiK tasks under external/, the forecast cache from
https://huggingface.co/datasets/yyoraa/drcik-forecast-cache unpacked under runs/champion_forecasts/,
inputs/method_evolution_v001 copied to runs/method_evolution/v001, and a logged-in `codex` CLI):

    PYTHONPATH=$PWD python experiments/self_evolving/final_method/scripts/regenerate_cards.py \
        --model gpt-6-astra --parts train,dev,public_test --workers 12 --out-prefix .scratch/cordp_cards_astra

Output: <out-prefix>_{train,dev,public_test}.json  ({task_id: {"confidence": c, "corrections": [[start_ts, end_ts, m], ...]}}).
Resumable: finished tasks are cached in <out-prefix>_cache/ and skipped on rerun.
"""
from __future__ import annotations
import argparse, hashlib, json, re, statistics, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import _parse
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore

ROOT = Path(".").resolve()
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SYSTEM = (
    "You are a forecasting-correction module. A strong statistical model has ALREADY produced "
    "a base forecast (median values with timestamps). Your ONLY job: decide whether the "
    "documented real-world events imply the TRUE values in specific sub-windows will differ "
    "from this base, and if so by what MULTIPLICATIVE factor.\n"
    "Rules:\n"
    "1. Anchor to the shown base scale. A correction multiplier is typically within [0.3, 3.0]; "
    "never produce values orders of magnitude off the base.\n"
    "2. Correct ONLY sub-windows where a document explicitly implies a change (holiday closure, "
    "outage, strike, promotion, weather event...). Give the timestamps and a one-line rationale "
    "quoting the document. Do NOT rescale the whole horizon on vague 'stabilized/steady' wording "
    "-- the base already has the level.\n"
    "3. If the documents do NOT clearly imply any deviation from the base, return "
    "\"relevant\": false with an empty corrections list. Do not guess.\n"
    "4. multiplier < 1 => lower than base, > 1 => higher.\n"
    'Output STRICT JSON: {"relevant": bool, "corrections": [{"start_timestamp": iso, '
    '"end_timestamp": iso, "multiplier": number, "rationale": str}], "confidence": number}.')
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["relevant", "corrections", "confidence"],
          "properties": {"relevant": {"type": "boolean"}, "confidence": {"type": "number"},
                         "corrections": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                             "required": ["start_timestamp", "end_timestamp", "multiplier", "rationale"],
                             "properties": {"start_timestamp": {"type": "string"}, "end_timestamp": {"type": "string"},
                                            "multiplier": {"type": "number"}, "rationale": {"type": "string"}}}}}}


def wd(ts):
    p = _parse(ts); return p.weekday() if p else -1


def build_user(task, base):
    n = task.numeric
    hv, hts = list(n.history_values), [str(x) for x in task.history_timestamps]
    fts = [str(x) for x in task.future_timestamps]
    we = [x for x, t in zip(hv, hts) if wd(t) >= 5]; wk = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]
    hist = {"typical_scale": round(statistics.median([abs(x) for x in hv]) or 1.0, 4),
            "overall_mean": round(statistics.mean(hv), 4),
            "weekday_mean": round(statistics.mean(wk), 4) if wk else None,
            "weekend_mean": round(statistics.mean(we), 4) if we else None,
            "last_values": [round(x, 4) for x in hv[-12:]]}
    docs = [{"document_id": d.document_id, "content": d.content[:3500]} for d in task.documents]
    return json.dumps({"target_name": task.target_name, "target_description": task.target_description,
                       "frequency": n.frequency, "history_summary": hist, "documents": docs,
                       "base_forecast_median": [[fts[i], round(base[i], 4)] for i in range(len(fts))]},
                      ensure_ascii=False)


def call(model, prompt, timeout):
    with tempfile.TemporaryDirectory() as td:
        sp, op = Path(td) / "s.json", Path(td) / "o.json"; sp.write_text(json.dumps(SCHEMA))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                        "-m", model, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=prompt, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout)
        return json.loads(op.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-astra"); ap.add_argument("--parts", default="train,dev,public_test")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-prefix", default=".scratch/cordp_cards_astra"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    split = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
    mr = ROOT / "runs/method_evolution/v001"
    fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906", mr / "methods.py",
                       mr / "skills.py", read_policy_file(str(mr / "policies.py")), None,
                       screening_hash=_load_screening_policy(str(mr / "dictionary.py")).fingerprint(),
                       runtime_identity={}, cache_only=True, identity_hash_override=IDH)
    cache = Path(a.out_prefix + "_cache"); cache.mkdir(parents=True, exist_ok=True)
    for part in a.parts.split(","):
        ids = tuple(split[part]["task_ids"])[: a.limit]
        tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS_DIR, ids)}

        def one(tid):
            t = tasks[tid]; n = t.numeric
            base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
            prompt = SYSTEM + "\n\nINPUT:\n" + build_user(t, base)
            key = hashlib.sha256((a.model + prompt).encode()).hexdigest()[:20]; cf = cache / f"{key}.json"
            if cf.exists(): return tid, json.loads(cf.read_text())
            for attempt in range(3):
                try:
                    out = call(a.model, prompt, a.timeout); cf.write_text(json.dumps(out)); return tid, out
                except Exception as e:
                    err = repr(e)
            print(f"{part} {tid}: failed ({err[:120]})", flush=True); return tid, None

        cards = {}
        with ThreadPoolExecutor(a.workers) as ex:
            for i, (tid, out) in enumerate(ex.map(one, ids), 1):
                if out is None: continue
                corrs = []
                for c in out.get("corrections") or []:
                    try: corrs.append([c["start_timestamp"], c["end_timestamp"], float(c["multiplier"])])
                    except (KeyError, TypeError, ValueError): pass
                try: conf = float(out.get("confidence"))
                except (TypeError, ValueError): conf = 0.0
                cards[tid] = {"confidence": conf, "corrections": corrs}
                if i % 10 == 0: print(f"{part}: {i}/{len(ids)}", flush=True)
        Path(f"{a.out_prefix}_{part}.json").write_text(json.dumps(cards, default=str))
        print(f"{part}: {len(cards)} tasks, {sum(1 for c in cards.values() if c['corrections'])} with corrections, "
              f"{sum(len(c['corrections']) for c in cards.values())} corrections -> {a.out_prefix}_{part}.json", flush=True)


if __name__ == "__main__":
    main()
