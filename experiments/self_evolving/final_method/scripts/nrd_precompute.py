"""Cache everything the N/R/D engine needs (2026-09-26). Labels stay in 'truth'; agents never see it."""
import json
from pathlib import Path
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import horizon_window_mask
ROOT=Path('.').resolve()
IDH="90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS="external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT=json.loads((ROOT/"splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr=ROOT/"runs/method_evolution/v001"
port=read_policy_file(str(mr/"policies.py"));scr=_load_screening_policy(str(mr/"dictionary.py"))
fs=ForecastStore(ROOT/"runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",mr/"methods.py",mr/"skills.py",port,None,screening_hash=scr.fingerprint(),runtime_identity={},cache_only=True,identity_hash_override=IDH)
METHODS=json.load(open('.scratch/self_evolving/full_methods.json'))
out=[]
for part,cf in (("train","train"),("dev","dev"),("public_test","public_test")):
    cards=json.loads((ROOT/f".scratch/cordp_cards_{cf}.json").read_text())
    ids=SPLIT[part]["task_ids"]
    for t in load_context_tasks_by_ids(TASKS,tuple(ids)):
        n=t.numeric; tid=n.task_id; H=n.prediction_length
        raw=json.loads((ROOT/TASKS/f"{tid}.json").read_text())
        et=((raw.get("showcase") or {}).get("entity") or {}).get("type") or "?"
        fc={}
        for m in METHODS:
            try:
                v=list(fs.forecast(m,tuple(n.history_values),H,n.frequency))
                if len(v)==H: fc[m]=v
            except Exception: pass
        fts=[str(x) for x in t.future_timestamps]
        corr=[]
        c=cards.get(tid) or {}
        for r in (c.get("corrections") or []):
            if len(r)<3: continue
            on=[i for i,o in enumerate(horizon_window_mask(fts,str(r[0]),str(r[1]))) if o]
            if on: corr.append([on[0],on[-1]+1,float(r[2])])
        out.append(dict(tid=tid,part=part,group=f"{et}|{(raw.get('task_metadata') or {}).get('frequency') or ''}",
            freq=n.frequency,H=H,history=list(n.history_values),truth=list(n.future_values),fc=fc,
            docs=[(d.content or "") for d in t.documents],conf=float(c.get("confidence") or 0.0),corr=corr))
json.dump(out,open('.scratch/self_evolving/nrd_cache.json','w'))
print(len(out),{p:sum(1 for o in out if o['part']==p) for p in ('train','dev','public_test')},min(len(o['fc']) for o in out))
