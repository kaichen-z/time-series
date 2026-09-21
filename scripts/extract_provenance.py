"""Reuse cached re-perception responses; parse each correction's PROVENANCE (quote/analog/
conservative). Test if provenance separates HELP from HARM + a grounded-only policy. No new LLM calls."""
from __future__ import annotations
import json, shutil, statistics
from pathlib import Path
from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object, JsonExtractionError
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import _parse, horizon_window_mask, apply_bounded_delta
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics

ROOT=Path('.').resolve(); CAP=5.0
IDH="90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS="external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT=json.loads((ROOT/"splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr=ROOT/"runs/method_evolution/v001"
port=read_policy_file(str(mr/"policies.py"));scr=_load_screening_policy(str(mr/"dictionary.py"))
fs=ForecastStore(ROOT/"runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
   mr/"methods.py",mr/"skills.py",port,None,screening_hash=scr.fingerprint(),
   runtime_identity={},cache_only=True,identity_hash_override=IDH)
llm=ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
     timeout_seconds=900, cache_dir=str(ROOT/".scratch/reperceive-cache")))
# --- replicate reperceive SYSTEM + build_user EXACTLY (so we hit the cache) ---
SYSTEM=("You are a forecasting-correction module. A strong statistical model has ALREADY produced a base "
 "forecast (median values with timestamps). Decide whether the documented real-world events imply "
 "the TRUE values in specific sub-windows differ from this base, and by what MULTIPLICATIVE factor.\n"
 "Rules:\n"
 "1. Anchor to the shown base scale. Correct ONLY sub-windows where a document EXPLICITLY implies a "
 "change (holiday closure, outage, strike, promotion, weather...). Quote the document in a rationale. "
 "Do NOT rescale the whole horizon on vague 'steady/stabilized' wording.\n"
 "2. MAGNITUDE PROVENANCE -- where the number comes from, in priority order:\n"
 "   (a) if the document states an explicit numeric change (e.g. 'fell 30%'), use exactly that;\n"
 "   (b) else estimate the size from the HISTORY analog provided (compare the relevant regime's "
 "level, e.g. weekend vs weekday or the recent level, to the overall level);\n"
 "   (c) if NEITHER a quote NOR a clear historical analog gives the size, DO NOT guess a large "
 "factor -- return a conservative multiplier within [0.9, 1.1], or omit the correction. "
 "Under-correcting is much better than over-correcting.\n"
 "3. If the documents do not clearly imply any deviation, return \"relevant\": false, empty list.\n"
 "4. multiplier < 1 => lower than base, > 1 => higher.\n"
 'Output STRICT JSON: {"relevant": bool, "corrections": [{"start_timestamp": iso, "end_timestamp": '
 'iso, "multiplier": number, "provenance": "quote|analog|conservative", "rationale": str}], '
 '"confidence": number}.')
def wd(ts): p=_parse(ts); return p.weekday() if p else -1
def build_user(task,base):
    n=task.numeric;hv=list(n.history_values);hts=[str(x) for x in task.history_timestamps]
    fts=[str(x) for x in task.future_timestamps]
    we=[x for x,t in zip(hv,hts) if wd(t)>=5];wk=[x for x,t in zip(hv,hts) if 0<=wd(t)<5]
    hist={"typical_scale":round(statistics.median([abs(x) for x in hv]) or 1.0,4),"overall_mean":round(statistics.mean(hv),4),
          "weekday_mean":round(statistics.mean(wk),4) if wk else None,"weekend_mean":round(statistics.mean(we),4) if we else None,
          "last_values":[round(x,4) for x in hv[-12:]]}
    docs=[{"document_id":d.document_id,"content":d.content[:3500]} for d in task.documents]
    return json.dumps({"target_name":task.target_name,"target_description":task.target_description,
        "frequency":n.frequency,"history_summary":hist,"documents":docs,
        "base_forecast_median":[[fts[i],round(base[i],4)] for i in range(len(fts))]},ensure_ascii=False)

cards=json.loads((ROOT/".scratch/cordp_cards_public_test.json").read_text())
ids=[t for t in SPLIT["public_test"]["task_ids"] if t in cards and cards[t].get("corrections")]
tasks={t.numeric.task_id:t for t in load_context_tasks_by_ids(TASKS,tuple(ids))}
def apply(base,ws,fts):
    out=list(base)
    for st,en,m in ws:
        for i,on in enumerate(horizon_window_mask(fts,st,en)):
            if on: out[i]=base[i]*m
    return list(apply_bounded_delta(base,out))
def jt(fc,tr): x=drcik_point_metrics(tr,fc,cap=CAP); return x["smae"]+x["srmse"]
rows=[]; miss=0
for tid in ids:
    t=tasks.get(tid)
    if t is None: continue
    n=t.numeric;truth=list(n.future_values);H=len(truth)
    base=list(fs.forecast("toto_2_0",tuple(n.history_values),n.prediction_length,n.frequency))
    if len(base)!=H: continue
    try:
        resp=llm.complete(system=SYSTEM,messages=[{"role":"user","content":build_user(t,base)}],temperature=0.3)
        out=parse_json_object(resp.text)
    except Exception:
        miss+=1; continue
    fts=[str(x) for x in t.future_timestamps]
    provs={}   # provenance -> list of (start,end,mult)
    for c in (out.get("corrections") or []):
        try: m=float(c.get("multiplier"))
        except (TypeError,ValueError): continue
        pv=(c.get("provenance") or "none").lower()
        provs.setdefault(pv,[]).append((str(c.get("start_timestamp")),str(c.get("end_timestamp")),m))
    rows.append(dict(tid=tid,base=base,truth=truth,fts=fts,provs=provs))
print(f"parsed {len(rows)} tasks (miss {miss})")
# 1) HELP/HARM rate by provenance (apply ONLY that provenance's corrections)
from collections import defaultdict
cnt=defaultdict(lambda:[0,0,0])  # prov -> [help, harm, n]
for r in rows:
    for pv,ws in r["provs"].items():
        d=jt(apply(r["base"],ws,r["fts"]),r["truth"])-jt(r["base"],r["truth"])
        cnt[pv][2]+=1
        if d<-1e-9: cnt[pv][0]+=1
        elif d>1e-9: cnt[pv][1]+=1
print("\nprovenance -> when applied alone: help/harm/n")
for pv,(h,ha,nn) in sorted(cnt.items()):
    print(f"  {pv:14s}: help {h} harm {ha} / {nn}")
# 2) grounded-only policy: apply quote+analog full; drop conservative
def policy(keep):
    bs=br=os_=or_=0.0;n=0;wins=reg=0
    for r in rows:
        ws=[w for pv,lst in r["provs"].items() if pv in keep for w in lst]
        out=apply(r["base"],ws,r["fts"]) if ws else r["base"]
        mb=drcik_point_metrics(r["truth"],r["base"],cap=CAP);mo=drcik_point_metrics(r["truth"],out,cap=CAP)
        bs+=mb["smae"];br+=mb["srmse"];os_+=mo["smae"];or_+=mo["srmse"];n+=1
        if mo["smae"]+mo["srmse"]<mb["smae"]+mb["srmse"]-1e-9: wins+=1
        elif mo["smae"]+mo["srmse"]>mb["smae"]+mb["srmse"]+1e-9: reg+=1
    return (bs-os_)/bs,(br-or_)/br,wins,reg
print("\n== grounded-provenance policy (55 subset) ==")
for lab,keep in [("all provenances",{"quote","analog","conservative","none"}),
                 ("quote only",{"quote"}),("quote+analog",{"quote","analog"}),
                 ("drop conservative",{"quote","analog","none"})]:
    sm,sr,w,rg=policy(keep); print(f"  {lab:20s}: sMAE {sm:+.2%} | sRMSE {sr:+.2%} | wins {w} reg {rg}")
fs.close()
