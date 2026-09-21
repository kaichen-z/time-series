"""Adversarial CHALLENGE of the extracted corrections. A skeptical reviewer LLM (fresh framing)
challenges each proposed correction on the 3 data-verified failure modes: (1) is the cited doc a
real DATED EVENT or a description of NORMAL/baseline behavior the base already models? (2) is it
INDEPENDENTLY corroborated? (3) is the direction unambiguous & magnitude supported, not guessed?
Verdict keep/shrink/reject -> apply -> measure whether it cuts the extraction-quality HARM without
killing HELP. One call per task (~55). No evolution yet -- this is the mechanism test."""
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
     timeout_seconds=900, cache_dir=str(ROOT/".scratch/challenge-cache")))
SYSTEM=("You are a SKEPTICAL adversarial reviewer of proposed time-series forecast corrections. A "
 "previous module proposed multiplying specific forecast windows by a factor, citing documents. For "
 "EACH proposed correction, challenge it and return a verdict. Be harsh; the base forecast is strong "
 "and already models normal/seasonal behavior.\n"
 "Challenge on THREE grounds:\n"
 "1. EVENT vs BASELINE: Does a document describe a SPECIFIC, DATED, ANOMALOUS event (closure, outage, "
 "strike, holiday, promotion, weather), or does it merely describe the series' NORMAL/baseline "
 "behavior, methodology, physics, or measurement setup (e.g. 'solar is 0 at night', 'baseline "
 "parameters', 'diagnostic')? If it's normal/baseline description -> REJECT (the base already has it).\n"
 "2. CORROBORATION: Is the event supported by INDEPENDENT evidence (multiple documents, or a clear "
 "historical precedent)? If it rests on ONE ambiguous document -> SHRINK or REJECT.\n"
 "3. DIRECTION & MAGNITUDE: Is the DIRECTION unambiguous for THIS specific entity (e.g. a holiday "
 "LOWERS a commuter road's occupancy but RAISES a leisure road's -- is it clear which)? Is the "
 "MAGNITUDE stated in a document or supported by history, or guessed? If direction is ambiguous, or "
 "magnitude is guessed and large -> SHRINK (halve it) or REJECT.\n"
 'Output STRICT JSON: {"reviews": [{"start_timestamp": iso, "end_timestamp": iso, '
 '"verdict": "keep|shrink|reject", "reason": str}]}. One review per proposed correction, same order.')
def wd(ts): p=_parse(ts); return p.weekday() if p else -1
def build_user(task,base,corrs):
    n=task.numeric;hv=list(n.history_values);hts=[str(x) for x in task.history_timestamps]
    fts=[str(x) for x in task.future_timestamps]
    we=[x for x,t in zip(hv,hts) if wd(t)>=5];wk=[x for x,t in zip(hv,hts) if 0<=wd(t)<5]
    hist={"overall_mean":round(statistics.mean(hv),4),"weekday_mean":round(statistics.mean(wk),4) if wk else None,
          "weekend_mean":round(statistics.mean(we),4) if we else None,"last_values":[round(x,4) for x in hv[-12:]]}
    docs=[{"document_id":d.document_id,"content":(d.content or "")[:3000]} for d in task.documents]
    props=[{"start_timestamp":c[0],"end_timestamp":c[1],"proposed_multiplier":c[2]} for c in corrs]
    return json.dumps({"target_name":task.target_name,"target_description":task.target_description,
        "frequency":n.frequency,"history_summary":hist,"documents":docs,
        "base_forecast_median":[[fts[i],round(base[i],4)] for i in range(0,len(fts),max(1,len(fts)//24))],
        "proposed_corrections":props},ensure_ascii=False)
cards=json.loads((ROOT/".scratch/cordp_cards_public_test.json").read_text())
ids=[t for t in SPLIT["public_test"]["task_ids"] if t in cards and cards[t].get("corrections")]
tasks={t.numeric.task_id:t for t in load_context_tasks_by_ids(TASKS,tuple(ids))}
def jt(fc,tr): x=drcik_point_metrics(tr,fc,cap=CAP); return x["smae"]+x["srmse"]
def apply(base,ws,fts):
    out=list(base)
    for st,en,m in ws:
        for i,on in enumerate(horizon_window_mask(fts,st,en)):
            if on: out[i]=base[i]*m
    return list(apply_bounded_delta(base,out))
new_cards={}
print(f"== challenging {len(ids)} tasks ==",flush=True)
for k,tid in enumerate(ids,1):
    t=tasks.get(tid)
    if t is None: continue
    n=t.numeric;base=list(fs.forecast("toto_2_0",tuple(n.history_values),n.prediction_length,n.frequency))
    corrs=[c for c in cards[tid]["corrections"] if len(c)>=3]
    try:
        resp=llm.complete(system=SYSTEM,messages=[{"role":"user","content":build_user(t,base,corrs)}],temperature=0.0)
        revs=(parse_json_object(resp.text).get("reviews") or [])
    except Exception:
        revs=[]
    kept=[]
    for i,c in enumerate(corrs):
        v=(revs[i].get("verdict") if i<len(revs) else "keep") or "keep"
        m=float(c[2])
        if v=="reject": continue
        if v=="shrink": m=1.0+0.5*(m-1.0)
        kept.append([c[0],c[1],m])
    new_cards[tid]={"confidence":cards[tid].get("confidence"),"corrections":kept}
    Path(".scratch/cordp_cards_challenged.json").write_text(json.dumps(new_cards,default=str))
    vs=[r.get("verdict") for r in revs]
    print(f"[{k}/{len(ids)}] {tid}: {len(corrs)}->{len(kept)} kept  verdicts={vs}",flush=True)
# eval whole-99: original vs challenged (trust-all)
def whole(cardset):
    bs=br=os_=or_=0.0;n=0;wins=reg=0
    for tid in SPLIT["public_test"]["task_ids"]:
        t=tasks.get(tid)
        if t is None: continue
        nn=t.numeric;truth=list(nn.future_values);H=len(truth)
        base=list(fs.forecast("toto_2_0",tuple(nn.history_values),nn.prediction_length,nn.frequency))
        if len(base)!=H: continue
        fts=[str(x) for x in t.future_timestamps]
        ws=[(str(r[0]),str(r[1]),float(r[2])) for r in (cardset.get(tid,{}).get("corrections") or []) if len(r)>=3]
        out=apply(base,ws,fts) if ws else base
        mb=drcik_point_metrics(truth,base,cap=CAP);mo=drcik_point_metrics(truth,out,cap=CAP)
        bs+=mb["smae"];br+=mb["srmse"];os_+=mo["smae"];or_+=mo["srmse"];n+=1
        if mo["smae"]+mo["srmse"]<mb["smae"]+mb["srmse"]-1e-9: wins+=1
        elif mo["smae"]+mo["srmse"]>mb["smae"]+mb["srmse"]+1e-9: reg+=1
    return (bs-os_)/bs,(br-or_)/br,wins,reg
print("\n== whole-99 trust-all: original vs challenged ==")
for nm,cs in [("original",cards),("CHALLENGED",new_cards)]:
    sm,sr,w,r=whole(cs); print(f"  {nm:12s}: sMAE {sm:+.2%} | sRMSE {sr:+.2%} | wins {w} reg {r}")
fs.close()
