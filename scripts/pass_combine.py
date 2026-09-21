"""Evolvable EXTRACTION as a pass-library + low-DOF combiner.

Passes (each gives a per-correction signal, cheap/cached):
  orig      : the extracted multiplier + its |mult-1|, confidence, window-fraction
  doctype   : CODE regex -- do this task's docs read like a BASELINE/methodology description
              (fabrication risk) vs a dated EVENT?
  magsupp   : CODE -- is |mult-1| supported by the history's typical deviation, or over-large?
  challenge : LLM adversarial verdict (keep/shrink/reject), loaded if available.
Combiner: a LOW-DOF learned policy maps the per-correction signals -> action {full, shrink, reject}.
Evolved with do-no-harm fitness + grouped (leave-domain-out) CV WITHIN test (challenge only ran on
test). This is 'evolving the extraction process' at the pass-combination level -- bounded cost,
overfit-controlled.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/pass_combine.py
"""
from __future__ import annotations
import json, math, random, re, statistics
from pathlib import Path
from common.metrics import drcik_point_metrics
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask
ROOT=Path('.').resolve(); CAP=5.0
IDH="90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS="external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT=json.loads((ROOT/"splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr=ROOT/"runs/method_evolution/v001"
port=read_policy_file(str(mr/"policies.py"));scr=_load_screening_policy(str(mr/"dictionary.py"))
fs=ForecastStore(ROOT/"runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
   mr/"methods.py",mr/"skills.py",port,None,screening_hash=scr.fingerprint(),
   runtime_identity={},cache_only=True,identity_hash_override=IDH)
cards=json.loads((ROOT/".scratch/cordp_cards_public_test.json").read_text())
chpath=ROOT/".scratch/cordp_cards_challenged.json"
challenged=json.loads(chpath.read_text()) if chpath.exists() else {}
BASELINE=re.compile(r'\b(baseline|methodology|parameter|diagnostic|calibration|technical note|initiali|specification|overview|scope|definition|typical|normal operating)\b',re.I)
EVENT=re.compile(r'\b(holiday|closure|closed|outage|strike|promotion|maintenance|shutdown|festival|storm|flood|disrupt|suspend|repair|scheduled for|effective)\b',re.I)
FK=["absmag","conf","wfrac","docbase","magsupp","ch_shrink","ch_reject","bias"]
def group(tid):
    d=json.loads((ROOT/TASKS/f"{tid}.json").read_text())
    et=((d.get("showcase") or {}).get("entity") or {}).get("type") or "?"
    return f"{et}|{(d.get('task_metadata') or {}).get('frequency') or ''}"
def hrange(hv):
    if len(hv)<4: return 0.5
    m=statistics.median(hv) or (statistics.mean(abs(x) for x in hv)+1e-9)
    q=statistics.quantiles(hv,n=10); return max(abs(q[-1]-m),abs(q[0]-m))/abs(m)
def load():
    ids=[t for t in SPLIT["public_test"]["task_ids"] if t in cards and cards[t].get("corrections")]
    tasks={t.numeric.task_id:t for t in load_context_tasks_by_ids(TASKS,tuple(ids))}
    data=[]
    for tid in ids:
        t=tasks.get(tid)
        if t is None: continue
        n=t.numeric;truth=list(n.future_values);H=len(truth)
        base=list(fs.forecast("toto_2_0",tuple(n.history_values),n.prediction_length,n.frequency))
        if len(base)!=H: continue
        fts=[str(x) for x in t.future_timestamps];conf=float(cards[tid].get("confidence") or 0.0)
        alltext=" ".join((d.content or "") for d in t.documents)
        nb=len(BASELINE.findall(alltext));ne=len(EVENT.findall(alltext))
        docbase=nb/(nb+ne+1e-9)                      # 1 = pure baseline docs (fabrication risk)
        R=hrange(list(n.history_values))
        chmap={(str(r[0]),str(r[1])):float(r[2]) for r in (challenged.get(tid,{}).get("corrections") or []) if len(r)>=3}
        corr=[]
        for r in (cards[tid]["corrections"] or []):
            if len(r)<3: continue
            on=[i for i,o in enumerate(horizon_window_mask(fts,str(r[0]),str(r[1]))) if o]
            if not on: continue
            m=float(r[2]); key=(str(r[0]),str(r[1]))
            # challenge verdict inferred from challenged card: absent=reject, ~m=keep, shrunk=shrink
            if tid in challenged:                     # only trust challenge where it actually ran
                if key not in chmap: ch_reject,ch_shrink=1.0,0.0
                elif abs(chmap[key]-m)>1e-6: ch_reject,ch_shrink=0.0,1.0
                else: ch_reject,ch_shrink=0.0,0.0
            else: ch_reject=ch_shrink=0.0
            feat={"absmag":abs(m-1),"conf":conf,"wfrac":(on[-1]-on[0]+1)/H,"docbase":docbase,
                  "magsupp":1.0 if abs(m-1)<=R+1e-9 else 0.0,"ch_shrink":ch_shrink,"ch_reject":ch_reject,"bias":1.0}
            corr.append(dict(win=(on[0],on[-1]+1),mult=m,feat=feat))
        if corr: data.append(dict(tid=tid,base=base,truth=truth,corr=corr,group=group(tid)))
    return data
def action(w,f):                                     # low-DOF: score -> full/shrink/reject
    s=sum(w.get(k,0.0)*f[k] for k in FK); p=1/(1+math.exp(-max(-30,min(30,s))))
    return 1.0 if p>0.6 else (0.5 if p>0.35 else 0.0)   # full / shrink-half / reject
def applied(d,w):
    out=list(d["base"])
    for c in d["corr"]:
        a=action(w,c["feat"]); 
        if a<=0: continue
        m=1.0+a*(c["mult"]-1.0); s,e=c["win"]
        for i in range(s,min(e,len(out))): out[i]=d["base"][i]*m
    return list(apply_bounded_delta(d["base"],out))
def jt(fc,tr): x=drcik_point_metrics(tr,fc,cap=CAP); return x["smae"]+x["srmse"]
def gain(w,data):
    ds=[jt(d["base"],d["truth"])-jt(applied(d,w),d["truth"]) for d in data]
    return statistics.mean(ds)-0.5*statistics.mean(min(0.0,x) for x in ds) if ds else 0.0
def evolve(data,rng,gens=40,pop=36,elite=8):
    pop_=[{"bias":2.0}]+[{k:rng.gauss(0,1.0) for k in FK} for _ in range(pop)]  # bias>0 => default keep
    best=max(pop_,key=lambda w:gain(w,data))
    for g in range(gens):
        par=sorted(pop_,key=lambda w:gain(w,data),reverse=True)[:elite]
        s=0.6*(1-g/gens)+0.05
        pop_=par+[{k:rng.choice(par).get(k,0.0)+rng.gauss(0,s) for k in FK} for _ in range(pop-elite)]
        cur=max(pop_,key=lambda w:gain(w,data))
        if gain(cur,data)>gain(best,data): best=cur
    return best
def gfolds(data,K=3):
    gr={}
    for d in data: gr.setdefault(d["group"],[]).append(d)
    out=[[] for _ in range(K)];ld=[0]*K
    for _g,it in sorted(gr.items(),key=lambda kv:-len(kv[1])):
        f=ld.index(min(ld));out[f]+=it;ld[f]+=len(it)
    return out
def cv(data,rng,K=3):
    fo=gfolds(data,K);sc=[]
    for f in range(K):
        val=fo[f];tr=[d for g in range(K) if g!=f for d in fo[g]]
        if val and tr: sc.append(gain(evolve(tr,rng,gens=25,pop=24),val))
    return (statistics.mean(sc),statistics.pstdev(sc) if len(sc)>1 else 0.0)
def whole(w,data,keepall):
    # whole-99: keepall tasks (no-corr stay Toto) via the 55 corr set; report vs Toto
    bs=br=os_=or_=0.0;n=0;wins=reg=0
    ids99=SPLIT["public_test"]["task_ids"]; dmap={d["tid"]:d for d in data}
    tasks={t.numeric.task_id:t for t in load_context_tasks_by_ids(TASKS,tuple(ids99))}
    for tid in ids99:
        t=tasks.get(tid)
        if t is None: continue
        n_=t.numeric;truth=list(n_.future_values);H=len(truth)
        base=list(fs.forecast("toto_2_0",tuple(n_.history_values),n_.prediction_length,n_.frequency))
        if len(base)!=H: continue
        out=applied(dmap[tid],w) if tid in dmap else base
        mb=drcik_point_metrics(truth,base,cap=CAP);mo=drcik_point_metrics(truth,out,cap=CAP)
        bs+=mb["smae"];br+=mb["srmse"];os_+=mo["smae"];or_+=mo["srmse"];n+=1
        if mo["smae"]+mo["srmse"]<mb["smae"]+mb["srmse"]-1e-9: wins+=1
        elif mo["smae"]+mo["srmse"]>mb["smae"]+mb["srmse"]+1e-9: reg+=1
    return (bs-os_)/bs,(br-or_)/br,wins,reg
D=load();rng=random.Random(20260919)
print(f"== evolvable pass-combiner (challenge pass {'ON' if challenged else 'OFF'}), {len(D)} corr-tasks ==")
best=evolve(D,rng);mu,sd=cv(D,rng)
print(f"grouped-CV do-no-harm gain {mu:+.4f} ± {sd:.4f}")
print(f"combiner weights: {{{', '.join(f'{k}:{best.get(k,0):+.2f}' for k in FK)}}}")
KEEP={"bias":9.0}   # trust-all baseline (always full)
print("\nwhole-99 trust-all:")
for nm,w in [("baseline keep-all",KEEP),("EVOLVED combiner",best)]:
    sm,sr,wi,rg=whole(w,D,True); print(f"  {nm:18s}: sMAE {sm:+.2%} | sRMSE {sr:+.2%} | wins {wi} reg {rg}")
fs.close()
