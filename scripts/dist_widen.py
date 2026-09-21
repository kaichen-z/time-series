"""Evolvable DISTRIBUTIONAL widening policy -- the redesigned evolve space that bypasses HELP/HARM.

Instead of moving the point (costly on a confounder), WIDEN the predictive interval in the
document-flagged windows. Widening is do-no-harm by construction (W>=1, never narrow -> worst case
= baseline), and the cost asymmetry (widening a false alarm is cheap, a real event is caught) means
we can act on ALL flagged windows with NO HELP/HARM gating. We evolve the per-task widening factor
W = 1 + relu(w . features) over text-side features, with grouped (leave-domain-out) CV, scored by
CRPS and an RCRPS-style ROI-weighted CRPS (the flagged windows are the region of interest).

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/dist_widen.py
"""
from __future__ import annotations
import json, math, random, statistics
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
fs=ForecastStore(ROOT/"runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
   mr/"methods.py",mr/"skills.py",port,None,screening_hash=scr.fingerprint(),
   runtime_identity={},cache_only=True,identity_hash_override=IDH)
SQ2=math.sqrt(2); SQPI=math.sqrt(math.pi); SQ2PI=math.sqrt(2*math.pi)
FK=["mag","magmax","conf","wfrac","nwin","dir","bias"]; ROIW=5.0

def crps(mu,sig,y):
    sig=max(sig,1e-9); z=(y-mu)/sig
    Phi=0.5*(1+math.erf(z/SQ2)); phi=math.exp(-0.5*z*z)/SQ2PI
    return sig*(z*(2*Phi-1)+2*phi-1/SQPI)

def base_sigma(hv):
    d=[abs(hv[i]-hv[i-1]) for i in range(1,len(hv))] if len(hv)>1 else [1.0]
    return max(statistics.mean(d[-48:] if len(d)>=48 else d),1e-6)

def group(tid):
    d=json.loads((ROOT/TASKS/f"{tid}.json").read_text())
    et=((d.get("showcase") or {}).get("entity") or {}).get("type") or "?"
    return f"{et}|{(d.get('task_metadata') or {}).get('frequency') or ''}"

def load(part):
    cards=json.loads((ROOT/f".scratch/cordp_cards_{part}.json").read_text())
    ids=[t for t in SPLIT[part]["task_ids"] if t in cards and cards[t].get("corrections")]
    tasks={t.numeric.task_id:t for t in load_context_tasks_by_ids(TASKS,tuple(ids))}
    data=[]
    for tid in ids:
        t=tasks.get(tid)
        if t is None: continue
        n=t.numeric;truth=list(n.future_values);H=len(truth)
        p50=list(fs.forecast("toto_2_0",tuple(n.history_values),n.prediction_length,n.frequency))
        if len(p50)!=H: continue
        fts=[str(x) for x in t.future_timestamps];flag=[False]*H;mults=[]
        for r in (cards[tid].get("corrections") or []):
            if len(r)>=3:
                mults.append(float(r[2]))
                for i,on in enumerate(horizon_window_mask(fts,str(r[0]),str(r[1]))):
                    if on: flag[i]=True
        if not any(flag): continue
        dirs=[1 if m>1 else -1 for m in mults]
        feat={"mag":statistics.mean(abs(m-1) for m in mults),"magmax":max(abs(m-1) for m in mults),
              "conf":float(cards[tid].get("confidence") or 0.0),"wfrac":sum(flag)/H,
              "nwin":min(1.0,len(mults)/3.0),"dir":1.0 if len(set(dirs))==1 else 0.0,"bias":1.0}
        data.append(dict(tid=tid,p50=p50,truth=truth,sig0=base_sigma(list(n.history_values)),
                         flag=flag,feat=feat,group=group(tid)))
    return data

def Wfac(w,d):
    return 1.0+max(0.0,sum(w.get(k,0.0)*d["feat"][k] for k in FK))   # widen-only (do-no-harm)

def task_gain(w,d):
    W=Wfac(w,d);H=len(d["truth"]);cb=cp=rb=rp=0.0
    for i in range(H):
        y=d["truth"][i];mu=d["p50"][i];s=d["sig0"]
        c0=crps(mu,s,y); c1=crps(mu,s*W,y) if d["flag"][i] else c0
        wt=ROIW if d["flag"][i] else 1.0
        cb+=c0;cp+=c1;rb+=wt*c0;rp+=wt*c1
    g=(cb-cp)/cb if cb>1e-9 else 0.0
    rg=(rb-rp)/rb if rb>1e-9 else 0.0
    return g,rg

def fit(w,data):
    gs=[task_gain(w,d)[0] for d in data]
    return statistics.mean(gs)-0.5*statistics.mean(min(0.0,x) for x in gs) if gs else 0.0

def evolve(data,rng,gens=40,pop=36,elite=8):
    pop_=[{k:0.0 for k in FK}]+[{k:rng.gauss(0,1.0) for k in FK} for _ in range(pop)]
    best=max(pop_,key=lambda w:fit(w,data))
    for g in range(gens):
        par=sorted(pop_,key=lambda w:fit(w,data),reverse=True)[:elite]
        s=0.6*(1-g/gens)+0.05
        pop_=par+[{k:rng.choice(par).get(k,0.0)+rng.gauss(0,s) for k in FK} for _ in range(pop-elite)]
        cur=max(pop_,key=lambda w:fit(w,data))
        if fit(cur,data)>fit(best,data): best=cur
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
        if val and tr: sc.append(fit(evolve(tr,rng,gens=25,pop=24),val))
    return (statistics.mean(sc),statistics.pstdev(sc) if len(sc)>1 else 0.0)

def report(w,data,tag):
    gs=[task_gain(w,d) for d in data]
    g=statistics.mean(x[0] for x in gs);rg=statistics.mean(x[1] for x in gs)
    wins=sum(1 for x in gs if x[0]>1e-9);reg=sum(1 for x in gs if x[0]<-1e-9)
    print(f"  {tag:26s} CRPS {g:+.2%} | RCRPS(ROI×{ROIW:.0f}) {rg:+.2%} | wins {wins} reg {reg}")

TR,TE=load("train"),load("public_test")
rng=random.Random(20260919)
print(f"== evolvable distributional widening (train {len(TR)} | test {len(TE)}) ==")
learned=evolve(TR,rng);mu,sd=cv(TR,rng)
print(f"grouped-CV CRPS gain {mu:+.4f} ± {sd:.4f}")
print(f"weights: {{{', '.join(f'{k}:{learned.get(k,0):+.2f}' for k in FK)}}}\n")
print("TEST99 (all flagged, NO HELP/HARM gating):")
report({"bias":0.0},TE,"no widen (baseline)")
report({"mag":3.0,"bias":0.0},TE,"hand: widen∝mag")
report(learned,TE,"EVOLVED widen policy")
fs.close()
