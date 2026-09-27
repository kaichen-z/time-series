import json
from common.metrics import drcik_point_metrics
D={d['tid']:d for d in json.load(open('.scratch/self_evolving/nrd_cache.json'))}
rep=json.load(open('.scratch/self_evolving/tl2_repair_evolved.json')); fc=json.load(open('.scratch/self_evolving/tl2_fc_evolved.json')); val=json.load(open('.scratch/self_evolving/tl3_val.json'))
def m(f,t): x=drcik_point_metrics(D[t]['truth'],f,cap=5.0); return x['smae'],x['srmse']
for margin in (None,0.0,0.05,0.1,0.2):
    line=[]
    for part in ('train','dev'):
        sb=so=rb=ro=0; w=r=n=0
        for t in [t for t in D if D[t]['part']==part]:
            b=m(D[t]['fc']['toto_2_0'],t); o=b
            if t in rep[part]:
                v=val.get(t); ok = margin is None or (v is not None and v['repaired'] < v['raw']*(1-margin))
                if ok: o=m(fc[t],t); n+=1
            sb+=b[0];so+=o[0];rb+=b[1];ro+=o[1]; dj=sum(b)-sum(o); w+=dj>1e-9; r+=dj<-1e-9
        line.append(f"{part}: repaired {n}, joint {((sb+rb)-(so+ro))/(sb+rb):+.2%} (sMAE {(sb-so)/sb:+.2%}, sRMSE {(rb-ro)/rb:+.2%}) W/R {w}/{r}")
    print('gate' if margin is not None else 'no gate', margin, ' | '.join(line))
