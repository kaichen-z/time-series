import json,sys,copy
sys.path.insert(0,'.scratch/self_evolving')
import nrd4 as N4
from common.metrics import drcik_point_metrics
TH=json.load(open('.scratch/self_evolving/toto_hindcast.json'))
D=[N4.R3.prep_task(d,TH) for d in json.load(open('.scratch/self_evolving/nrd_cache.json'))]
for d in D: d['_sig']={}
rep=json.load(open('.scratch/self_evolving/tl2_repair_evolved.json')); fc=json.load(open('.scratch/self_evolving/tl2_fc_evolved.json')); val=json.load(open('.scratch/self_evolving/tl3_val.json'))
R=json.load(open('.scratch/self_evolving/nrd4_final_numerical_retrieval_decision.json'))
def m(f,t): x=drcik_point_metrics(t['truth'],f,cap=5.0); return x['smae'],x['srmse']
for seed,v in R['final'].items():
    team=v['team']; line=[]
    for part in ('train','dev'):
        sb=so=rb=ro=0;w=r=0
        for d in [d for d in D if d['part']==part]:
            toto=d['fc']['toto_2_0']; b=m(toto,d)
            vv=val.get(d['tid']); acc = d['tid'] in rep[part] and vv is not None and vv['repaired']<vv['raw']*0.9
            dd=d
            if acc:
                dd=copy.copy(d); dd['fc']={'toto_2_0':fc[d['tid']]}
            out=N4.run_team(team,dd); o=m(out,d)
            sb+=b[0];so+=o[0];rb+=b[1];ro+=o[1]; dj=sum(b)-sum(o); w+=dj>1e-9; r+=dj<-1e-9
        line.append(f"{part}: joint {((sb+rb)-(so+ro))/(sb+rb):+.2%} (sMAE {(sb-so)/sb:+.2%}, sRMSE {(rb-ro)/rb:+.2%}) W/R {w}/{r}")
    print('seed',seed,' | '.join(line))
