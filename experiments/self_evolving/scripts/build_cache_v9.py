import json
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import horizon_window_mask
cards=json.load(open('.scratch/self_evolving/cards_v9.json')); D=json.load(open('.scratch/self_evolving/nrd_cache.json'))
T="external/Dr-CiK/full-download/Dr-CiK_public/tasks"; n=0
for d in D:
    c=cards.get(d['tid']) or {}; d['conf']=float(c.get('confidence') or 0); d['corr']=[]
    raw=json.load(open(f"{T}/{d['tid']}.json")); fts=[str(x) for x in raw['series']['future_timestamps']]
    for r in c.get('corrections') or []:
        try: on=[i for i,o in enumerate(horizon_window_mask(fts,str(r[0]),str(r[1]))) if o]
        except Exception: on=[]
        if on: d['corr'].append([on[0],on[-1]+1,float(r[2])]); n+=1
json.dump(D,open('.scratch/self_evolving/nrd_cache_v9.json','w')); print('corrections',n,'tasks with',sum(bool(d['corr']) for d in D))
