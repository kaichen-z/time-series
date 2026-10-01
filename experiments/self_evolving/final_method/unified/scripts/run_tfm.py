import json, numpy as np, timesfm
m=timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(timesfm.ForecastConfig(max_context=1024,max_horizon=256,normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
J=[j for j in json.load(open("bt/jobs.json")) if j["H"]<=256]; out={}
from collections import defaultdict
byH=defaultdict(list)
for j in J: byH[j["H"]].append(j)
for H,js in byH.items():
    for i in range(0,len(js),64):
        b=js[i:i+64]; p,_=m.forecast(horizon=H,inputs=[j["history"][-1024:] for j in b])
        for j,r in zip(b,np.asarray(p,float)): out[j["key"]]=r[:H].tolist()
json.dump(out,open("bt/tfm.json","w")); print("tfm",len(out))
