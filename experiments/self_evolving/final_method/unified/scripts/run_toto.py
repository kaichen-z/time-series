import json, torch
from toto2 import Toto2Model
P=32; dev=torch.device("cuda"); M=Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(dev).eval(); out={}
for j in json.load(open("bt/jobs.json")):
    hv=j["history"]; pad=(-len(hv))%P
    x=torch.tensor([0.0]*pad+hv,dtype=torch.float32,device=dev).reshape(1,1,-1); m=torch.ones_like(x,dtype=torch.bool)
    if pad: m[...,:pad]=False
    with torch.no_grad(): q=M.forecast({"target":x,"target_mask":m,"series_ids":torch.zeros((1,1),dtype=torch.long,device=dev)},horizon=j["H"],decode_block_size=None,has_missing_values=False)
    q=[qq.reshape(-1).tolist() if hasattr(qq,"reshape") else qq for qq in (q if isinstance(q,(list,tuple)) else list(q))]
    out[j["key"]]=q[4]
json.dump(out,open("bt/toto.json","w")); print("toto",len(out))
