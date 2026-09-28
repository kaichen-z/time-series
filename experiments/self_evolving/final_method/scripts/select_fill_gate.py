"""Make the two fixed glue steps learnable on Train (2026-09-27):
  Numerical part 2: fill method in {phase_median, linear, snaive, truncate}
  Decision: repair-acceptance margin m (accept iff val_rep < (1-m) * val_raw; m=inf -> never repair)
chosen by Train outcomes of the full pipeline (part-1 base + repair + nrd4 team) with the same do-no-harm
fitness as part 1 (mean gain + 3 x mean negative gain).  Honest estimate: choose on 2 stratified folds,
score the 3rd; then choose on all Train and open Dev once."""
import os, json, sys, copy, itertools, statistics
sys.path.insert(0, '.scratch/self_evolving')
import nrd4 as N4, num_part1 as P1, nrd_coevolve as C
from common.metrics import drcik_point_metrics
TH = json.load(open('.scratch/self_evolving/toto_hindcast.json'))
D = [N4.R3.prep_task(d, TH) for d in json.load(open(os.environ.get('CACHE', '.scratch/self_evolving/nrd_cache_full.json')))]
for d in D: d['_sig'] = {}; h = d['history']; d['_last'], d['_lo'], d['_hi'] = h[-1], min(h), max(h)
prog = json.load(open(os.environ.get('PART1', '.scratch/self_evolving/numerical_part1.json')))['elites'][0]['program']
FV = json.load(open(os.environ.get('FV', '.scratch/self_evolving/fill_variants.json')))
team = json.load(open(os.environ.get('TEAMS', 'experiments/self_evolving/final_method/artifacts/nrd4_final_teams.json')))['final']['1']['team']
FILLS = ["phase_median", "linear", "snaive", "truncate"]; MARGINS = [0.0, 0.05, 0.1, 0.2, 0.3, None]
def out(d, fill, m):
    dd = copy.copy(d); dd['fc'] = dict(d['fc']); v = FV.get(d['tid'], {}).get(fill)
    if m is not None and v and v['val_raw'] is not None and v['val_rep'] < v['val_raw'] * (1 - m): dd['fc']['toto_2_0'] = v['forecast']
    dd['fc'] = {'toto_2_0': P1.run(prog, dd)}; return N4.run_team(team, dd)
def jt(f, d): x = drcik_point_metrics(d['truth'], f, cap=5.0); return x['smae'] + x['srmse']
cache = {}
def gains(cfg, ds):
    return [cache.setdefault((cfg, d['tid']), d['base_jt'] - jt(out(d, *cfg), d)) for d in ds]
def fitness(cfg, ds): g = gains(cfg, ds); return statistics.mean(g) + 3 * statistics.mean(min(0, x) for x in g)
CFGS = list(itertools.product(FILLS, MARGINS))
C.FOLD_MODE = 'strat'; train = [d for d in D if d['part'] == 'train']; folds = C.gfolds(train, 3)
for k in range(3):
    tr = [d for j in range(3) if j != k for d in folds[j]]; best = max(CFGS, key=lambda c: fitness(c, tr))
    g = gains(best, folds[k]); print(f"CV fold {k}: chose fill={best[0]} margin={best[1]} -> held-out mean gain {statistics.mean(g):+.4f}, W/R {sum(x>1e-9 for x in g)}/{sum(x<-1e-9 for x in g)}")
scores = sorted(((fitness(c, train), c) for c in CFGS), reverse=True)
print("Train ranking (top 6):", [(round(f, 4), c) for f, c in scores[:6]])
best = scores[0][1]; print("chosen on all Train:", best)
for part in ('train', 'dev'):
    S = [d for d in D if d['part'] == part]; sb = so = rb = ro = 0; w = r = 0
    for d in S:
        b = drcik_point_metrics(d['truth'], d['fc']['toto_2_0'], cap=5.0); o = drcik_point_metrics(d['truth'], out(d, *best), cap=5.0)
        sb += b['smae']; so += o['smae']; rb += b['srmse']; ro += o['srmse']; dj = (b['smae'] + b['srmse']) - (o['smae'] + o['srmse']); w += dj > 1e-9; r += dj < -1e-9
    n = len(S); print(f"{part}: sMAE {sb/n:.4f}->{so/n:.4f} ({(sb-so)/sb:+.2%}) sRMSE {rb/n:.4f}->{ro/n:.4f} ({(rb-ro)/rb:+.2%}) W/R {w}/{r}")
json.dump({"chosen": best, "ranking": [(f, c) for f, c in scores]}, open(os.environ.get('OUT', '.scratch/self_evolving/fill_gate_choice.json'), 'w'))
