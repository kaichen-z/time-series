"""Two-stage: (1) Numerical dictionary self-evolution (history-only hindcasts, MAP-Elites; nrd_dict.py),
(2) nrd4 three-agent co-evolution in which the evolved dictionary is NOT blended into Toto but turned into
Numerical judgement features for the Retrieval validator:
   dcell = how much the cell's dictionary elites beat Toto in hindcast (cell level),
   dtask = how much the best dictionary member beats Toto in this task's own hindcast.
The dictionary is re-evolved inside every CV fold from that fold's Train part only. (2026-09-27)"""
import sys, random, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd4 as N4
import nrd_dict as N

STATE = {"A": None, "cell": {}, "key": 0}
N4.FEATS[:] = N4.FEATS[:-1] + ["dcell", "dtask", "bias"]
_orig_feats = N4.feats


def dict_feats(d):
    k = STATE["key"]
    if d.get("_dk") != k:
        A = STATE["A"]; best = None
        if d.get("toto_h") is not None:
            for m in A.members():
                e = N.hind_err(m, d)
                if e is not None and (best is None or e < best): best = e
        d["_dtask"] = max(-1.0, min(1.0, (d["toto_h"] - best) / (d["toto_h"] + 1e-9))) if best is not None else 0.0
        d["_dcell"] = max(-1.0, min(1.0, -STATE["cell"].get(d["cell"], 0.0)))
        d["_dk"] = k
    return d["_dcell"], d["_dtask"]


def feats(team, d, s, e, m):
    f = _orig_feats(team, d, s, e, m)
    f["dcell"], f["dtask"] = dict_feats(d)
    return f


N4.feats = feats
_orig_evolve = N4.evolve


def evolve(train, rng, a, roles, K=3, log=None):
    if "numerical" in roles:
        A, curve = N.evolve_dictionary(train, {d["tid"]: d["toto_h"] for d in train if d["toto_h"] is not None},
                                       gens=a.dict_gens_stage1, children=24, rng=rng)
        STATE.update(A=A, cell=A.summary(), key=STATE["key"] + 1, curve=[round(c["mean_rel"], 4) for c in curve])
    team, lin = _orig_evolve(train, rng, a, roles, K=K, log=log)
    lin["dict_curve"] = STATE.get("curve"); team["numerical"]["dict_members"] = len(STATE["A"].members()) if STATE["A"] else 0
    return team, lin


N4.evolve = evolve
_orig_main_parse = None

if __name__ == "__main__":
    import argparse
    _AP = argparse.ArgumentParser.parse_args
    def parse(self, *x, **k):
        self.add_argument("--dict-gens-stage1", type=int, default=15)
        return _AP(self, *x, **k)
    argparse.ArgumentParser.parse_args = parse
    N4.main()
