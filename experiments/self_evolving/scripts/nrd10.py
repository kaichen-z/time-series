"""nrd4 on the UNION of two independent extractions (old Claude cards + precise GPT cards), with
cross-extractor agreement and source as validator features (2026-09-27)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd4 as N4
N4.FEATS[:] = N4.FEATS[:-1] + ["agree", "src_new", "bias"]
_orig = N4.feats
def feats(team, d, s, e, m):
    f = _orig(team, d, s, e, m); ag = sn = 0.0
    for s2, e2, m2, a, n in d.get("agree", []):
        if s2 == s and e2 == e and abs(m2 - m) < 1e-3: ag, sn = float(a), float(n); break
    f["agree"] = ag; f["src_new"] = sn; return f
N4.feats = feats
if __name__ == "__main__":
    N4.main()
