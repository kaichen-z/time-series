"""DSL vocabulary AUTO-GROWTH for the retrieval selection DSL. Instead of only tuning weights
over a FIXED feature vocabulary, this GROWS the vocabulary: propose new feature functions, add
each to the registry, re-evolve the weights, and KEEP the feature only if it improves held-out
(dev) F1 -- otherwise drop it. This is "evolving the DSL itself", not just programs in it.

The proposer here is a set of hand-written candidate features (to demonstrate the mechanism
cheaply and deterministically); in production the proposer is an LLM (llm_design.safe_compile
turns LLM-written feature source into a validated fn) -- same keep-if-it-helps loop.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/grow_retrieval_dsl.py
"""
from __future__ import annotations
import random, re
import scripts.evolve_retrieval as R   # load / evolve / fitness / register_feature / feats / FEATURE_FNS

_EVENT = re.compile(r"\b(maintenance|outage|closed|closure|shutdown|downtime|strike|promotion|"
                    r"holiday|repair|upgrade|incident|disruption|suspended|offline|malfunction)\b")
_DIRECTIVE = re.compile(r"\b(shall|must|procedure|directive|standard operating|"
                        r"effective immediately|effective from|scheduled for|will be performed)\b")
_PCT = re.compile(r"\d+(?:\.\d+)?\s?(?:%|percent)")
_YR = re.compile(r"\b(?:19|20)\d{2}\b")


# ---- CANDIDATE NEW FEATURES (the "proposals") ----
def cand_event_specificity(task, doc):
    return min(1.0, len(_EVENT.findall((doc.content or "").lower())) / 2.0)


def cand_directive(task, doc):
    return 1.0 if _DIRECTIVE.search((doc.content or "").lower()) else 0.0


def cand_has_pct(task, doc):
    return 1.0 if _PCT.search(doc.content or "") else 0.0


def cand_recency(task, doc):
    fy = [int(m.group(0)) for m in _YR.finditer(" ".join(str(x) for x in task.future_timestamps))]
    dy = [int(m.group(0)) for m in _YR.finditer(doc.content or "")]
    if not fy or not dy:
        return 0.5
    return max(0.0, 1.0 - min(abs(d - fy[0]) for d in dy) / 5.0)


CANDIDATES = {"event_specificity": cand_event_specificity, "directive": cand_directive,
              "has_pct": cand_has_pct, "recency": cand_recency}

MARGIN = 0.003        # keep a proposed feature only if dev F1 improves by at least this
rng = random.Random(7)

TR, DV = R.load("train"), R.load("dev")
best, _, _ = R.evolve(TR, rng, gens=30)
base_dev = R.fitness(best, DV)
print(f"== DSL auto-growth (keep a proposed feature iff dev F1 improves >= {MARGIN}) ==")
print(f"base vocabulary ({len(R.feats())}): {R.feats()}")
print(f"base dev F1 = {base_dev:.3f}\n")

kept = []
for name, fn in CANDIDATES.items():
    R.register_feature(name, fn)                       # tentatively grow the vocabulary
    TRn, DVn = R.load("train"), R.load("dev")          # recompute features with the new word
    b, _, _ = R.evolve(TRn, rng, gens=30)
    dv = R.fitness(b, DVn)
    if dv >= base_dev + MARGIN:
        kept.append(name); base_dev = dv; best = b; TR, DV = TRn, DVn
        print(f"  + propose '{name}': dev F1 -> {dv:.3f}   ✓ KEEP (vocabulary grew)")
    else:
        del R.FEATURE_FNS[name]                         # reject: shrink back
        print(f"  - propose '{name}': dev F1 {dv:.3f}   ✗ drop (no help)")

TE = R.load("public_test")
print(f"\n== grown vocabulary ({len(R.feats())}): {R.feats()}")
print(f"kept new words: {kept or 'NONE'}")
print(f"final F1: train {R.fitness(best,TR):.3f} | dev {R.fitness(best,DV):.3f} | test {R.fitness(best,TE):.3f}")
print("final weights:", {k: round(best.get(k, 0.0), 2) for k in R.feats() + ['bias']})
print("\n== per-distractor-subtype rejection on TEST (with grown vocabulary) ==")
for st, (rj, tot) in R.subtype_rejection(TE, best).items():
    print(f"    {st:11s} {rj}/{tot} = {100*rj/tot:.0f}%")
