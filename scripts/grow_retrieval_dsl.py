"""DSL vocabulary AUTO-GROWTH for the retrieval selection DSL. Instead of only tuning weights
over a FIXED feature vocabulary, this GROWS the vocabulary: propose new feature functions, add
each to the registry, re-evolve the weights, and KEEP the feature only if it improves held-out
(dev) F1 -- otherwise drop it. This is "evolving the DSL itself", not just programs in it.

The PROPOSER is switchable:
  PROPOSER=hand (default): a fixed set of hand-written candidate features -- cheap, deterministic,
      good for demonstrating the keep-if-it-helps mechanism without any LLM call.
  PROPOSER=llm            : an LLM WRITES new feature source, llm_design.safe_compile sandboxes and
      validates it, and the SAME keep-if-it-helps loop decides what survives. The human still writes
      the SEED vocabulary (evolve_retrieval.py); the LLM only ADDS, and only validated, auditable
      source that improves held-out dev F1 is kept.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD [PROPOSER=llm] .venv/bin/python scripts/grow_retrieval_dsl.py
"""
from __future__ import annotations
import os, random, re, shutil, statistics
import scripts.evolve_retrieval as R   # load / evolve / fitness / register_feature / feats / FEATURE_FNS
from evolving_loop.data import load_context_tasks_by_ids

PROPOSER = os.environ.get("PROPOSER", "hand")   # hand | llm

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


_HAND_CANDIDATES = {"event_specificity": cand_event_specificity, "directive": cand_directive,
                    "has_pct": cand_has_pct, "recency": cand_recency}


def _llm_candidates():
    """Have an LLM propose new feature source; return only the ones that safe_compile AND pass a
    smoke-test on real train rows. Falls back to {} on any LLM/parse failure (loop still runs)."""
    from scripts.llm_design import propose_features, validate_on
    from common.llm import ClaudeCLIClient, ClaudeCLIConfig
    sample = load_context_tasks_by_ids(R.TASKS_DIR, tuple(R.SPLIT["train"]["task_ids"]))[:12]
    examples = [(d.content or "", d.role == "supporting") for t in sample for d in t.documents]
    supporting = [e for e in examples if e[1]][:6]
    distract = [e for e in examples if not e[1]][:6]
    client = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                             cache_dir=str(R.ROOT / ".scratch/dsl-design-cache")))
    try:
        proposed = propose_features(client, set(R.feats()), supporting + distract, n=6)
    except Exception as e:
        print(f"  (LLM proposer failed: {type(e).__name__}: {e}; no candidates)")
        return {}
    cands = {name: fn for name, fn in proposed.items() if validate_on(fn, sample)}
    print(f"LLM proposed {len(proposed)} compiled / {len(cands)} passed validation: {list(cands)}\n")
    return cands


CANDIDATES = _llm_candidates() if PROPOSER == "llm" else _HAND_CANDIDATES

MARGIN = 0.003        # keep a proposed feature only if K-fold-CV train F1 improves by at least this
K = 5                 # cross-validation folds WITHIN the 80-train (dev/test are never touched here)
rng = random.Random(7)


def cv_score(data, gens=30):
    """K-fold cross-validation F1 WITHIN train: for each fold, evolve weights on the other folds
    and score the held-out fold. A feature is judged by whether it helps ACROSS folds -- not by a
    single 20-task dev, which is too small and gets overfit. Folds are assigned by task index
    (i % K), stable across candidates so comparisons are fair. dev/test stay held-out and honest."""
    scores = []
    for f in range(K):
        val = [d for i, d in enumerate(data) if i % K == f]
        tr = [d for i, d in enumerate(data) if i % K != f]
        if not val or not tr:
            continue
        b, _, _ = R.evolve(tr, random.Random(7 + f), gens=gens)
        scores.append(R.fitness(b, val))
    return statistics.mean(scores) if scores else 0.0


TR = R.load("train")
base_cv = cv_score(TR)
print(f"== DSL auto-growth (keep a proposed feature iff {K}-fold-CV train F1 improves >= {MARGIN}) ==")
print(f"base vocabulary ({len(R.feats())}): {R.feats()}")
print(f"base {K}-fold-CV train F1 = {base_cv:.3f}\n")

kept = []
for name, fn in CANDIDATES.items():
    R.register_feature(name, fn)                       # tentatively grow the vocabulary
    TRn = R.load("train")                              # recompute features with the new word
    cv = cv_score(TRn)
    if cv >= base_cv + MARGIN:
        kept.append(name); base_cv = cv; TR = TRn
        print(f"  + propose '{name}': CV F1 -> {cv:.3f}   ✓ KEEP (vocabulary grew)")
    else:
        del R.FEATURE_FNS[name]                         # reject: shrink back
        print(f"  - propose '{name}': CV F1 {cv:.3f}   ✗ drop (no help)")

best, _, _ = R.evolve(TR, rng, gens=30)                 # final policy: evolve on ALL of train
DV, TE = R.load("dev"), R.load("public_test")
print(f"\n== grown vocabulary ({len(R.feats())}): {R.feats()}")
print(f"kept new words: {kept or 'NONE'}")
print(f"final F1: train {R.fitness(best,TR):.3f} | dev {R.fitness(best,DV):.3f} | test {R.fitness(best,TE):.3f}")
print("final weights:", {k: round(best.get(k, 0.0), 2) for k in R.feats() + ['bias']})
print("\n== per-distractor-subtype rejection on TEST (with grown vocabulary) ==")
for st, (rj, tot) in R.subtype_rejection(TE, best).items():
    print(f"    {st:11s} {rj}/{tot} = {100*rj/tot:.0f}%")
