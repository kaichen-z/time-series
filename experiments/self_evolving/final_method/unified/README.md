# Unified framework: Dr-CiK, Time-MMD, TimesX (2026-10-01)

One pipeline, one search space, three datasets; every choice is made on the dataset's own TRAIN split, dev checks,
test reported once (`results/final_results.json`).

| test sMAE / sRMSE | Toto | TimesFM | **ours** | choice |
|---|---|---|---|---|
| Dr-CiK | .381 / .590 | .383 / .609 | **.284 / .430** | evolved version (dev has only 20 tasks) |
| Time-MMD | .202 / .251 | .185 / .228 | **.185 / .228** | dev picks TimesFM (ties framework blend) |
| TimesX in-distribution | .078 / .095 | .072 / .089 | **.070 / .089** (nMAE .728; PostTime .738) | framework |
| TimesX held-out variables | .084 / .102 | .062 / .078 | **.051 / .066** | framework (cold start) |

Pipeline (`scripts/framework.py`): ② document-gated history repair (fires where anomaly intervals were extracted) →
③ numerical step: the Dr-CiK-evolved program vs a unified Toto/TimesFM/seasonal(/Chronos-2) blend search with grouping
chosen by 5-fold train CV; options `SPLIT=1` (separate weights for the first/second half of the horizon) and `C2=1`
(Chronos-2 member); cold start for variables unseen in train (own-history backtests) → ④ the evolved correction function.
Final rule (`scripts/final_select.py`): if dev has >= 100 tasks, choose on dev among framework variants and single TSFMs.

Other folders:
- `coevo_drcik/`: per-dataset Numerical (`forecast`) <-> Decision (`adjust`) co-evolution (Codex agents, alternating
  phases, hidden fold); `evolved_forecast_drcik.py` / `evolved_adjust_drcik.py` are the accepted Dr-CiK modules
  (dev .2687/.4184, test .2843/.4298). On Time-MMD / TimesX the evolved modules did not improve dev/test.
- `timesx/`: TimesX task construction (PostTime split: 88 ID / 11 OOD variables, cutoff 2025-01-30), baselines,
  text-correction extraction and the negative text results (direction accuracy 55%; trust-all hurts; embedding ridge <1%).
- `scripts/`: TSFM runners (Moirai, Chronos-Bolt, Chronos-2, TimesFM context variants), history backtests, Time-MMD
  statistical methods and past covariates (no gain: covariates are contemporaneous with the target).

Notes: paths assume the repo root as cwd with caches under `.scratch/self_evolving/` and a working dir `work/`
(TimesX data from github.com/haoxin1998/TimesX-project). TimesFM's Time-MMD advantage is almost entirely the US
influenza domain on 2014–2024 windows (test joint error .24 vs .57–.63 for other models, not seen on 1999–2014
train windows), consistent with possible pretraining overlap; without that domain the framework blend beats TimesFM.
