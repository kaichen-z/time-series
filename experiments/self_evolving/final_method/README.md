# Final method: two-part Numerical + gated history repair + three-agent future correction (Dr-CiK)

**Numerical, part 1: evolve a candidate dictionary of forecast-combination programs**
- Generation 0 is the single methods of the dictionary (Toto is just one of them): the 31 original methods (deep forecasters and statistical baselines) plus the full statistical portfolio from [`scripts/stat_full.py`](scripts/stat_full.py). Of the 93 statistical methods, 48 produce forecasts on Dr-CiK; the others are intermittent-demand methods, need longer histories, or do not support the task frequency. The 37 methods that cover at least 90% of Train tasks are used.
- Programs combine the cached method forecasts by:
  - weighted sums;
  - difference corrections (`c·(M_a − M_b)`);
  - shrinkage toward the last observation;
  - clipping to the history range.
- Evolved with typed mutations on Train outcomes. Fitness = mean gain + 3 × mean negative gain, a do-no-harm penalty.
- A diverse set of elites is kept as the dictionary; the best elite is the base forecaster.
- Evolution picks Toto by itself as the main component: **0.95·Toto + 0.05·arima_auto**, **shrunk 20% toward the last observation** (no difference term, no clipping).
- **Refined by agent co-evolution** (see *Co-evolution* below): a Codex research agent, checked by a hidden Train fold, added a 10% TimesFM-2.5 term. The main method's base forecast is therefore **0.854·Toto + 0.046·arima_auto + 0.100·timesfm_2_5, shrunk 20%** ([`artifacts/numerical_part1_main.json`](artifacts/numerical_part1_main.json)).

**Numerical, part 2 + Retrieval + Decision: the effective evolution on top of that base**
1. **History repair.** Retrieval (GPT with evolved extraction instructions) finds past anomalies that will not recur. Numerical imputes them and Toto re-forecasts. Decision accepts a repair only if a history-only back-test error drops by at least **30%**; both this margin and the fill method (**same-phase median**, chosen among phase median / linear / seasonal naive / truncate) are **learned on Train** by `select_fill_gate.py`. When accepted, the Toto term of the base program uses the repaired-history forecast.
2. **Future correction.** **Code-evolved correction functions, routed by task type** ([`artifacts/correction_function_main.py`](artifacts/correction_function_main.py)) apply the document corrections to future steps. It started as the evolved nrd4 team rewritten as code (Numerical: magnitude calibrator and per-cell Toto trust; Retrieval: per-correction validator; Decision: strength and trust gate) and was then rewritten by Codex agents (Meta-Harness + CORAL style, see below).

## Results

| | Dev sMAE | Dev sRMSE | Test sMAE (exploratory) | Test sRMSE (exploratory) | Test improved / worsened |
|---|---|---|---|---|---|
| Toto | 0.3921 | 0.5871 | 0.3814 | 0.5903 | — |
| Hand-designed pass-combiner | — | — | 0.3564 (+6.5%) | 0.5310 (+10.1%) | 30 / 1 |
| Toto-anchored variant (no Numerical part 1) | −0.3% | +0.03% | 0.3291 (+13.7%) | 0.5208 (+11.8%) | 32–34 / 2–3 |
| Previous version (hand-set 10% margin) | 0.3567 (+9.0%) | 0.5231 (+10.9%) | 0.3405 (+10.7%) | 0.5222 (+11.5%) | 64–65 / 34–35 |
| Previous version (31-method dictionary) | 0.3451 (+12.0%) | 0.5110 (+13.0%) | 0.3326 (+12.8%) | 0.5102 (+13.6%) | 64–65 / 34–35 |
| Previous main (evolved part 1 without the TimesFM term) | 0.3380 (+13.8%) | 0.5064 (+13.8%) | 0.3281 (+14.0%) | 0.5097 (+13.7%) | 64–65 / 34–35 |
| Previous main (nrd4 team for future correction, 3-seed mean) | 0.3285 (+16.2%) | 0.4885 (+16.8%) | 0.3253 (+14.7%) | 0.5018 (+15.0%) | 63–64 / 35–36 |
| Code-evolved correction function (CORAL group only) | 0.2822 (+28.0%) | 0.4258 (+27.5%) | 0.3071 (+19.5%) | 0.4683 (+20.7%) | 64 / 35 |
| **This method (5 correction functions routed by task type)** | **0.2822 (+28.0%)** | **0.4258 (+27.5%)** | **0.2923 (+23.4%)** | **0.4412 (+25.3%)** | **65 / 34** |

- The correction function is deterministic (it replaces the seed-dependent nrd4 team), so its row is a single run; the previous main row is the 3-seed mean (per seed test sMAE 0.3223 / 0.3281 / 0.3254, sRMSE 0.4982 / 0.5059 / 0.5014).
- On dev, the average error drops a lot but only 9 of 20 tasks improve (11 get worse, mostly slightly); the gain comes from a few badly forecast tasks.
- Dev alone favours TimesFM (TimesFM by itself: dev 0.3106 / 0.4502, but test 0.3829 / 0.6094, about Toto), so the dev gain of the TimesFM term overstates it; the test gain is smaller but consistent across seeds.
- **The dev gate is passed on both metrics** (dev is used once, after all evolution on Train).

Caveats:
- test99 had been opened before, so test numbers are exploratory.
- The part-1 penalty weight (3) was chosen after seeing a penalty-0.5 run on dev and test, which is mild contamination.
- The fill method and repair margin are chosen on Train only. Nested CV held-out gains (joint error reduction) are +0.230, +0.155 and +0.338.
- Still hand-designed: the extraction schema (5 interval kinds) and the fixed CorDP cards that nrd4 corrects.
- The shrink/clip transform applies to every task. It helps the few badly-forecast tasks a lot and slightly worsens many others, so about 35 test tasks get (mostly slightly) worse. The Toto-anchored variant is the "almost no harm" alternative.

## Evidence extraction quality (all splits)

Evidence extraction quality against the annotated `gt_evidence` on every split (the instructions were evolved on Train only; dev and test annotations were never used during evolution). Mean per task over tasks with annotations:

| Split | Instructions | Tasks | F1 | Recall | Precision |
|---|---|---|---|---|---|
| Train | initial | 79 | 0.039 | 0.054 | 0.091 |
| Train | evolved | 79 | 0.477 | 0.483 | 0.670 |
| Dev | initial | 19 | 0.000 | 0.000 | 0.000 |
| Dev | evolved | 19 | 0.335 | 0.316 | 0.667 |
| Test (public 99) | initial | 95 | 0.019 | 0.026 | 0.050 |
| Test (public 99) | evolved | 95 | 0.443 | 0.439 | 0.693 |

## Code evolution of the correction function: Meta-Harness + CORAL (2026-09-28)

The future-correction step (how document corrections change the forecast) was the only part that had only been tuned as parameters. Here agents **rewrite its code**.

- **Interface**: `adjust(view) -> list[H]`. `view` has the history, the base forecast, the documents, document statistics, cell / task Toto back-test errors, a normal-deviation scale, and the extracted corrections (start, end, multiplier). No labels.
- **Seed**: the previous main method's nrd4 team (seed 1) rewritten as plain code ([`scripts/meta_harness/seed_harness.py`](scripts/meta_harness/seed_harness.py)); it reproduces the previous main method exactly (visible 0.1948, dev 0.3285 / 0.4885).
- **Meta-Harness part**: the proposer edits code and can read every earlier version, its scores, and per-task traces of the visible tasks (truth, base-only gain, gain of each correction applied alone).
- **CORAL part**: several agents with shared attempts / notes / skills, heartbeats (notes, consolidation, redirection), and a separated evaluator ([`hevald.py`](scripts/meta_harness/hevald.py)): scores and per-task gains on Train folds 0–1, only pass/fail on hidden fold 2.
- **Runs** (Codex gpt-6-sol, 20 submissions per agent): single agent (plain Meta-Harness); 3 agents with shared memory (CORAL); 3 independent agents.
- **Correction source**: before this, the old document cards were compared with cards from the unified extractor in nested CV (held-out joint gain ≈ +6% vs ≈ +2%); the old cards were kept.

| Run | Visible fitness (seed 0.195) | Hidden fold (seed ≈ 0.35) | Dev sMAE / sRMSE | Test sMAE / sRMSE (exploratory) |
|---|---|---|---|---|
| Seed (previous main) | 0.195 | — | 0.3285 / 0.4885 | 0.3223 / 0.4982 |
| Single agent (Meta-Harness) | 0.312 | 0.431 | 0.3288 / 0.4887 | 0.3046 / 0.4583 |
| **3 agents, shared memory (CORAL)** | 0.306 | 0.401 | **0.2822 / 0.4258** | 0.3071 / 0.4683 |
| Independent agent 1 | 0.334 | 0.387 | 0.3254 / 0.4843 | 0.3144 / 0.4713 |
| Independent agent 2 | 0.264 | 0.370 | 0.3300 / 0.4893 | 0.3120 / 0.4800 |
| Independent agent 3 | 0.332 | 0.431 | 0.3285 / 0.4885 | 0.3104 / 0.4630 |

**Routing the five functions** ([`scripts/meta_harness/ensemble.py`](scripts/meta_harness/ensemble.py)). The functions are good on different task types, so they were combined, choosing on Train only (visible folds, hidden fold as check):

| Combination | Visible | Hidden |
|---|---|---|
| CORAL function alone | 0.306 | 0.401 |
| Best single function (independent 1) | 0.334 | 0.387 |
| Mean / median / rank-weighted mean | 0.325 / 0.333 / 0.336 | 0.411 / 0.408 / 0.413 |
| **Route by cell** | **0.372** | **0.444** |

Routing table (per cell, the function with the best mean gain on the visible folds): daily and minute-level → single agent; hourly non-seasonal → independent 1; hourly seasonal → independent 3; second-level → CORAL. Dev is identical to the CORAL function alone (0.2822 / 0.4258; the dev tasks with corrections all route to it), test (exploratory) improves from 0.3071 / 0.4683 to 0.2923 / 0.4412. The routed version is the main method's correction step.

The shared-memory (CORAL) function was the best single function by dev. What the agents changed (readable in the code):
1. **Event windows**: the extractor often includes the "back to normal" step; windows are shortened by one step, and window lengths are corrected from phrases such as "one-hour" or "four-day".
2. **Bounds**: the ±50% per-step bound is lifted for outage / zero events and short hourly surges, so the documented multiplier is applied directly.
3. **Physical constraints**: forecasts of non-negative series are floored at 0; a positive shift is not applied when the documents report zero readings.
4. The CORAL function also extrapolates very smooth, monotonically declining series with a quadratic (a change of the base forecast, not a document correction).

Caveats: some rules are triggered by document phrases and may be specific to how Dr-CiK documents are written; dev was used once for all five candidates, so picking the best by dev is slightly optimistic; only 12 dev tasks carry corrections. Agent logs show no access to dev, test or hidden-fold labels; the evaluator's private data directory was readable in principle, but was not accessed. Code, notes, skills, all submissions and results per run: [`artifacts/meta_harness/`](artifacts/meta_harness/).

## Co-evolution: one end-to-end score, then agents (2026-09-28)

Motivation: the parts above are evolved with their own scores (extraction with evidence F1, part 1 with base-forecast error), so an improvement of one part need not improve the final forecast. Evolving the extraction instructions on all 80 Train tasks raised evidence F1 from 0.477 to 0.553, yet the pipeline got worse (dev 0.3477 / 0.5180; Train then chose "never repair").

**Step 1: cooperative co-evolution under one end-to-end score** ([`scripts/coevolution/e2e.py`](scripts/coevolution/e2e.py), [`coevo.py`](scripts/coevolution/coevo.py)).
- Every role (Numerical: program, calibrator; Retrieval: validator, extraction instructions; Decision: fill, margin, strength, trust gate) mutates in turn; a change is kept only if the whole pipeline's error on the Train folds improves.
- Nested CV (evolve on 2 folds, score the 3rd): Train fitness rises, but the held-out fold gets **worse in all three folds** (+0.247→+0.215, +0.172→+0.132, +0.338→+0.201; with a stricter "no fold worse" rule +0.226 / +0.145 / +0.295). Random search around the main method overfits 80 tasks. Logs: [`artifacts/coevolution/step1_logs/`](artifacts/coevolution/step1_logs/).

**Step 2: agent proposers with a hidden fold (CORAL / Meta-Harness style)** ([`scripts/coevolution/`](scripts/coevolution/): `evald.py`, `submit.py`, `run_agent.py`, `TASK.md`).
- Codex agents (gpt-6-sol) edit any part of the config. They never run the evaluator: they queue a config, and a separate daemon returns scores and per-task gains on two **visible** Train folds plus only pass/fail from a **hidden** third fold. A config becomes the shared best only if the visible score improves and the hidden fold does not get worse.
- Shared memory: all attempts, notes, reusable scripts, and full traces of the visible tasks. Heartbeats: take notes every episode, consolidate every second episode, change direction after an episode without progress.
- Setup: group A = 3 agents with shared memory; group B = 3 independent agents; budget 20 submissions each (agents used 5–14).
- 17 submissions improved the visible folds but were stopped by the hidden fold. Accepted: A, 3% then 5% Moirai in the base forecast; B, **10% TimesFM-2.5**. Changes to extraction, validator and Decision never passed.
- Dev (once, at the end) / test seed 1: main 0.3380 / 0.5064 and 0.3251 / 0.5061; A 0.3354 / 0.5014 and 0.3244 / 0.5035; **B 0.3285 / 0.4885 and 0.3223 / 0.4982**. B was confirmed over 3 seeds (table above) and adopted.
- Shared memory did **not** beat the best independent agent here (small sample: 3 agents per group).
- Audit: no agent read dev, test or hidden-fold label files. One group-A agent read the evaluator's state file (hidden-fold aggregate scores, no per-task labels) in episode 3, after its accepted changes in episode 2.
- Notes, scripts, attempts and best configs of every run: [`artifacts/coevolution/step2/`](artifacts/coevolution/step2/).

## Dictionary expansion: all 5 foundation models and 5 combined policies (ablation)

The portfolio also contains 5 foundation models (Toto-2.0, Chronos-Bolt, TimesFM-2.5, Moirai-2.0, Granite-TTM-r2) and 5 combined policies. Each combined policy pairs one foundation model with one statistical method, either as a weighted mean or as a router switched by a series signal.
- [`scripts/timesfm_full.py`](scripts/timesfm_full.py) runs TimesFM-2.5 with the repo adapter's settings (separate venv, `timesfm` ≥ 2.5).
- [`scripts/tsfm_full.py`](scripts/tsfm_full.py) runs Moirai-2.0 and Granite-TTM through the repo's manifest-bound worker runtime (worker envs from `configs/tsfm-environments/{uni2ts,granite_tsfm}.txt`; see `scripts/tsfm_workers.example.json`).
- [`scripts/combined_full.py`](scripts/combined_full.py) computes the 5 combined policies with the repo's own combination code and writes `nrd_cache_full2.json`.
- Coverage: TimesFM and Moirai 199/199 tasks. Granite-TTM covers only 126/199, because it supports only hourly/daily/weekly data and horizons ≤ 96. Part 1 now selects from **43 methods**.

Single methods (sMAE / sRMSE):

| Method | Train | Dev | Test (looked at only, not used for selection) |
|---|---|---|---|
| Toto-2.0 | 0.3884 / 0.6018 | 0.3921 / 0.5871 | 0.3814 / 0.5903 |
| TimesFM-2.5 | 0.3567 / 0.5637 (worse than Toto on 51 of 80 tasks) | 0.3106 / 0.4502 | 0.3829 / 0.6094 |
| Moirai-2.0 | 0.4249 / 0.6314 | 0.3511 / 0.5027 | 0.3934 / 0.6175 |
| TimesFM + seasonal naive (combined) | 0.3560 / 0.5654 | 0.3008 / 0.4506 | 0.3796 / 0.5966 |

Part 1 re-evolved on the 43-method dictionary, with the original Toto-referenced fitness and with a **Toto-free fitness** ([`scripts/num_part1_rel.py`](scripts/num_part1_rel.py)). The Toto-free fitness scores each task's relative gain over the median error of all single methods, clipped to [−1, 1]. Artifacts are in [`artifacts/dictionary_expansion/`](artifacts/dictionary_expansion/).

| Fitness | Best program | Train (part 1 only) | Dev (part 1 only) |
|---|---|---|---|
| vs Toto, penalty 3 | Toto + small difference term, shrink 0.20 | 0.3266 / 0.5273 | 0.3551 / 0.5122 |
| vs Toto, penalty 0 | TimesFM+seasonal + robust trend, shrink 0.28, clip | 0.3055 / 0.4762 | 0.3963 / 0.5453 (overfits) |
| vs Toto, penalty 0.5 | TimesFM + SES + … , clip | 0.3002 / 0.4729 | 0.3873 / 0.5474 (overfits) |
| Toto-free, penalty 0, ≤ 2 terms | 0.93·Toto + 0.07·ARIMA, shrink 0.09 | 0.3192 / 0.5239 | 0.3576 / 0.5305 |
| Toto-free, penalty 0, ≤ 4 terms | 0.89·Toto + 0.11·ARIMA, shrink 0.10 | 0.3350 / 0.5382 | 0.3516 / 0.5210 |
| Toto-free, penalty 0.5, ≤ 2 terms | 0.92·Toto + 0.08·ARIMA, shrink 0.12 | 0.3154 / 0.5180 | 0.3529 / 0.5234 |

Findings:
- **Even when Toto has no special role in the fitness, evolution picks Toto + a little ARIMA + shrinkage.** So the base forecaster is not Toto just because Toto was the reference. Nested CV held-out sMAE for the Toto-free (0.5, ≤ 2) setting: +13.5%, +10.9%, +5.8%.
- TimesFM's dev advantage does not hold on Train (51/80 tasks worse than Toto) or on test (≈ Toto). Selecting on Train correctly does not choose it.
- The full pipeline with the Toto-free part 1 (0.5, ≤ 2) gives dev 0.3415 / 0.5106, slightly worse than the main method (0.3380 / 0.5064), so **the main method is unchanged**.

## Scripts that produce the evolved artifacts

Per-generation results are in [`EVOLUTION_LOG.md`](EVOLUTION_LOG.md); the design of every evolve space (genome, mutations, fitness, selection, data use) is in [`EVOLVE_SPACE_zh.md`](EVOLVE_SPACE_zh.md).

| Artifact | Produced by | What evolves |
|---|---|---|
| [`artifacts/numerical_part1_dictionary.json`](artifacts/numerical_part1_dictionary.json) | [`scripts/stat_full.py`](scripts/stat_full.py) → [`scripts/num_part1.py`](scripts/num_part1.py) (PEN=3) | Numerical part 1: combination programs over 37 methods |
| [`artifacts/tl2_evolved_extraction_instructions.json`](artifacts/tl2_evolved_extraction_instructions.json) | [`scripts/tl2_evolve.py`](scripts/tl2_evolve.py) (uses [`tl2.py`](scripts/tl2.py)) | Retrieval extraction instructions (GPT mutator, gt_evidence F1 on Train only) |
| [`artifacts/fill_and_gate_choice.json`](artifacts/fill_and_gate_choice.json) | [`scripts/fill_variants.py`](scripts/fill_variants.py) → [`scripts/select_fill_gate.py`](scripts/select_fill_gate.py) | Numerical part-2 fill method + Decision repair margin (Train) |
| [`artifacts/numerical_part1_main.json`](artifacts/numerical_part1_main.json) | agent co-evolution, [`scripts/coevolution/`](scripts/coevolution/) | final part-1 program (evolved elite + 10% TimesFM term found by an agent) |
| [`artifacts/nrd4_final_teams.json`](artifacts/nrd4_final_teams.json) | [`scripts/nrd4.py`](scripts/nrd4.py) `--gens 15 --teams 32` (uses [`nrd_coevolve.py`](scripts/nrd_coevolve.py), [`nrd3.py`](scripts/nrd3.py), [`nrd_dict.py`](scripts/nrd_dict.py)) | three-agent co-evolution: calibrator, validator, Decision |

## Run (from repo root)

```bash
export PYTHONPATH=$PWD; mkdir -p .scratch/self_evolving; cp experiments/self_evolving/final_method/scripts/*.py .scratch/self_evolving/
python .scratch/self_evolving/nrd_precompute.py                          # task cache + 31 cached method forecasts
python .scratch/self_evolving/stat_full.py                               # + 93-method statistical portfolio -> nrd_cache_full.json
<timesfm-env>/python .scratch/self_evolving/timesfm_full.py; python .scratch/self_evolving/tsfm_full.py moirai_2_0,granite_ttm_r2
python .scratch/self_evolving/combined_full.py                           # + foundation models and combined policies -> nrd_cache_full2.json
<toto2-env>/python .scratch/self_evolving/toto_hindcast.py               # history-only Toto back-tests
A=experiments/self_evolving/final_method/artifacts; cp $A/tl2_evolved_extraction_instructions.json .scratch/self_evolving/tl2_best.json
python .scratch/self_evolving/extract_and_repair.py $A/tl2_evolved_extraction_instructions.json .scratch/self_evolving/repair.json
<toto2-env>/python .scratch/self_evolving/fill_variants.py               # Toto on repaired history + history-only validation, per fill method
# (to re-learn fill method + margin: select_fill_gate.py)
python .scratch/self_evolving/eval_full_pipeline.py --part1 $A/numerical_part1_main.json \
    --repair .scratch/self_evolving/repair.json --fill-variants .scratch/self_evolving/fill_variants.json \
    --teams $A/nrd4_final_teams.json          # previous main (nrd4 team for future correction)
python experiments/self_evolving/final_method/scripts/meta_harness/final_check.py dev,public_test main=$A/correction_function_main.py   # this method
```

- Re-evolve instead of using the saved artifacts: `num_part1.py`, `tl2_evolve.py`, and `nrd4.py --gens 15 --teams 32 --open test`.
- The Toto-anchored variant is evaluated by `eval_repair_plus_nrd4.py`.
- GPT calls go through the `codex` CLI (`gpt-6-luna`, `gpt-6-sol`).
