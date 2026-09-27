# Final method: gated history repair + three-agent future correction (Dr-CiK)

The best configuration found so far. It has two complementary parts:

1. **History repair (Retrieval extraction evolved + Numerical repair + Decision gate).**
   - The Retrieval agent (GPT, with evolved instructions) reads all documents of a task. It extracts a timeline of past anomalies (measurement errors, one-off disturbances, recurring anomalies) and whether each will recur.
   - Numerical imputes the non-recurring anomalous history steps (same-phase median), and Toto re-forecasts from the repaired history.
   - Decision accepts a repair only if a **history-only back-test** confirms it: Toto forecasts the held-out tail of the history at least 10% better from the repaired history than from the raw one.
2. **Future correction (nrd4 three-agent co-evolution).** On top of that forecast, the evolved nrd4 team applies document corrections to future steps:
   - Numerical judgements: magnitude calibrator and per-cell Toto trust;
   - Retrieval: per-correction validator;
   - Decision: strength and trust gate.

## Results (exploratory: test99 had been opened before)

| test99 | sMAE | sRMSE | improved / worsened |
|---|---|---|---|
| Toto | 0.3814 | 0.5903 | — |
| Hand-designed pass-combiner | 0.3564 (+6.5%) | 0.5310 (+10.1%) | 30 / 1 |
| nrd4 alone (3-seed mean) | 0.3681 (+3.5%) | 0.5539 (+6.2%) | 29–31 / 1–2 |
| **This method (3-seed mean)** | **0.3291 (+13.7%)** | **0.5208 (+11.8%)** | **32–34 / 2–3** |

- Seeds: sMAE +14.6 / +12.9 / +13.7%, sRMSE +12.5 / +11.0 / +11.9%.
- Dev (20 tasks, 2–3 repaired): joint −0.10%. That is neutral, and it fails the strict "neither metric worse" gate by sMAE −0.3%.
- The gate margin is insensitive: any margin from 0 to 0.2 gives identical test results.

## How the pieces were evolved

- **Extraction instructions** (`artifacts/tl2_evolved_extraction_instructions.json`): evolved for 4 generations on a 30-task **train** minibatch.
  - Fitness = F1 against `annotations.gt_evidence`, which is used only on train.
  - GPT-6-sol rewrites the instructions from concrete missed and false-positive evidence.
  - F1: 0.054 → 0.464; full train recall 5% → 48%, precision 9% → 67%.
- **nrd4 team** (`artifacts/nrd4_final_teams.json`, 3 seeds): 15 generations of joint team selection on stratified train folds.
  - Fitness = mean fold gain with a regression penalty, minus 0.25 × fold std.
  - Full-round rollback.
  - Generation 0 = exact Toto.

## Run (from repo root)

```bash
export PYTHONPATH=$PWD; mkdir -p .scratch/self_evolving; cp experiments/self_evolving/final_method/scripts/*.py .scratch/self_evolving/
python .scratch/self_evolving/nrd_precompute.py                                    # task cache + Toto/dictionary forecasts
<toto2-env>/python .scratch/self_evolving/toto_hindcast.py                         # history-only Toto back-tests
python .scratch/self_evolving/extract_and_repair.py \
    experiments/self_evolving/final_method/artifacts/tl2_evolved_extraction_instructions.json .scratch/self_evolving/repair.json
<toto2-env>/python .scratch/self_evolving/tl2_toto.py .scratch/self_evolving/repair.json .scratch/self_evolving/repair_fc.json
<toto2-env>/python .scratch/self_evolving/tl3_validate.py .scratch/self_evolving/repair.json .scratch/self_evolving/repair_val.json
python .scratch/self_evolving/eval_repair_plus_nrd4.py --repair .scratch/self_evolving/repair.json \
    --forecasts .scratch/self_evolving/repair_fc.json --validation .scratch/self_evolving/repair_val.json \
    --teams experiments/self_evolving/final_method/artifacts/nrd4_final_teams.json
```

- To re-evolve instead of using the saved artifacts, run `../scripts/tl2_evolve.py` (extraction) and `nrd4.py --gens 15 --teams 32 --open dev` (team).
- GPT calls go through the `codex` CLI (models `gpt-6-luna`, `gpt-6-sol`).
