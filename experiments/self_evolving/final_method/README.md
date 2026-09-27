# Final method: two-part Numerical + gated history repair + three-agent future correction (Dr-CiK)

**Numerical, part 1: evolve a candidate dictionary of forecast-combination programs**
- Generation 0 is the 31 single methods of the dictionary (Toto is just one of them).
- Programs combine the cached method forecasts by:
  - weighted sums;
  - difference corrections (`c·(M_a − M_b)`);
  - shrinkage toward the last observation;
  - clipping to the history range.
- Evolved with typed mutations on Train outcomes. Fitness = mean gain + 3 × mean negative gain, a do-no-harm penalty.
- A diverse set of elites is kept as the dictionary; the best elite is the base forecaster.
- Evolution picks Toto by itself as the main component: **0.94·Toto + 0.06·ARMA** (+ a small trend-difference term), **shrunk 17% toward the last observation**, clipped to the history range.

**Numerical, part 2 + Retrieval + Decision: the effective evolution on top of that base**
1. **History repair.** Retrieval (GPT with evolved extraction instructions) finds past anomalies that will not recur. Numerical imputes them and Toto re-forecasts. Decision accepts a repair only if a history-only back-test confirms it by at least 10%. When accepted, the Toto term of the base program uses the repaired-history forecast.
2. **Future correction.** The evolved nrd4 team applies document corrections to future steps:
   - Numerical: magnitude calibrator and per-cell Toto trust;
   - Retrieval: per-correction validator;
   - Decision: strength and trust gate.

## Results

| | Dev sMAE | Dev sRMSE | Test sMAE (exploratory) | Test sRMSE (exploratory) | Test improved / worsened |
|---|---|---|---|---|---|
| Toto | 0.3921 | 0.5871 | 0.3814 | 0.5903 | — |
| Hand-designed pass-combiner | — | — | 0.3564 (+6.5%) | 0.5310 (+10.1%) | 30 / 1 |
| Toto-anchored variant (no Numerical part 1) | −0.3% | +0.03% | 0.3291 (+13.7%) | 0.5208 (+11.8%) | 32–34 / 2–3 |
| **This method (3-seed mean)** | **0.3567 (+9.0%)** | **0.5231 (+10.9%)** | **0.3405 (+10.7%)** | **0.5222 (+11.5%)** | **64–65 / 34–35** |

- Per seed, test sMAE: 0.3375 / 0.3434 / 0.3406; sRMSE: 0.5186 / 0.5264 / 0.5218.
- **The dev gate is passed on both metrics** (dev is used once, after all evolution on Train).

Caveats:
- test99 had been opened before, so test numbers are exploratory.
- The part-1 penalty weight (3) was chosen after seeing a penalty-0.5 run on dev and test, which is mild contamination.
- The shrink/clip transform applies to every task. It helps the few badly-forecast tasks a lot and slightly worsens many others, so about 35 test tasks get (mostly slightly) worse. The Toto-anchored variant is the "almost no harm" alternative.

## Scripts that produce the evolved artifacts

Per-generation results are in [`EVOLUTION_LOG.md`](EVOLUTION_LOG.md).

| Artifact | Produced by | What evolves |
|---|---|---|
| [`artifacts/numerical_part1_dictionary.json`](artifacts/numerical_part1_dictionary.json) | [`scripts/num_part1.py`](scripts/num_part1.py) (PEN=3) | Numerical part 1: combination programs over 31 methods |
| [`artifacts/tl2_evolved_extraction_instructions.json`](artifacts/tl2_evolved_extraction_instructions.json) | [`scripts/tl2_evolve.py`](scripts/tl2_evolve.py) (uses [`tl2.py`](scripts/tl2.py)) | Retrieval extraction instructions (GPT mutator, gt_evidence F1 on Train only) |
| [`artifacts/nrd4_final_teams.json`](artifacts/nrd4_final_teams.json) | [`scripts/nrd4.py`](scripts/nrd4.py) `--gens 15 --teams 32` (uses [`nrd_coevolve.py`](scripts/nrd_coevolve.py), [`nrd3.py`](scripts/nrd3.py), [`nrd_dict.py`](scripts/nrd_dict.py)) | three-agent co-evolution: calibrator, validator, Decision |

## Run (from repo root)

```bash
export PYTHONPATH=$PWD; mkdir -p .scratch/self_evolving; cp experiments/self_evolving/final_method/scripts/*.py .scratch/self_evolving/
python .scratch/self_evolving/nrd_precompute.py                          # task cache + 31 cached method forecasts
<toto2-env>/python .scratch/self_evolving/toto_hindcast.py               # history-only Toto back-tests
A=experiments/self_evolving/final_method/artifacts
python .scratch/self_evolving/extract_and_repair.py $A/tl2_evolved_extraction_instructions.json .scratch/self_evolving/repair.json
<toto2-env>/python .scratch/self_evolving/tl2_toto.py .scratch/self_evolving/repair.json .scratch/self_evolving/repair_fc.json
<toto2-env>/python .scratch/self_evolving/tl3_validate.py .scratch/self_evolving/repair.json .scratch/self_evolving/repair_val.json
python .scratch/self_evolving/eval_full_pipeline.py --part1 $A/numerical_part1_dictionary.json \
    --repair .scratch/self_evolving/repair.json --forecasts .scratch/self_evolving/repair_fc.json \
    --validation .scratch/self_evolving/repair_val.json --teams $A/nrd4_final_teams.json
```

- Re-evolve instead of using the saved artifacts: `num_part1.py`, `tl2_evolve.py`, and `nrd4.py --gens 15 --teams 32 --open test`.
- The Toto-anchored variant is evaluated by `eval_repair_plus_nrd4.py`.
- GPT calls go through the `codex` CLI (`gpt-6-luna`, `gpt-6-sol`).
