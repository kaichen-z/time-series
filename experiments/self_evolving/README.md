# Self-evolving Numerical / Retrieval / Decision on Dr-CiK (and Time-MMD)

Experiments from 2026-09-26/27. Base forecaster: Toto-2.0-22m, frozen. Three agents:
- **Numerical** sees only the numeric history.
- **Retrieval** sees only the documents.
- **Decision** chooses among existing candidates or falls back to Toto.

Generation 0 is always exact Toto.

Full write-ups:
- `reports/REPORT_2026-09-27.md` (English) and `reports/REPORT_2026-09-27_zh.md` (Chinese);
- talk scripts in `reports/TALK_*`.

## Final method (exploratory test result)

**Gated history repair (tl2 + tl3) + nrd4 future corrections.**

| test99 (Toto sMAE 0.3814 / sRMSE 0.5903) | sMAE | sRMSE | improved / worsened |
|---|---|---|---|
| Hand-designed pass-combiner | 0.3564 | 0.5310 | 30 / 1 |
| nrd4 | 0.3681 | 0.5539 | 29–31 / 1–2 |
| **tl2 gated repair + nrd4** | **0.3291** | **0.5208** | **32–34 / 2–3** |

Caveats:
- test99 had been opened several times before this, so the result is exploratory.
- The combination is neutral on dev (−0.1%).
- `annotations.gt_evidence` is used only on train, as Retrieval's evolution signal.

## Pipeline and files (`scripts/`)

Run everything from the repo root with `PYTHONPATH=$PWD`. Caches go to `.scratch/self_evolving/` and are **not** committed.

| Step | Script | What it does |
|---|---|---|
| data | `nrd_precompute.py` | caches Dr-CiK tasks, 31 cached dictionary forecasts, and old CorDP cards → `nrd_cache.json` |
| Toto hindcasts | `toto_hindcast.py` (`_tmmd`) | history-only Toto back-tests (toto2 env), used by Numerical |
| engine v1/v2 | `nrd_coevolve.py` | team co-evolution core: fitness, stratified folds, rollback |
| dictionary evolution | `nrd_dict.py` | typed program grammar, history-only hindcast fitness, MAP-Elites archive |
| nrd3 | `nrd3.py` (`nrd2.py`) | dictionary + blending + per-correction validator |
| **nrd4** | `nrd4.py` | Numerical judgements (calibrator, Toto trust) + Retrieval validator ES + 2-parameter Decision |
| nrd5 | `nrd5.py`, `nrd5_extract.py` | GPT-evolved extraction program (codex CLI) |
| nrd6 | `nrd6.py` | ridge residual corrector + conformal gate |
| nrd7 | `nrd7.py` | evidence-verified, multi-model consensus extraction |
| nrd8 | `nrd8.py` | evolvable window/direction transform |
| nrd9 | `nrd9_extract.py`, `build_cache_v9.py` | precise-time re-extraction of cards |
| nrd10 | `nrd10.py` | union of cards + agreement features |
| nrd11 | `nrd11.py` | two-stage: dictionary evolution → judgement features → co-evolution |
| nrd12 | `nrd12.py` | GEPA-style instance Pareto selection + islands + ensemble |
| tl (v1) | `tl_extract.py`, `tl_numerical.py`, `tl_toto.py`, `tl_eval.py` | state-timeline extraction and history repair |
| **tl2** | `tl2.py`, `tl2_evolve.py`, `tl2_toto.py`, `tl2_eval.py` | schema v2; extraction instructions evolved against gt_evidence F1 (GPT mutator); repair + Toto re-forecast |
| **tl3 gate** | `tl3_validate.py`, `eval_gated_repair.py` | history-only validation of each repair: accept iff Toto back-tests the held-out tail better from the repaired history |
| **final** | `eval_repair_plus_nrd4.py` | gated repair feeds Toto's input; nrd4 team corrects the future |

`results/`:
- small JSON summaries: final teams and per-seed metrics;
- `tl2_best.json`: evolved extraction instructions and their F1 curve;
- `tl4_instr.json`: the same instructions plus a natural-cycle constraint.

## Reproduce the final method (sketch)

```bash
python scripts/nrd_precompute.py
toto-env/python scripts/toto_hindcast.py
python scripts/nrd4.py --gens 15 --teams 32 --open test --out nrd4_final.json   # future corrections
python scripts/tl2_evolve.py                                                   # evolve extraction, repair train/dev
toto-env/python scripts/tl2_toto.py tl2_repair_evolved.json tl2_fc_evolved.json
toto-env/python scripts/tl3_validate.py tl2_repair_evolved.json tl3_val.json
python scripts/eval_repair_plus_nrd4.py
```

Paths inside the scripts point to `.scratch/self_evolving/`; adjust as needed.
