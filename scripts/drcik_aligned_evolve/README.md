# Dr-CiK-budget-aligned evolution for official TimesX / Time-MMD

This package transfers the **stage / episode / submission / acceptance budget** of the Dr-CiK pipeline
(runsheet v3.3.2: L4 + L5 + L7, `gpt-5.6-sol`, reasoning effort high) to a new three-module implementation
(Numerical `forecast.py`, Retrieval `retrieve.py`, Decision `adjust.py`) on the official TimesX and Time-MMD Train
partitions. Budget-equivalent; module semantics adapted. It does **not** claim to reproduce the Dr-CiK-specific
configs (tl2 extraction instructions, nrd4 calibration, validator) themselves.

## Mapping to runsheet v3.3.2

| v3.3.2 | Here | Episodes / max submissions |
|---|---|---|
| 80 Train tasks, 3 folds (0,1 visible, 2 hidden) | 80 official Train tasks, deterministic whole-group 3 folds 27/27/26, fold 2 hidden | — |
| L4 step-2 agents A1–A3 shared + B1–B3 independent, 4 episodes × 5 | A1/A2/A3 = numerical/retrieval/decision (shared run), B1/B2/B3 = same roles, independent runs; each agent submits only its own module | 24 / 120 |
| L5 meta-harness R1: S1 + C1–C3 + I1–I3, 4 × 5; routing; R2: 3 × 4 × 5 on new folds (prep_mh2) | same agents, Decision (correction function) only; routing per (freq, H) cell; R2 on the frozen secondary whole-group folds | 28 + 12 = 40 / 200 |
| L7 coevo_x: numerical-a, decision-a, numerical-b, decision-b; 2 agents × 2 episodes × 5 | same phases and counts; Retrieval frozen | 16 / 80 |
| acceptance: visible > frozen round base + eps AND hidden ≥ base − 1e-9, synchronous rounds | identical (`sync_round.py`, `rounds.py` from v3.3.2) | — |
| fitness: Toto-relative sMAE+sRMSE (cap 5), PEN 0.5, STD_W 0.25 | identical (official nMAE/nMSE / normalised MSE only as post-freeze reporting) | — |

L4 champion = run with the highest final visible fitness (tie → shared run A); L5 final = R2 champion; final modules are
frozen with SHA-256 in `OUT/final/`.

## Files
- `build_pack.py` — 80-task pack, primary + secondary folds, views (no labels / group_id), private truth, receipt.
- `orchestrate.py` — stages, run dirs, routing, `--dry-run` (no model call; codex replaced by a failing guard; ledger must stay empty).
- `evald.py` — evaluator daemon (one per run dir); `pipeline_runner.py` — isolated module execution; `submit.py` — agent submission.
- `run_agent.py` — one Codex episode (`gpt-5.6-sol`, `model_reasoning_effort="high"`), via the sol56 accounting shim.
- `rounds.py`, `sync_round.py`, `sol56_codex_shim.py` — unchanged from v3.3.2 (`sol56_code/new`).
- `seeds/` — L4 start modules; `prompts/TASK_TEMPLATE.md` — agent task text.
- `tests/fake_codex.py` — scripted stand-in for codex used only for end-to-end plumbing tests (no model).

## Run
```bash
python3 build_pack.py --dataset timesx   --out packs/timesx
python3 build_pack.py --dataset time_mmd --out packs/time_mmd
python3 orchestrate.py --dataset timesx   --pack packs/timesx   --out runs/timesx   --dry-run
python3 orchestrate.py --dataset time_mmd --pack packs/time_mmd --out runs/time_mmd --dry-run
# real runs (each up to 80 episodes / 400 submissions of gpt-5.6-sol high) -- only after explicit approval:
python3 orchestrate.py --dataset timesx   --pack packs/timesx   --out runs/timesx
python3 orchestrate.py --dataset time_mmd --pack packs/time_mmd --out runs/time_mmd
```
Official Dev / sealed Test IDs, labels and event cards are never in a pack (asserted in `build_pack.py`).
