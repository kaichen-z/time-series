# Dr-CiK-budget-aligned evolution for official TimesX / Time-MMD — protocol v3 (anti-overfit)

Transfers the **stage / episode / submission budget** of the Dr-CiK pipeline (runsheet v3.3.2: L4 + L5 + L7,
`gpt-5.6-sol`, reasoning effort high) to a three-module implementation (Numerical `forecast.py`, Retrieval
`retrieve.py`, Decision `adjust.py`) on the official TimesX and Time-MMD Train partitions. Budget-equivalent; module
semantics adapted; it does **not** reproduce Dr-CiK-specific configs (tl2 instructions, nrd4 calibration, validator).

History: the first full run (2026-10-10) memorised the per-task visible answers that the v3.3.2 protocol showed to agents
(archived as `INVALID_OVERFIT`). Protocol v3 replaces the v3.3.2 visible/hidden fold protocol as follows.

## Data boundary (per dataset, `packs/<ds>/`)
- 80 official Train tasks, deterministic **whole-group** 3-fold split (27/27/26), no group in two folds:
  - **F0 feedback** — the only fold that ever enters an agent run dir;
  - **F1 selection** — host-only; used **once**, after every stage is frozen, to choose the final program;
  - **F2 final test** — host-only; opened **once**, only after the final program is locked.
- One private truth file per fold (`private/eval_F{0,1,2}_*.json`); official Dev and sealed Test never enter a pack.
- Views carry no labels and a generic target description; series/group names are in `private/forbidden_terms.json`.

## What agents can see
- A run dir holds only F0, keyed by **per-run opaque row handles** (`r_…`, fresh for every run dir and every host-side
  scoring call, shuffled); views contain no task/document/group/series identity (document ids are fresh `d_…`);
  the handle→task map stays in `private/` (not mounted). The handles themselves are temporary keys of the view file.
- Feedback per submission: one aggregate F0 fitness quantised to 1e-3, the runtime-error count, eligibility, round,
  remaining budget. No per-task values, task ids/handles, rankings, better/worse counts or pass/fail of other folds.
- `shared/traces/visible_summary.json`: means over (freq, H) cells with >= 3 tasks and overall (3 decimals).
- Each episode runs inside bubblewrap (`sandbox_codex.py`): only system dirs, the codex install, a per-run CODEX_HOME,
  `RUN/shared` + `RUN/queue` (rw), `RUN/results` (ro), the workspace and `submit.py`; own PID namespace; clean env.

## Submission guards (`evald.py`, agent submissions only)
Rejected if: > 20 KB; > 300 numeric constants or > 3000 digit characters; > 2000 characters of string data (docstrings
excluded); banned modules/builtins (encoders, compression, hashing, file/OS access, exec/eval/compile/open/hash/getattr…);
series/group names in string literals; or base/final forecast within 1e-3 x Toto error on >= max(3, 10 %) F0 tasks.

## Stages (budget identical to v3.3.2)
| Stage | Agents / rounds | Episodes / max submissions | Gain, eps |
|---|---|---|---|
| L4 | A1/A2/A3 = numerical/retrieval/decision (shared run); B1/B2/B3 same roles, independent runs; 4 rounds | 24 / 120 | BJT − jt, 1e-4 |
| L5 | R1: S1 + C1–C3 + I1–I3 (decision only), 4 rounds; routing per (freq, H) cell on F0; R2: R1–R3, 4 rounds | 40 / 200 | BJT − jt, 1e-4 |
| L7 | numerical-a, decision-a, numerical-b, decision-b; 2 agents x 2 rounds per phase; retrieval frozen | 16 / 80 | (BJT − jt)/mean BJT, 1e-5 |

Acceptance per synchronous round: F0 fitness > frozen round champion + eps (best eligible wins at round close). L4 stage
seed = A or best B by F0, tie → B. Deviations from v3.3.2: no hidden fold during evolution (v3.3.2 gated on it) and no
L5-R2 re-split (it would expose F1/F2 groups). `rounds.py`, `sync_round.py`, `sol56_codex_shim.py` are unchanged from
v3.3.2; their "hidden" slot is a constant 0 in v3.

## Final selection (`final_select.py`)
After L7, every frozen candidate (seed, 4 L4 runs, 5 L5-R1 runs, routed, L5-R2, L7) is run once on freshly anonymised
F1 views. Candidates with **any** F1 runtime error are not selectable; if none is error-free the selection **fails
closed** (no LOCK, `SELECTION_FAILED.json`, F2 never opened). Otherwise the best F1 robust gain (ties → earlier, seed
first) is locked in `final/LOCK.json`, then F2 is scored once (`final/F2_final_test.json`; `final/access_log.json`).

## Residual risk (disclosed)
History values and timestamps are content: within F0 an agent can still recognise rows by them, so F0 can be overfit.
The scientific safeguard is the one-time host-only selection on unseen groups (F1) and the single locked test on F2.

## Files
`build_pack.py` (packs), `anon.py` (opaque handles), `orchestrate.py` (stages, `--dry-run`, `--leak-test`, real runs),
`evald.py` (evaluator daemon), `pipeline_runner.py`, `submit.py`, `run_agent.py`, `sandbox_codex.py`, `final_select.py`,
`seeds/`, `prompts/TASK_TEMPLATE.md`, `rounds.py` / `sync_round.py` / `sol56_codex_shim.py` (v3.3.2),
`tests/` (fake agents, leak probe, anti-memorisation, closure proof, group-rule and fail-closed tests, `cheat_samples/` =
the 8 champions of the invalid run), `fixtures/` (frozen task→split indices of the official handoff), `packs/` (frozen).

## Verify (no model calls)
```bash
python -I -S -B run_tests.py            # fresh 0700 work dir outside the package; package bytes hashed before/after
```
## Real runs (only after explicit approval)
```bash
python3 orchestrate.py --dataset timesx   --pack packs/timesx   --out runs/timesx   --real-codex CODEX_BIN
python3 orchestrate.py --dataset time_mmd --pack packs/time_mmd --out runs/time_mmd --real-codex CODEX_BIN
```
