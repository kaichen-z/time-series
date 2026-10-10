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
| fitness: L4/L5 gain = BJT − jt (eps 1e-4, hidden tol 1e-9); L7 gain = (BJT − jt)/mean BJT (eps 1e-5, tol 1e-6); jt = sMAE+sRMSE (cap 5), PEN 0.5, STD_W 0.25 | identical per stage (official nMAE/nMSE / normalised MSE only as post-freeze reporting) | — |
| L4 final: champion of group A vs group B with the higher HIDDEN robust gain, tie → B | group B champion = best B run by (visible ↓, hidden ↓, id ↑); then A vs B by hidden, tie → B | — |
| failure rules: 2 h episode timeout, > 5 % failed units stops the stage, global cap stops all, no retries | identical (episodes; receipts per stage) | — |

L5 final = R2 champion; final modules are frozen with SHA-256 in `OUT/final/`.

## Isolation
Every agent episode runs inside bubblewrap (`sandbox_codex.py`): only system dirs, the codex install, a per-run CODEX_HOME,
`RUN/shared` + `RUN/queue` (rw), `RUN/results` (ro), the agent workspace and `submit.py` are mounted; own PID namespace,
clean environment. `RUN/private` (truth, folds, state), packs, official data repositories and the ledger are not visible.
`--leak-test` runs an active probe inside the same sandbox (must fail to read every forbidden path / other processes).
Real-codex check: one owner-approved smoke episode (2026-10-10 09:01Z, run by yyoraa on commit 09d35be) passed inside the sandbox
(1 turn, ledger reported, 1 submission scored, hidden check pass). Two earlier smoke calls failed with 0 model output because
the resolver dir was not mounted (fixed in 09d35be; see receipts/smoke1_failed_dns.json). The dry-run/leak/fake-e2e receipts
were produced before that one-line sandbox fix, so their recorded sandbox_codex.py SHA differs from the current file.

## Retrieval seed (pre-registered, pure function of the view)
Events with direction up/down and confidence >= 0.6 become corrections either (1) on the forecast steps covered by their
dates when `future_timestamps` exist (e.g. known holidays), or (2) as *recency*: events within 14 days (daily/weekly) /
45 days (monthly) of the latest pre-origin event date in the task's own documents move the first third of the horizon by
3 % x confidence. This expresses event recency/persistence; it does not assume knowledge of the forecast-window dates.
Coverage (receipt): TimesX 74/80 tasks, 134 corrections; Time-MMD 71/80 tasks, 230 corrections.

## Restart safety
Run dirs are never overwritten; a completed stage writes `OUT/receipts/<stage>.json` and is skipped with `--resume`;
an interrupted stage requires `--restart-stage`, which moves its run dirs to `OUT/aborted/<time>/` (ledger charges kept).

## Files
- `build_pack.py` — 80-task pack, primary + secondary folds, views (no labels / group_id), private truth, receipt.
- `orchestrate.py` — stages, run dirs, routing, `--dry-run` (no model call; codex replaced by a failing guard; ledger must stay empty).
- `evald.py` — evaluator daemon (one per run dir); `pipeline_runner.py` — isolated module execution; `submit.py` — agent submission.
- `run_agent.py` — one Codex episode (`gpt-5.6-sol`, `model_reasoning_effort="high"`), via the sol56 accounting shim.
- `rounds.py`, `sync_round.py`, `sol56_codex_shim.py` — unchanged from v3.3.2 (`sol56_code/new`).
- `seeds/` — L4 start modules; `prompts/TASK_TEMPLATE.md` — agent task text.
- `sandbox_codex.py` — bubblewrap isolation of each agent episode.
- `tests/fake_codex.py`, `tests/fake_codex_fail.py`, `tests/leak_probe.py` — scripted stand-ins (no model) for end-to-end,
  failure-stop and leak tests.

## Run
```bash
python3 build_pack.py --dataset timesx   --out packs/timesx   --repo REPO --official-root ROOT
python3 build_pack.py --dataset time_mmd --out packs/time_mmd --repo REPO --official-root ROOT
python3 orchestrate.py --dataset timesx --pack packs/timesx --out runs/timesx_leak --leak-test --forbid <official data dirs>
python3 orchestrate.py --dataset timesx   --pack packs/timesx   --out runs/timesx   --dry-run
python3 orchestrate.py --dataset time_mmd --pack packs/time_mmd --out runs/time_mmd --dry-run
# real runs (each up to 80 episodes / 400 submissions of gpt-5.6-sol high) -- only after explicit approval:
python3 orchestrate.py --dataset timesx   --pack packs/timesx   --out runs/timesx   --real-codex CODEX_BIN
python3 orchestrate.py --dataset time_mmd --pack packs/time_mmd --out runs/time_mmd --real-codex CODEX_BIN
```
Official Dev / sealed Test IDs, labels and event cards are never in a pack (asserted in `build_pack.py`).
