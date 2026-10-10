# Dr-CiK-budget-aligned evolution for official TimesX / Time-MMD — protocol v6 (full official Train)

Protocol v6 corrects the final candidate-pool omission in v5. The host injects six immutable candidates from
`fixed_candidates/manifest.json`: raw Toto, TimesFM, Moirai, Chronos-Bolt, and the equal-weight mean and median of
those four forecasts. They use no retrieval correction and a pass-through decision module, so they remain genuine
numerical baselines. `final_select.py` scores them together with every frozen evolved candidate in the same single F1
opening and records the manifest plus every fixed-module SHA-256 in `LOCK.json`. Agents cannot write this directory.

This package does not authorize a run. Existing v5 locks remain immutable, and official Test stays separately gated.

> **Protocol disclosure (2026-10-10): v6 is a second Dev selection, not a one-shot Dev experiment.**
> TimesX Dev was already opened by v5, and v6 was designed after that result exposed a candidate-pool omission. Per the
> project decision, v6 still uses the same Dev once for its final frozen-candidate selection. It must therefore be
> reported as a second development selection; only the still-sealed official Test can provide the generalisation
> confirmation. Stage 0 below uses Train labels only and does not erase or weaken this disclosure.

Transfers the **stage / episode / submission budget** of the Dr-CiK pipeline (runsheet v3.3.2: L4 + L5 + L7,
`gpt-5.6-sol`, reasoning effort high) to a three-module implementation (Numerical `forecast.py`, Retrieval
`retrieve.py`, Decision `adjust.py`) on the **full** official TimesX and Time-MMD Train partitions. Budget-equivalent;
module semantics adapted; it does **not** reproduce Dr-CiK-specific configs (tl2 instructions, nrd4 calibration, validator).

History: the first full run (2026-10-10) memorised per-task visible answers (archived `INVALID_OVERFIT`); v3/v4 fixed the
protocol on an 80-task subset (superseded). v5 kept the protections and used ALL official Train tasks; v6 additionally
makes the raw foundation-model and simple-ensemble baselines mandatory final candidates.

## Data boundary (formal mode: `all_train`, packs `packs/<ds>_all_train/`)
- **Evolve (F0)** on ALL Train: TimesX **2,064**, Time-MMD **45,948** tasks (agents; opaque handles, aggregate feedback).
- **Select (F1)** once on Dev: TimesX **550**, Time-MMD **5,780** (host-only, after every stage is frozen; fail closed).
- **Final check** on official Test (TimesX 1,695 ID + 211 OOD; Time-MMD 11,936): a separately approved step, NOT run by
  this package; Test labels are never read (sealed Test ids are only used to assert none entered a pack).
- Splits: Time-MMD = official MM-TSFlib chronological 70 / 10 / 20 per series (z-score fitted on the Train segment only).
  TimesX = official Test; Train/Dev is OUR frozen internal date split of the 2,723 official post-training samples
  (Dev = target starts on/after 2024-09-01), not an official Dev.
- **Time boundary**: every Train window whose prediction target ends on/after 2024-09-01 is purged (TimesX: 109 of
  2,173; max horizon 12), so no Train target reaches the Dev period. Time-MMD needs none (Train targets end before Dev).
- **Information time**: every fact / event card an agent sees is dated strictly before the forecast origin; later or
  undated ones are removed (Time-MMD: 4,548 fact blocks, 3,339 cards, 1,135 emptied documents; TimesX: 2,061 cards;
  scenario text = official target-period header + the kept dated events). TimesX holiday information is kept as a
  deterministic calendar covariate (`calendar: true`). Receipt: `time_boundary`, `information_time_filter`.
- Ablation option (not used): `build_pack.py --split train_2to1` (F0/F1 by whole domain inside Train, Dev final).
- One private truth file per part (`private/eval_{F0_feedback,F1_selection[,FINAL_dev]}.json`).
- Views carry no labels and a generic target description; series/group names are in `private/forbidden_terms.json`.
- **Storage** (`viewstore.py`): `shared/store/forecasts.f32` = all frozen forecasts as little-endian float32;
  `views_meta.json` = views + `{method: [offset, length]}`; `documents.json` = documents (deduplicated). The receipt
  records dtype/endianness/shape, 64 MB chunk SHA-256s, all file SHA-256s and the decode error (relative ≤ 6e-8).

## Stage 0: frozen Train-only reference and seed

Before evolution, `reference/select_seed_cv.py` searches 4,004 numerical configurations: simplex weights in steps of
0.1 over Toto, TimesFM, Moirai, Chronos-Bolt and a cyclic seasonal-naive forecast, crossed with shrink-to-last values
`{0, 0.1, 0.2, 0.3}`. It uses only F0/official Train labels. Forecast origins are divided into five chronological
blocks; block 0 is not scored and the remaining **K=4 test blocks** are aggregated fold-macro (each nonempty block has
equal weight, not each task). TimesX weekly has only three nonempty scored blocks. Time-MMD first draws a deterministic
seed-0 sample of at most 1,500 tasks per `(frequency, horizon)`, records every selected task id, and then creates the
chronological blocks. Exact ties retain the earlier fixed grid entry.

The selected numerical program is frozen twice with identical forecast semantics: as the immutable host-side scoring
reference (`reference/reference_forecast.py`) and as the agents' numerical seed (`seeds/seed_forecast.py`). Evolution
keeps the Dr-CiK robust objective but measures gain relative to this strong frozen reference rather than Toto:
`mean(reference_jt - program_jt) + 0.5 * mean(min(0, reference_jt - program_jt))`. The reference SHA is stored in every
run's `stage.json`/private evaluation record and in the final lock; a reference change during final selection fails
closed. CV evidence, fold sizes, sampled ids, selected weights and evidence hashes are frozen under `reference/`.

## What agents can see
- A run dir holds only F0, as its own store `RUN/shared/store` keyed by **per-run opaque row handles** (`r_…`, fresh for
  every run dir and every host-side scoring call, shuffled; forecasts re-laid in that order); no task/document/group/series
  identity (document ids are fresh `d_…`); the handle→task map stays in `private/` (not mounted). Agents read it with the
  bundled `RUN/shared/store/viewstore.py`.
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
| L4 | A1/A2/A3 = numerical/retrieval/decision (shared run); B1/B2/B3 same roles, independent runs; 4 rounds | 24 / 120 | reference_jt − jt, 1e-4 |
| L5 | R1: S1 + C1–C3 + I1–I3 (decision only), 4 rounds; routing per (freq, H) cell on F0; R2: R1–R3, 4 rounds | 40 / 200 | reference_jt − jt, 1e-4 |
| L7 | numerical-a, decision-a, numerical-b, decision-b; 2 agents x 2 rounds per phase; retrieval frozen | 16 / 80 | (reference_jt − jt)/mean reference_jt, 1e-5 |

Acceptance per synchronous round: F0 fitness > frozen round champion + eps (best eligible wins at round close). L4 stage
seed = A or best B by F0, tie → B. Deviations from v3.3.2: no hidden fold during evolution (v3.3.2 gated on it) and no
L5-R2 re-split (it would expose F1 tasks). `rounds.py`, `sync_round.py`, `sol56_codex_shim.py` are unchanged from
v3.3.2; their "hidden" slot is a constant 0.

## Final selection (`final_select.py`)
After L7, every frozen candidate (seed, 4 L4 runs, 5 L5-R1 runs, routed, L5-R2, L7) is run once on a freshly anonymised
F1 store. Candidates with **any** F1 runtime error are not selectable; if none is error-free the selection **fails
closed** (no LOCK, `SELECTION_FAILED.json`, final check never opened). Otherwise the candidate with the **lowest mean
joint error** is selected; exact ties retain the earlier candidate in the frozen order (the six host-pinned baselines
come first). `LOCK.json` records mean joint error, robust/mean gain relative to the reference, better/worse counts and
runtime errors for every candidate. Then `train_2to1`: official Dev is scored once (`final/FINAL_dev.json`);
`all_train`: `final/FINAL_PENDING.json` (official Test needs a separate approval). `final/access_log.json` counts every
label-file opening (expected F1 = 1; FINAL = 1 after lock, or 0 for `all_train`).

## Residual risk (disclosed)
History values and timestamps are content: within F0 an agent can still recognise rows by them, so F0 can be overfit.
The safeguard is the one-time host-only selection on unseen tasks (F1) and the single locked final check.
In `train_2to1` the units are very uneven (Time-MMD Environment = 65 % of Train), so F0/F1 task shares are not 2:1.

## Files
`build_pack.py` (packs), `viewstore.py` (float32 store), `anon.py` (opaque handles), `orchestrate.py` (stages, `--dry-run`, `--leak-test`, real runs),
`evald.py` (evaluator daemon), `pipeline_runner.py`, `submit.py`, `run_agent.py`, `sandbox_codex.py`, `final_select.py`,
`reference.py` and `reference/` (Stage-0 selection, frozen reference, manifest and CV evidence),
`seeds/`, `prompts/TASK_TEMPLATE.md`, `rounds.py` / `sync_round.py` / `sol56_codex_shim.py` (v3.3.2),
`tests/` (fake agents, leak probe, anti-memorisation, closure/storage proof, test sub-pack builder, group-rule and fail-closed tests, `cheat_samples/` =
the 8 champions of the invalid run), `fixtures/` (frozen task→split indices of the official handoff), `packs/` (frozen).

## Verify (no model calls)
```bash
python -I -S -B run_tests.py --split all_train   # fresh 0700 work dir outside the package; package bytes hashed before/after
```
Closure/storage proof, leak test and run-dir proof run on the full pack; the fake-agent end-to-end, anti-memorisation,
group-rule and fail-closed tests run on a deterministic 48-task-per-part sub-pack (real runs refuse sub-packs).

## Real runs (only after explicit approval)
```bash
python3 orchestrate.py --dataset timesx   --pack packs/timesx_all_train   --out runs/timesx   --real-codex CODEX_BIN
python3 orchestrate.py --dataset time_mmd --pack packs/time_mmd_all_train --out runs/time_mmd --real-codex CODEX_BIN
```
Disk: every run dir holds its own anonymised F0 store (Time-MMD ≈ 0.35 GB each, ~17 per dataset run).
