# Handoff — time-series repo

Written 2026-08-24, updated same session after the metric rewrite, cap removal, and test debloat.
Snapshot of state so another session can pick up cleanly.

## Repo layout

- Main working dir: `/home/ik832/time-series`, on branch `feature/compositional-skill-evolution`
  (base commit `e89477a`; substantial uncommitted work on top, see below).
- Second worktree: `/tmp/ts-main-v007`, on branch `main` (commit `4bc9b87`). Used to run Yiyun's
  dictionary-curation experiment (v007) in isolation from the feature branch.
- Remotes exist (`origin/main`, `origin/khoutaibi`). **Nothing has been pushed. Nothing has been
  committed either — "lets not push yet" is still in force, and no commit was requested this
  session.**

## What the work is about

Two parallel evolution pipelines in `numerical_agent/`:

1. **`numerical_agent/evolution/`** — compositional skill evolution (the feature branch's focus).
   Evolves a Python module of forecasting methods across generations, using measured results.
2. **`numerical_agent/curation/`** — dictionary curation (Yiyun's pipeline). Bootstraps 111
   statistical methods from a catalog via LLM codegen, then curates/evolves the dictionary.

A third, older pipeline, **`evolving_loop/`**, is also live (console script `evolving-agent`,
actively edited) and was touched this session because its scoring shared `common/metrics.py`
with the two above. **`drcik_agent/`** (triad/codex/paper-agent baselines) is untouched except
that its test coverage for triad/codex/paper/backbones was dropped — see below.

## Metric definition (the main change this session) — now exactly the Dr-CiK paper's formula

`common/metrics.py` implements the paper's scaled metrics with **no deviation and no cap**:

```
a = ((1/T) * sum(|y_t|))^-1        # mean absolute truth over the forecast horizon
sMAE  = a * (1/T) * sum(|forecast_t - truth_t|)
sRMSE = a * sqrt((1/T) * sum((forecast_t - truth_t)^2))
```

- `mean_absolute_truth(y_true)` is the denominator — the mean absolute value of the **truth over the
  horizon being scored**, not the history, and **no seasonal period is involved anywhere**. This
  replaced an earlier (wrong) implementation this session that scaled by a seasonal-naive error
  computed from history; caught and corrected mid-session.
- `scaled_mae(y_true, y_pred)` / `scaled_rmse(y_true, y_pred)` / `change_smae(y_true, y_pred,
  last_observed)` — no `cap` parameter, no `SCALED_CAP` constant anywhere in the codebase. A
  catastrophic forecast keeps its true magnitude (e.g. sMAE can be 1000.0), which is itself the
  evidence for deleting a method.
- `smape`, `mae`, `rmse`, `change_mae` are **not** metrics to score or rank by — `mae`/`rmse`/
  `change_mae` remain only as internal building blocks; `smape` and `score_forecast` were
  **deleted entirely**, including from `evolving_loop` (baseline.py, evaluation.py,
  coding_agent/evolution.py) which used to depend on them.
- **Failed hindcast folds now score `math.inf`**, not a capped penalty — a crashed method is
  disqualified outright rather than averaged in as a plausible number. See
  `evolving_loop/coding_agent/evolution.py` (`_validate`, `_validate_tsfm`,
  `_repeat_last_hindcast`) and `numerical_agent/curation/scaled.py`'s `ScaledMetric.__call__`. A
  `_improvement()` helper in `evolution.py:92` guards `inf - inf` → `0.0` rather than `NaN` when
  every candidate in a generation fails.
- `evolving_loop/co_evolution.py`'s three reward normalizations (`coding`, `decision`, `forecast`)
  used the old cap as a divisor to bound rewards into `[0, 1]`. That's reward-shaping, not the
  metric, so it now has its own local `REWARD_ERROR_CEILING = 5.0` constant, decoupled from the
  metric — reward values are unchanged, just no longer coupled to `common/metrics.py`.
- `numerical_agent/evolution/execution.py`'s `Outcome`/`MethodReport` no longer carry
  `smae_uncapped`/`srmse_uncapped`/`mean_smae_uncapped`/`mean_srmse_uncapped` — those existed
  only to distinguish from the (now-deleted) cap, so they were redundant duplicates of
  `smae`/`srmse` and were removed, along with the matching prompt text in
  `numerical_agent/evolution/prompts.py`.
- `numerical_agent/curation/scaled.py` lost its entire per-task scale-lookup apparatus
  (`scales_by_item`, `needs_scale`, the `_score()` context-passing shim in
  `numerical_agent/curation/__init__.py`) — none of it is needed once the scale comes from the
  truth being scored rather than from history/frequency/period.

**Open, unresolved: the status thresholds are still on the old (wrong) scale.**
`accepted_max_error` 1.0 / `specialized_max_error` 2.0 in `numerical_agent/config.py` were chosen
when 1.0 meant "ties seasonal-naive" under the old (incorrect) definition. Under the current
metric, sMAE 1.0 means the error is as large as the series itself, so these thresholds now accept
almost everything and need recalibrating empirically — probably by measuring a generation and
picking real values, not guessing.

**Also open, never resolved:** `drcik_agent/metrics.py`'s `development_scale`/`smae_proxy`/
`srmse_proxy`/`scrps_proxy` still use the old seasonal-naive-scale-with-cap definition. Left alone
deliberately — it's a separate pipeline, self-described as a proxy for the maintainers' private
scorer — but it now disagrees with `common/metrics.py`. Flagged to the user, not actioned.

## Tests debloated this session

`tests/` went from **65 files / ~11.9k lines / 739 tests** to **30 files / ~10.6k lines / 710
tests**. Current baseline: `python3 -m pytest tests/ -q` → **710 passed, 2 failed**, both
pre-existing on clean HEAD and unrelated to any of this session's work:
`test_fresh30_launcher_dry_run_renders_four_methods_for_one_smoke_task` (now in
`tests/test_scripts.py`) and `test_prompt_advertises_exactly_the_sandbox_allow_list` (in
`tests/test_numerical_codegen.py`, sandbox allow-list gained `torch`, prompt text not yet
updated).

- **Deleted outright** (confirmed dead to the user despite being console-script-reachable and
  README-documented — source code untouched, only tests dropped): `test_triad.py`,
  `test_codex_agents.py`, `test_codex_baseline.py`, `test_codex_triad.py`, `test_paper_agents.py`,
  `test_chronos_backbone.py`, `test_timesfm_backbone.py`. ~27 tests lost, the only real coverage
  loss from the debloat.
- **Consolidated** (13 merges + 1 rename, zero coverage loss — every surviving test is the same
  test that existed before, just relocated): `test_common_llm.py`, `test_common_metrics.py`,
  `test_common_sandbox.py` (renamed from `test_evolving_agent_sandbox.py`),
  `test_evolution_core.py`, `test_evolving_agent.py`, `test_evolving_data.py`,
  `test_evolving_harness.py`, `test_collection.py`, `test_collection_catalog.py`,
  `test_evolution_module.py` (absorbed seed), `test_evolution_execution.py` (absorbed seeding),
  `test_numerical_dictionary_contracts.py` (absorbed statistical_base),
  `test_numerical_cli.py` (method_dataset_cli + dictionary_curation_script + numerical_agent_cli),
  `test_scripts.py`. Several merges hit genuine name collisions between same-named helpers with
  different bodies (three different `_task()` fixtures, two different `method(...)` builders, two
  different `FIXTURES` path constants) that had to be renamed per-file before merging, not blindly
  concatenated.
- `scripts/build_method_dataset.sh` referenced 6 of the old test paths; updated to the 4 new ones.
- **Untouched, no fragmentation to fix**: `test_co_evolution.py`, `test_source_evolution.py`,
  `test_knowledge_base.py`, `test_evolving_cli.py`, `test_dictionary_curation_adapter.py`,
  `test_numerical_codegen.py`, `test_evolution_history.py`, `test_evolution_loop.py`,
  `test_skills_reference.py`, `test_analysis_skills.py`, `test_minimal_system.py`,
  `test_gap_control.py`, `test_regime_normalization.py`, `test_explicit_values.py`,
  `test_forecast_workspace.py`, `test_code_evolution.py`.

## Uncommitted work on the feature branch

91 files touched (git status), roughly: 27 modified source files (`common/`, `evolving_loop/`,
`numerical_agent/`, 2 `scripts/*.sh`), 42 deleted test files, 13 new/untracked test files, plus
`numerical_agent/curation/scaled.py` (rewritten, still untracked),
`numerical_agent/evolution/history.py` (untracked, see below), `runs/dictionary_curation_v007/`
(untracked run output), and this file. Run `git status --porcelain` for the exact list — it is
long and reorganizing it into a commit is worth doing deliberately rather than `git add -A`.

**Outstanding: none of this session's work is committed.** That is the top pending task whenever
the user is ready — review `git status`/`git diff` carefully before staging, since 42 deletions +
13 new files in `tests/` will look alarming in a diff without the context above.

## The evolution-history feature — appears complete, not just partially implemented

Plan file: `/PHShome/ik832/.claude/plans/ok-can-you-add-wise-hearth.md` (from a prior session).

Problem it solves: `build_improve_request` passed only the current module source and the current
generation's measurements, so every generation re-derived conclusions earlier generations already
paid for.

Design: derive memory from git, not a memory file. The evolution repo's commit history already
encodes every operation (`- {op} {name}: {reason}` lines under a `generation N: K operations`
subject, written by `commit_module`).

Status: `numerical_agent/evolution/history.py` (untracked) has `Operation`, `History`,
`parse_history`, fully covered by `tests/test_evolution_history.py` (untracked, 24 tests, all
passing). `numerical_agent/evolution/prompts.py` already has `describe_past_generations()` and
`build_improve_request(history=..., live=...)`, wired into `evolve_once` in
`numerical_agent/evolution/__init__.py` (`parse_history(git(root, "log", ...))` feeds
`build_improve_request`), and `IMPROVE_METHODS_PROMPT` has the "Anything the history above establishes is
settled" rule. This looks done, not partial — a prior handoff's "partially implemented" note was
stale. Worth a final read-through against the original plan file before calling it fully closed,
but no code gaps were found this session.

## The v007 run — CURRENT STATE: KILLED, never relaunched

This was the active task before the metric work took over. The user asked to kill it; the process
is dead and the monitor is stopped. Not touched again this session.

- Output dir: `/home/ik832/time-series/runs/dictionary_curation_v007/` (`full/`, `smoke/`,
  `run.log`)
- Launched from `/tmp/ts-main-v007` via
  `nohup bash scripts/run_dictionary_curation.sh full > run.log 2>&1 & disown`
- Config verified live at the time: `method_metric`/`dictionary_metric` = `"smae"` (now scored
  with the corrected paper formula if relaunched from current code, not the seasonal-naive
  version that was live then), `accepted_max_error` 1.0, `specialized_max_error` 2.0 (still
  needs recalibrating per the open item above), 10 generations, 80 train / 20 dev tasks, 111
  methods.
- Model `Qwen/Qwen3.5-27B`, pinned `--device cuda:7`.
- Progress when killed: still in **seed bootstrap**, ~21 of 111 methods implemented. **No
  generation ever completed, so there are no sMAE/sRMSE table values to report.**
- Speed was the real problem: many seeds took ~2.35e6 ms (~40 min) each — never diagnosed.
  **If relaunching, both the latency problem and the stale thresholds need addressing first,
  and note the scoring code underneath has changed since this run started.**

### GPU lessons learned (important, cost hours — unchanged this session)

- `nvidia-smi -q -d COMPUTE` shows compute mode. On this node GPUs **0, 4, 5 are
  `Exclusive_Process`** and held by other users' jobs — a second process gets `CUDA error:
  device(s) is/are busy or unavailable` **regardless of free memory shown**. Not an OOM; do not
  misdiagnose it as one.
- GPUs **1, 2, 3, 6, 7 are `Default`** mode and shareable.
- **Bug, still unfixed: `shard_max_memory()` in `common/llm.py` ignores `CUDA_VISIBLE_DEVICES`
  remapping.** It queries raw physical GPU indices, so `CUDA_VISIBLE_DEVICES=1,2,3,6,7` makes it
  request devices that don't exist in the remapped view → `RuntimeError: Invalid device
  argument`. **Do not use `CUDA_VISIBLE_DEVICES` + auto-sharding until this is fixed.**
- Working approach: a single explicit `--device cuda:N` pin on a `Default`-mode GPU with genuine
  headroom.
- `pkill` on the launcher does **not** kill the child `numerical_agent curate` process. Find
  strays with `pgrep -af "numerical_agent curate"` and `nvidia-smi
  --query-compute-apps=pid,used_memory,gpu_uuid --format=csv`, then `kill -9` each.

## Pending tasks

1. **Commit this session's work** on `feature/compositional-skill-evolution`. Not done, not
   requested yet. Do not push.
2. **Recalibrate `accepted_max_error`/`specialized_max_error`** in `numerical_agent/config.py`
   for the corrected, uncapped metric scale — empirically, from a real generation's measurements.
3. Decide whether to bring `drcik_agent/metrics.py`'s proxy metrics in line with the corrected
   `common/metrics.py` definition, or leave the divergence as-is.
4. Decide whether to relaunch v007 — diagnose the ~40 min/call latency first, and note the scoring
   under it has changed twice since it was launched.
5. Open question, never resolved: whether to re-run the other evolution branches now that the
   optimization objective has changed metric definition twice this session (seasonal-naive →
   corrected paper formula, capped → uncapped).
6. `test_prompt_advertises_exactly_the_sandbox_allow_list` and
   `test_fresh30_launcher_dry_run_renders_four_methods_for_one_smoke_task` remain failing on a
   clean run — pre-existing, not investigated this session either.

## User preferences

- Terse docstrings: one to two sentences, no filler.
- Build one step, verify, stop for approval before the next.
- Do not push to any remote until explicitly told.
- Only commit when explicitly asked.
