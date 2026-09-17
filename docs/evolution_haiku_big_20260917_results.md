# Real-evolve big run (Haiku) — results, 2026-09-17

Branch: `fix/p2-feasible-child`. First full P2→P5 real-evolve completed on the
Linux server with the Claude CLI (Haiku proposer), on the migrated repo at
`/home/yiqi/yiyun/draft/time-series`.

## Configuration

| Item | Value |
|------|-------|
| Manifest | `configs/evolution_v2/real/real-30m-toto-claude-server.json` |
| Proposer LLM | Claude **Haiku** (`EVOLVE_CLAUDE_MODEL=haiku`; manifest model binding is contract-frozen to `gpt-5.6-luna`) |
| Flags | `--no-time-limit --p2-generations 40 --p3-steps 8 --p4-candidates 3 --eval-train-size 8 --eval-dev-size 3 --eval-folds 2` |
| Seed champion | `gpt56sol_high_toto_balanced_v3_20260906` (evolved earlier with gpt-5.6-luna) |
| Forecasts | cache-only (`champion_forecasts`, 49,340 rows; covers train/dev/test) |
| Data split | `drcik_public_80_20_99_v3` — train 80 / dev 20 / public_test 99 |

Note on evaluation scope per stage:
- **P2 (numerical methods)** evaluates on the **full 80 train** (hyperband rungs 8→32→80, 5-fold group CV).
- **P3/P4/P5** evaluate on a small entity-disjoint **projection** (here train 8 / dev 3), via `select_real_task_projection`.

## Stage results (all `complete`, phase COMPLETE)

| Stage | Evolves | Time | Outcome |
|-------|---------|------|---------|
| P2 | numerical methods | 4356s (~73m) | 40 generations; **nothing beat seed** (promotion = seed only) |
| P3 | numerical+retrieval+decision (cooperative) | 117s | **accepted** a joint-arm mutation → **retrieval+decision evolved** (retrieval `2c51c4bf→7d3abbc2`, decision `c2fc1c50→a95b1225`), numerical unchanged (`46f6a1fd`). Passed the strict `dev_passed` gate (train non-regression + strictly lower dev joint on the 3 dev tasks). |
| P4 | `choose_arm` routing source | 3200s (~53m) | 3 candidates; best train_gain +0.093 but **failed the dev gate** → routing stays seed (`return "numerical"`) |
| P5 | infrastructure protocol | 0s | sealed (cache replay after the projection fix) |

Champion = seed numerical + **evolved retrieval+decision** + seed routing.

## Held-out test (99 public_test) — champion vs seed

Evaluated with `scripts/eval_champion_on_test.py` (materializes the seed numerical
registry over the 99 test tasks from the champion supply release + cached
forecasts, then runs the two-stage pipeline with champion vs seed
retrieval/decision; numerical is identical so it cancels in the comparison).

| Config | sMAE | sRMSE | joint | coverage |
|--------|------|-------|-------|----------|
| seed | 0.38259 | 0.58820 | **0.48539** | 1.0 |
| champion | 0.38259 | 0.58820 | **0.48539** | 1.0 |

**Relative gain on test = +0.0000** — the dev-validated retrieval+decision
improvement did **not** generalize to the 99 held-out test tasks (champion ≡ seed;
the decision agent falls back to the same final numerical forecast on test).

## Diagnosis

- **P2** evaluated on the full 80 train and still found no improvement in 40
  generations → not a small-sample issue; likely a **weak proposer (Haiku)** vs a
  luna-evolved seed, and/or low headroom for the fixed method family. This is the
  accuracy-determining layer.
- **P3/P4/P5** use a tiny projection (train 8 / dev 3) → acceptance signal is
  noisy; a 3-task "dev win" does not generalize (P3 passed dev but tied on test;
  P4's train gain failed dev).

## Highest-leverage next steps
1. Stronger proposer for P2 (Opus/Sonnet, or gpt-5.6-luna via codex if available).
2. Larger acceptance projection for P3/P4/P5 (dev pool has 20 tasks / 12 distinct
   entities → dev up to ~10; train up to ~44).

## Robustness/infra fixes landed this run (all pushed)
- `767405b` P4 source `limit` cap 2→len(templates)=3.
- `fab294c` resume real-evolve from a mid-stage break (crash / token exhaustion in
  `--no-time-limit`) instead of failing closed; seal-verification failures still
  fail closed.
- `c4958a7` P5 protocol compatibility made projection-size agnostic.
- `510213f` P3 steps / P4 candidates / task projection made adjustable CLI flags;
  `EVOLVE_CLAUDE_MODEL` override for the Claude proposer model.
