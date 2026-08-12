# Run scripts

Numbered bash wrappers around `evolving-agents`. Run them in order; each prints its
resolved configuration before doing anything, so you can see exactly what it will use.

```bash
./scripts/00_smoke.sh            # minutes  - does anything work at all?
./scripts/01_evolve_coding.sh    # long     - Loop A (run first, run longest)
./scripts/02_evolve_retrieval.sh # long     - Loop B (needs Loop A's winner)
./scripts/03_evolve_system.sh    # long     - Loop C (needs A's and B's winners)
./scripts/04_baselines.sh        # medium   - the numbers everything is compared against
./scripts/05_final_eval.sh       # once     - the held-out test split, one time only
```

## Configuration

Everything lives in `env.sh` and is overridable from your shell:

```bash
EA_GENERATIONS=3 ./scripts/01_evolve_coding.sh      # shorter run
EA_USE_SAMPLE=1 ./scripts/01_evolve_coding.sh       # 3-task sample instead of all 199
EA_TRACE_LEVEL=full ./scripts/01_evolve_coding.sh   # print every prompt and reasoning block
EA_WORKER_DEVICE=cuda:3 ./scripts/01_evolve_coding.sh
```

| Variable | Default | What it does |
|---|---|---|
| `EA_USE_SAMPLE` | `0` | `1` uses the 3-task sample dir instead of the full dataset |
| `EA_OUT_ROOT` | `/raid/.../evolving_agents_out` | Everything written by a run |
| `EA_GENERATIONS` / `EA_POPULATION` / `EA_KEEP_ELITE` | `10` / `6` / `2` | Evolution budget |
| `EA_MINIBATCH` | `20` | Tasks scored per individual per generation |
| `EA_TRACE_LEVEL` | `summary` | `off` / `summary` / `full` |
| `EA_WORKER_MODEL` | `Qwen/Qwen2.5-14B-Instruct` | Runs inside the three agents |
| `EA_EVOLVER_MODEL` | `Qwen/Qwen3.5-35B-A3B-FP8` | Rewrites bundles; called rarely, so it can be big |
| `EA_WORKER_DEVICE` / `EA_EVOLVER_DEVICE` | auto | Pinned to the two freest GPUs at launch |
| `EA_SEED` | `7` | Splits, minibatches, sampling |
| `EA_LOG_DIR` | `<repo>/logs` | Where the full log files go |
| `EA_CONSOLE_LEVEL` | `INFO` | Terminal verbosity; `WARNING` for a near-silent run |
| `EA_LOG_LEVEL` | `INFO` | Log-file verbosity |
| `EA_TRACE_CONSOLE` | `0` | `1` also streams the per-call trace to the terminal |

**Why the defaults point at `/raid`:** the root filesystem is effectively full (a few GB
free, shared with other users) and `/raid` has hundreds of GB. Model downloads (`HF_HOME`),
the LLM cache, checkpoints and run logs all go to `/raid` so a long run cannot fill `/` and
take other people's jobs down with it. Each script refuses to start if the target disk is
too tight.

## Logs: the terminal stays readable, the file has everything

The per-call trace is thousands of lines, so it is **not** printed to your terminal. Each
run writes a timestamped log file and tells you where it is:

```
==> full log (every prompt, tool call and reasoning block):
    /home/ik832/time-series/logs/loop_a_20260811_1120.log
==> watch it live in another terminal with:
    tail -f /home/ik832/time-series/logs/loop_a_20260811_1120.log
```

| Stream | Gets | Typical volume |
|---|---|---|
| terminal (stderr) | config, per-task progress, generation results, warnings, errors | ~10 lines/task |
| log file | all of the above **plus** every LLM call/response, tool call, reasoning block, traceback | ~50+ lines/task |
| stdout | the summary JSON only, redirected to `results/<run>.summary.json` | one object |

```bash
EA_TRACE_CONSOLE=1 ./scripts/01_evolve_coding.sh   # stream the trace live too
EA_CONSOLE_LEVEL=WARNING ./scripts/01_evolve_coding.sh   # near-silent; errors only
EA_LOG_LEVEL=DEBUG ./scripts/01_evolve_coding.sh   # even more in the file
EA_LOG_DIR=/raid/.../logs ./scripts/01_evolve_coding.sh
```

Logs are timestamped per run, so reruns never overwrite each other. Uncaught tracebacks are
written into the log file too, not just onto a terminal you are trying not to read.

## Where output goes

```
$EA_OUT_ROOT/
  runs/loop_{a,b,c}.jsonl        one compact record per task (hashes only, no prompt text)
  runs/reasoning/<model>/*.txt   full reasoning blocks, written when --trace-level summary
  checkpoints/<run>/gen_NNN.json one per generation; re-running resumes from the last one
  bundles/<agent>/vNNN.json      every bundle evolution ever produced, with parent pointers
  results/<run>.summary.json     the final summary each script produced
logs/<run>_<timestamp>.log       the full human-readable log for that run
```

Three places carry detail, for different purposes: the **log file** is for reading a run
back, `runs/*.jsonl` is compact and machine-parseable for analysis across a whole run, and
`runs/reasoning/` holds the raw reasoning blocks addressed by prompt hash.

## Two things to know

**Runs resume.** Killing a run and re-running the same script picks up from the last
completed generation. To start clean, delete that run's checkpoint directory.

**Nothing is re-generated twice.** Every LLM call is cached on disk by
`(model, prompt, draw index, thinking mode)`. A re-run replays from cache and costs nothing,
which is also what makes results reproducible.

## Loop ordering is a real dependency, not a suggestion

Loop B freezes Loop A's winning coding bundle; Loop C seeds from both. Running them out of
order works mechanically but freezes seed bundles instead of evolved ones — the scripts warn
when they spot this. The winning bundle id is printed as `best_individual` at the end of each
run; pass its file path to the next script.

## A caveat on the numbers

`sMAE`/`sRMSE`/`sCRPS` come from `dr_cik.evaluation`, which is our own best-effort proxy —
Dr-CiK never publishes its official scale-normalizer formula, and no official scorer ships
with the dataset. Every summary carries a `note` saying so. Use these to compare bundles
against each other, not as leaderboard-comparable absolutes.
