# Audited Haiku event-card split

This directory separates commit `f8e4762185b354582363930ee219295aebee3c6e`
into TimesX and Time-MMD without changing document IDs or introducing labels,
numeric histories, or Test identities.

Use the `*_events_valid.jsonl` files for downstream training/evolution. They
preserve every document row and every valid event, while quarantining 301
invalid or unsafe event records listed in `validation_errors.jsonl`:

- 120 reversed date ranges (`time_start > time_end`)
- 3 invalid calendar dates
- 64 events with empty evidence
- 114 TimesX scenario events starting at or after the forecast origin

Future TimesX holiday cards are retained because their calendar is known in
advance. Future-starting scenario cards are excluded conservatively rather
than interpreted as known-ahead information.

The raw `*_events.jsonl` files are immutable benchmark splits of the supplied
Haiku output and retain those invalid events for provenance. No invalid event
was silently repaired or reinterpreted.

`*_document_split_membership.jsonl` records whether each document belongs to
the frozen Train or Dev task partition. The event files themselves remain
label-free. `split_manifest.json` contains all counts and SHA-256 hashes.

Regenerate and audit the split with:

```bash
python scripts/split_official_haiku_events.py \
  --handoff-root handoff/official_ts_llm \
  --output-dir handoff/official_ts_llm/haiku_output/split
```

Two deterministic Train-only pilot task files are included:

- `timesx_train3_event_tasks.jsonl`: all 3 TimesX Train groups, one task per
  group (the runner internally uses 2 Train / 1 validation task).
- `time_mmd_train10_event_tasks.jsonl`: 10 deterministically selected Time-MMD
  Train groups, one task per group (internally 8 / 2).

Their `.receipt.json` files bind source hashes, task/document joins, selected
groups, and output hashes. Official external Dev and sealed Test are absent.

Run the frozen one-generation pilot from the repository root:

```bash
scripts/run_official_event_evolution.sh \
  handoff/official_ts_llm/haiku_output/split/timesx_train3_event_tasks.jsonl \
  runs/official_event_evolve/timesx

scripts/run_official_event_evolution.sh \
  handoff/official_ts_llm/haiku_output/split/time_mmd_train10_event_tasks.jsonl \
  runs/official_event_evolve/time_mmd
```

This wrapper fixes `gpt-5.6-sol`, high reasoning, the accepted seed policy,
four children, successive halving, and Train-only internal selection. It does
not evaluate official external Dev or Test. This is a small reproducible pilot,
not a full-dataset training claim.
