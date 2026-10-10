# Audited Haiku event-card split

This directory separates commit `f8e4762185b354582363930ee219295aebee3c6e`
into TimesX and Time-MMD without changing document IDs or introducing labels,
numeric histories, or Test identities.

Use the `*_events_valid.jsonl` files for downstream training/evolution. They
preserve every document row and every valid event, while quarantining 187
invalid event records listed in `validation_errors.jsonl`:

- 120 reversed date ranges (`time_start > time_end`)
- 3 invalid calendar dates
- 64 events with empty evidence

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

The evolution runner is still invoked with a benchmark-aligned task file, not
the event JSONL directly:

```bash
scripts/run_meta_harness_v2.sh TASKS_PATH OUTPUT_DIR
```

Before running, join the relevant `*_events_valid.jsonl` to the official task
documents by unchanged `document_id`. Use only the Train partition to evolve;
freeze the winning policy before a one-time Dev evaluation. The current wrapper
defaults to medium reasoning, so set its Codex reasoning effort to `high` to
reproduce the frozen pilot protocol.
