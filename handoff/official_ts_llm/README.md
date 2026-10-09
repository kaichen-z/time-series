# Official Train/Dev LLM handoff

This directory is the label-free handoff for running Claude Haiku document
extraction and Opus-assisted offline evolution against the same opaque task
identities used by the official benchmark loader.

## Files

- `timesx_train_dev_documents.jsonl`: 2,723 TimesX Train/Dev tasks and 10,892
  public documents. Each row contains `task_id`, `split`, public timestamps,
  frequency, horizon, target description, and four document records.
- `time_mmd_train_dev_task_ids.jsonl`: 51,728 Time-MMD Train/Dev task IDs with
  split, group, frequency, horizon, and references to official retrieved-text
  documents at the forecast origin.
- `time_mmd_documents.jsonl`: deduplicated Time-MMD `Final_Search_2/4/6`
  retrieved facts. `Final_Output` is deliberately excluded because it is a
  closed-source-LLM prediction rather than source evidence.
- `receipt.json`: exact source commits, loader fingerprints, row counts, and
  SHA-256 hashes.
- `alignment_manifest.json`: the full path/hash/split protocol consumed by the
  executable official loader.

The bundle contains **no Test identities, Test labels, Train/Dev labels,
numeric histories, model forecasts, anchor caches, or credentials**.

## Official datasets

- TimesX: <https://github.com/haoxin1998/TimesX-project>, pinned to
  `26bc70cfa669407c71ae1ad6ce2194be9cfe9a0b`.
- Time-MMD / MM-TSFlib: <https://github.com/AdityaLab/MM-TSFlib>, pinned to
  `e789ce78c9bafd8e3ba0d8850f9ad2becbe83548`.

The executable source/hash contract is in
`evolving_loop/official_benchmark_loader.py`. Regenerate this handoff with:

```bash
PYTHONPATH=. python scripts/export_official_llm_handoff.py \
  --manifest handoff/official_ts_llm/alignment_manifest.json \
  --timesx-root /path/to/TimesX-project \
  --time-mmd-root /path/to/MM-TSFlib \
  --output handoff/official_ts_llm
```

## Haiku output contract

Process every unique document in `timesx_train_dev_documents.jsonl` and
`time_mmd_documents.jsonl` without using future numerical values. Return one
JSONL row per `document_id`; the task files provide the task-to-document join:

```json
{
  "document_id": "official_...",
  "extractor": {"provider": "anthropic", "model": "...", "prompt_sha256": "..."},
  "events": [
    {
      "time_start": "YYYY-MM-DD or null",
      "time_end": "YYYY-MM-DD or null",
      "direction": "up|down|mixed|unknown",
      "confidence": 0.0,
      "evidence": "short source-grounded paraphrase"
    }
  ]
}
```

Requirements:

1. Preserve every `task_id` and `document_id` exactly; do not synthesize IDs.
2. Do not infer or emit target values, forecasts, weights, or Test records.
3. Use `null`/`unknown` rather than inventing dates or direction.
4. Record the exact model identifier and prompt hash.
5. Keep the output one line per document and valid UTF-8 JSONL.

Opus can work on prompts/policies using these IDs and Haiku cards, but final
candidate scoring remains Train/CV-only in this repository. Dev is evaluated
once after freezing the policy; Test remains sealed.
