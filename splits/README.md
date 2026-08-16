# Dr-CiK public evolution splits

The repository freezes two deterministic, entity-disjoint manifests over the 199 tasks in the
public Dr-CiK development set. The official 80 hidden-test tasks are never included.

## Recommended formal protocol

Use `drcik_public_80_20_99_v1.json` for the main experiment.

| Partition | Tasks | Entities | Permitted use |
|---|---:|---:|---|
| Train | 80 | 47 | Outcome analysis, dictionary/skill mutation, and fitting |
| Dev | 20 | 10 | Parent/child selection and early stopping only |
| Public Test | 99 | 56 | One frozen evaluation after all method selection |

Train and Dev form a 100-task development pool, while the other 99 public tasks provide a large
untouched test set. Detailed Dev failures must not be fed back into the Evolver; only the aggregate
accept/reject signal may be used. Public Test must not be accessed until prompts, dictionaries,
budgets, and harness topology are frozen.

The manifest SHA-256 is
`3cc81f45878c1aae93e5ba48dc367df6553698db6661dbe06fbe5efb06afca92`.

## Legacy exploratory protocol

`drcik_public_v1.json` preserves the earlier 139 Train / 30 Dev / 30 Public-Test split so historical
experiments remain reproducible. It should not be mixed with results from the recommended protocol.

## Construction rules

Both manifests use seed `20260816`, keep each entity in exactly one partition, and balance frequency,
forecast-horizon bucket, reasoning hops, and task origin where possible. Assignment uses metadata
only. It does not inspect future values, ground-truth evidence, document relevance labels, or
document text.

After downloading the full public task export, reproduce the recommended manifest with:

```bash
python -m evolving_agent.split_manifest \
  --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --output splits/drcik_public_80_20_99_v1.json \
  --seed 20260816
```

To reproduce the legacy manifest, pass explicit sizes:

```bash
python -m evolving_agent.split_manifest \
  --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --output splits/drcik_public_v1.json \
  --seed 20260816 \
  --train-size 139 \
  --dev-size 30 \
  --public-test-size 30
```

Each manifest contains only task IDs, entity names, marginal metadata distributions, and a digest.
It contains no evaluator-only labels or document contents.
