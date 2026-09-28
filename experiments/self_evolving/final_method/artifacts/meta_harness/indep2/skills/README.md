# Visible correction inspection

`inspect_corrections.py` lists each correction's window, multiplier, median
forecast level relative to `sigma_main_calib`, steps changed by a harness,
base-only gain, and full solo-correction gain.

From the run directory:

```sh
python3 shared/skills/inspect_corrections.py shared/best_harness.py
```

The script reads only `shared/views_train.json` and
`shared/traces/visible_tasks.jsonl`. It does not evaluate hidden tasks or
compute official fitness. Several visible tasks share the same underlying
series, so treat similar rows as correlated evidence.
