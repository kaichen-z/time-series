# Visible trace numerical probe

`trace_program_probe.py` screens a small numerical-program change using only
`shared/traces/visible_tasks.jsonl` and the current `shared/best_config.json`.

Examples:

```bash
python3 shared/skills/trace_program_probe.py --shrink 0.18
python3 shared/skills/trace_program_probe.py --method stl_ets --weight 0.02
```

It prints an approximate gain by visible fold and the largest task changes.
The script excludes the two accepted history repairs, infers existing document
corrections from the current forecast, and approximates the joint-error scale.
It cannot replace a submission or predict the hidden check. In episode 3, a
small seasonal blend and a lower shrink both improved visible scores but failed
the hidden check.
