# Visible correction audit

Run `python3 shared/skills/audit_visible_corrections.py --run-dir "$RUN_DIR"` from the run directory. It reconstructs the current base program from the **visible** trace and reports each extracted correction's proposed multiplier, observed median forecast factor, document features, and total task gain. Task 43 is marked because an accepted history repair also changes its forecast. Overlapping correction windows can make an individual ratio ambiguous. The script never reads hidden labels or evaluator internals.
