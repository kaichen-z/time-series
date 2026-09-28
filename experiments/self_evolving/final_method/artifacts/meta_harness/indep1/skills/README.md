# Visible submission comparison

Run `python3 visible_delta.py 0006_i1 0014_i1` from this directory to compare two submitted harnesses. The script reports fitness and every visible task whose gain changed, together with its cell and first document title. It reads only `shared/attempts/` and the unlabeled `shared/views_train.json`.

Use this after each submission to see whether a rule changed the intended tasks and how concentrated its gain is. The output is diagnostic, not a substitute for the evaluator's hidden check.
