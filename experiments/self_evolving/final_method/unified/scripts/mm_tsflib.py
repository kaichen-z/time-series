"""Train and evaluate the Autoformer+BERT MM-TSFlib port. usage: mm_tsflib.py train|predict|score|smoke dataset"""
import sys
from learned_common import OUT, TEST, Tasks, fit, load, predict, score
from learned_models import MMTSF

mode, dataset = sys.argv[1:3]
path = OUT / ("_smoke/mm_tsflib" if mode == "smoke" else "mm_tsflib") / dataset
model = MMTSF()
tasks = lambda rows: Tasks(rows, model.tokenizer, model.context, model.horizon)
if mode in ("train", "smoke"):
    train, dev = load(dataset, ("train",)), load(dataset, ("dev",))
    if mode == "smoke": train, dev = train[:8], dev[:4]
    fit(model, tasks(train), tasks(dev), path / "best.pt", epochs=1 if mode == "smoke" else 20, batch=8)
elif mode == "predict":
    model.load_trainable_state(__import__("torch").load(path / "best.pt", map_location="cpu"))
    predict(model, tasks([]), load(dataset, TEST[dataset]), path / "predictions.json", batch=8)
elif mode == "score": score("mm_tsflib", dataset)
else: raise SystemExit("usage: mm_tsflib.py train|predict|score|smoke drcik|tmmd|timesx")
