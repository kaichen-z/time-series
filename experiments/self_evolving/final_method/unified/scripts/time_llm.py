"""Train and evaluate the GPT-2 Time-LLM baseline. usage: time_llm.py train|predict|score|smoke dataset"""
import sys
from learned_common import OUT, TEST, Tasks, fit, load, predict, score
from learned_models import TimeLLM

mode, dataset = sys.argv[1:3]
path = OUT / ("_smoke/time_llm" if mode == "smoke" else "time_llm") / dataset
model = TimeLLM()
tasks = lambda rows: Tasks(rows, model.tokenizer, model.context, model.horizon)
if mode in ("train", "smoke"):
    train, dev = load(dataset, ("train",)), load(dataset, ("dev",))
    if mode == "smoke": train, dev = train[:8], dev[:4]
    fit(model, tasks(train), tasks(dev), path / "best.pt", epochs=1 if mode == "smoke" else 20, batch=4)
elif mode == "predict":
    model.load_trainable_state(__import__("torch").load(path / "best.pt", map_location="cpu"))
    predict(model, tasks([]), load(dataset, TEST[dataset]), path / "predictions.json", batch=4)
elif mode == "score": score("time_llm", dataset)
else: raise SystemExit("usage: time_llm.py train|predict|score|smoke drcik|tmmd|timesx")
