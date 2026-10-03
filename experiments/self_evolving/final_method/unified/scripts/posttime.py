"""PostTime reproduction. usage: posttime.py prior|sft|rlvr|predict|score|smoke dataset"""
import gc, json, math, re, shutil, sys
from collections import defaultdict

import numpy as np
import torch

from learned_common import OUT, TEST, load, score, seed

BASE = "google/gemma-3-4b-it"
mode, dataset = sys.argv[1:3]
path = OUT / ("_smoke/posttime" if mode.startswith("smoke") else "posttime") / dataset
prior_file = OUT / "posttime" / dataset / "prior.json"


def priors():
    import timesfm as tfm
    rows = load(dataset, ("train", "dev") + TEST[dataset])
    out = json.load(open(prior_file)) if prior_file.exists() else {}
    groups = defaultdict(list)
    for row in rows:
        if row["tid"] not in out: groups[len(row["truth"])].append(row)
    model = tfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(tfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True,
                                     use_continuous_quantile_head=True, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))
    path.mkdir(parents=True, exist_ok=True)
    for horizon, group in groups.items():
        for start in range(0, len(group), 64):
            batch = group[start:start + 64]
            forecasts, _ = model.forecast(horizon=horizon, inputs=[row["history"][-1024:] for row in batch])
            for row, values in zip(batch, forecasts): out[row["tid"]] = np.asarray(values)[:horizon].tolist()
            json.dump(out, open(prior_file, "w"))
    print(prior_file, len(out))


def prompt(row, prior):
    history = ", ".join(f"{x:.5g}" for x in row["history"][-96:])
    forecast = ", ".join(f"{x:.5g}" for x in prior)
    return ("Revise a numerical time-series forecast using only the history and context. "
            "Return JSON with keys decision (preserve or revise) and forecast (a numerical list of the same length).\n"
            f"Frequency: {row['freq']}\nHistory: [{history}]\nTimesFM forecast: [{forecast}]\n"
            f"Context: {row.get('text', '')[:6000]}\nJSON:")


def answer(row, prior):
    before = np.mean(np.abs(np.asarray(prior) - row["truth"]))
    decision = "preserve" if before < 1e-8 else "revise"
    return json.dumps({"decision": decision, "forecast": [round(x, 8) for x in row["truth"]]})


def parse(text, horizon):
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        values = [float(x) for x in data["forecast"]]
    except Exception:
        values = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    return values if len(values) == horizon and all(math.isfinite(x) for x in values) else None


def base_model():
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    return AutoModelForCausalLM.from_pretrained(BASE, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))


class SFTData(torch.utils.data.Dataset):
    def __init__(self, rows, prior, tokenizer, limit=None):
        self.rows, self.prior, self.tokenizer = rows[:limit] if limit else rows, prior, tokenizer
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; p = prompt(row, self.prior[row["tid"]]); a = answer(row, self.prior[row["tid"]])
        left = self.tokenizer(p, truncation=True, max_length=6144, add_special_tokens=True)["input_ids"]
        right = self.tokenizer(a + self.tokenizer.eos_token, truncation=True, max_length=2048,
                               add_special_tokens=False)["input_ids"]
        ids = left + right
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(left) + right}


def collate(tokenizer):
    def run(rows):
        length = max(len(row["input_ids"]) for row in rows)
        def pad(values, fill): return values + [fill] * (length - len(values))
        return {"input_ids": torch.tensor([pad(row["input_ids"], tokenizer.pad_token_id) for row in rows]),
                "attention_mask": torch.tensor([pad(row["attention_mask"], 0) for row in rows]),
                "labels": torch.tensor([pad(row["labels"], -100) for row in rows])}
    return run


def sft(smoke=False):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    seed(); prior = json.load(open(prior_file)); tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token; model = prepare_model_for_kbit_training(base_model())
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    args = TrainingArguments(output_dir=str(path / "sft_steps"), num_train_epochs=1 if smoke else 3,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-4,
        bf16=True, logging_steps=10, save_strategy="epoch", report_to="none")
    data = SFTData(load(dataset, ("train",)), prior, tokenizer, 8 if smoke else None)
    Trainer(model=model, args=args, train_dataset=data, data_collator=collate(tokenizer)).train()
    model.save_pretrained(path / "sft_steps" / "checkpoint-final")
    tokenizer.save_pretrained(path / "sft_steps" / "checkpoint-final")
    del model; gc.collect(); torch.cuda.empty_cache()
    select("sft", load(dataset, ("dev",))[:4] if smoke else load(dataset, ("dev",)))


def reward(completions, truth, prior, **kwargs):
    out = []
    for completion, target, baseline in zip(completions, truth, prior):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        values = parse(text, len(target))
        if values is None: out.append(0.0); continue
        mae = np.mean(np.abs(np.asarray(values) - target))
        base = np.mean(np.abs(np.asarray(baseline) - target))
        out.append(float(np.clip(.5 + .5 * (1 - mae / (base + 1e-8)), 0, 1)))
    return out


def rlvr(smoke=False):
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    seed(); prior = json.load(open(prior_file)); tokenizer = AutoTokenizer.from_pretrained(BASE)
    rows = load(dataset, ("train",))
    if smoke: rows = rows[:8]
    data = Dataset.from_list([{"prompt": prompt(row, prior[row["tid"]]), "truth": row["truth"],
                               "prior": prior[row["tid"]]} for row in rows])
    model = PeftModel.from_pretrained(base_model(), path / "sft", is_trainable=True)
    args = GRPOConfig(output_dir=str(path / "rlvr_steps"), num_train_epochs=1,
                      per_device_train_batch_size=1, gradient_accumulation_steps=8,
                      learning_rate=1e-5, bf16=True, max_completion_length=128 if smoke else 2048,
                      num_generations=4, report_to="none")
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=args, train_dataset=data,
                          processing_class=tokenizer)
    trainer.train(); trainer.model.save_pretrained(path / "rlvr_steps" / "checkpoint-final")
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    select("rlvr", load(dataset, ("dev",))[:4] if smoke else load(dataset, ("dev",)))


def run_rows(adapter, rows, prior):
    from peft import PeftModel
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base_model(), adapter).eval()
    out = {}
    for row in rows:
        inputs = tokenizer(prompt(row, prior[row["tid"]]), return_tensors="pt", truncation=True,
                           max_length=6144).to(model.device)
        limit = min(2048, 64 + 10 * len(row["truth"]))
        with torch.no_grad(): ids = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        text = tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out[row["tid"]] = parse(text, len(row["truth"])) or prior[row["tid"]]
    del model; gc.collect(); torch.cuda.empty_cache()
    return out


def select(stage, rows):
    prior = json.load(open(prior_file)); candidates = sorted((path / f"{stage}_steps").glob("checkpoint-*"))
    scores = {}
    for candidate in candidates:
        forecasts = run_rows(candidate, rows, prior)
        values = []
        for row in rows:
            target = np.asarray(row["truth"]); pred = np.asarray(forecasts[row["tid"]])
            scale = max(float(np.mean(np.abs(target))), 1e-8)
            values.append(min(5.0, float(np.mean(np.abs(target - pred)) / scale)))
        scores[candidate.name] = float(np.mean(values))
        print(stage, candidate.name, "dev_smae", round(scores[candidate.name], 4), flush=True)
    best = min(candidates, key=lambda candidate: scores[candidate.name])
    shutil.copytree(best, path / stage, dirs_exist_ok=True)
    json.dump({"stage": stage, "best": best.name, "dev_smae": scores[best.name], "candidates": scores},
              open(path / f"{stage}_selection.json", "w"), indent=2)


def generate():
    prior = json.load(open(prior_file)); adapter = path / ("rlvr" if (path / "rlvr").exists() else "sft")
    rows = load(dataset, TEST[dataset]); output = path / "predictions.json"
    out = json.load(open(output)) if output.exists() else {}
    pending = [row for row in rows if row["tid"] not in out]
    for i in range(0, len(pending), 20):
        out.update(run_rows(adapter, pending[i:i + 20], prior))
        json.dump(out, open(output, "w"))
        print(min(i + 20, len(pending)), len(pending), flush=True)


if mode == "prior": priors()
elif mode == "sft": sft()
elif mode == "smoke": sft(True)
elif mode == "rlvr": rlvr()
elif mode == "smoke_rlvr": rlvr(True)
elif mode == "predict": generate()
elif mode == "score": score("posttime", dataset)
else: raise SystemExit("usage: posttime.py prior|sft|rlvr|predict|score|smoke|smoke_rlvr drcik|tmmd|timesx")
