# Inputs needed to rebuild the task cache (`scripts/nrd_precompute.py`)

Copy them to the paths the scripts expect (from the repo root):

```bash
I=experiments/self_evolving/final_method/inputs
mkdir -p .scratch/self_evolving runs/method_evolution/v001
cp $I/cordp_cards_{train,dev,public_test}.json .scratch/
cp $I/full_methods.json .scratch/self_evolving/
cp $I/method_evolution_v001/* runs/method_evolution/v001/
```

| File | What it encodes |
|---|---|
| `cordp_cards_{train,dev,public_test}.json` | Document-derived **future corrections** per task (one-time LLM extraction from the task documents): `{task_id: {"confidence": c, "corrections": [[start_timestamp, end_timestamp, multiplier], ...]}}`. `nrd_precompute.py` maps the timestamps to forecast-step windows; these are the corrections the future-correction step (nrd4 team / evolved correction functions) decides how to apply. No labels are used to build them. |
| `full_methods.json` | The list of the 31 method names cached in the task cache (deep forecasters + statistical baselines, incl. `toto_2_0`, `chronos_bolt`). |
| `method_evolution_v001/` | The method portfolio the forecasts come from: `methods.py` (statistical method implementations), `skills.py`, `policies.py` (foundation-model and combined policies), `dictionary.py` (screening policy; its fingerprint keys the forecast cache), `excluded_methods.json`. |

The forecast cache `runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906/` (≈199 MB, read by `nrd_precompute.py` with `cache_only=True`) is on Hugging Face: [yyoraa/drcik-forecast-cache](https://huggingface.co/datasets/yyoraa/drcik-forecast-cache). Unpack it with `mkdir -p runs/champion_forecasts && tar -xzf champion_forecasts_gpt56sol_high_toto_balanced_v3_20260906.tar.gz -C runs/champion_forecasts`.
