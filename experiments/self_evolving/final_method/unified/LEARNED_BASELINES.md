# Learned multimodal baselines

This directory contains the reproducible learned-baseline protocol for Dr-CiK, Time-MMD, and
TimesX. The benchmark reports task-level sMAE and sRMSE for Time-LLM, MM-TSFlib, and PostTime.
The implementation never uses a test target for training, prompting, checkpoint selection, or
RL rewards.

## What is reproduced

| Name | Status | Configuration |
|---|---|---|
| Time-LLM | Architecture port | Frozen `openai-community/gpt2`, patch embedding, reprogramming attention, learned forecast head |
| MM-TSFlib | Dataset adapter and architecture port | Autoformer-style trend/seasonal encoder, frozen `google-bert/bert-base-uncased`, attention pooling |
| PostTime | Independent reproduction | `google/gemma-3-4b-it` revises frozen TimesFM-2.5 forecasts; LoRA SFT followed by GRPO/RLVR |

Time-LLM and MM-TSFlib are ports to the common task interface, not copied snapshots of their
training repositories. The original MM-TSFlib implementation only directly supports its
preprocessed Time-MMD layout. PostTime code and checkpoints were not public when this protocol
was written, so its implementation is explicitly an independent reproduction of the published
recipe. These distinctions should be retained in the paper.

The ports were checked against Time-LLM revision `b13e881f86cd0475ce1b72c17110430663334955`
and MM-TSFlib revision `e789ce78c9bafd8e3ba0d8850f9ad2becbe83548`. To inspect those exact sources:

```bash
git clone https://github.com/KimMeen/Time-LLM.git external/Time-LLM
git -C external/Time-LLM checkout b13e881f86cd0475ce1b72c17110430663334955
git clone https://github.com/AdityaLab/MM-TSFlib.git external/MM-TSFlib
git -C external/MM-TSFlib checkout e789ce78c9bafd8e3ba0d8850f9ad2becbe83548
```

## Data

Run all commands from the repository root. The builder reads:

- Dr-CiK from `.scratch/self_evolving/nrd_cache_full2.json`.
- Time-MMD numerical and textual files from `external/Time-MMD`, using
  `splits/timemmd_531_181_182_v1.json` (531 train, 181 dev, 182 test).
- TimesX from `work/timesx/tasks.json` (2,173 train, 550 dev, 1,695 ID test, 211 OOD test).

For Time-MMD, only documents ending before the forecast cutoff are included. The eight most
recent eligible report/search rows are retained. TimesX context concatenates background,
scenario, holiday, and covariate fields. Dr-CiK uses the task documents already present in the
cache. Histories are forward-filled; an initial missing prefix is filled with zero. A non-finite
Time-MMD training target is forward-filled from the last observed value so every training label
is finite; dev/test targets in the committed split are already finite.

```bash
PYTHONPATH=$PWD .venv/learned/bin/python \
  experiments/self_evolving/final_method/unified/scripts/learned_data.py
```

The output is `work/learned_baselines/tasks.jsonl`. Each row contains `tid`, `dataset`, `part`,
`freq`, `history`, `truth`, and `text`. Test truth is stored only so the final scorer can run;
training code loads only `train`, checkpoint selection loads only `dev`, and model inputs never
contain `truth`.

## Environment

The experiments were designed for CUDA GPUs with bfloat16 support. Time-LLM and MM-TSFlib fit
on one 48 GB GPU. PostTime uses 4-bit Gemma with LoRA and can run on one 48 GB GPU; multiple
independent dataset runs can be assigned to separate GPUs.

The smoke-tested environment used Python 3.11.15, PyTorch 2.14.1, Transformers 5.18.0,
Accelerate 1.15.0, PEFT 0.21.2, TRL 1.14.1, bitsandbytes 0.50.2, and TimesFM 3.0.2 on
NVIDIA RTX 6000 Ada 48 GB GPUs. Record the final paper environment with
`uv pip freeze --python .venv/learned/bin/python` and the GPU driver with `nvidia-smi`.

```bash
uv venv .venv/learned --python 3.11
uv pip install --python .venv/learned/bin/python -r requirements.txt
```

Gemma requires acceptance of Google's model license and an authenticated Hugging Face account:

```bash
hf auth login
```

Model weights remain in the Hugging Face cache. Adapters, checkpoints, forecasts, and metrics
are written below `runs/learned_baselines/`. Smoke checkpoints are isolated under
`runs/learned_baselines/_smoke/` and are never considered by a full run.

## Smoke tests

Build the task file first, then run a one-epoch, eight-example check:

```bash
S=experiments/self_evolving/final_method/unified/scripts
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/learned/bin/python $S/time_llm.py smoke tmmd
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/learned/bin/python $S/mm_tsflib.py smoke tmmd
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py prior tmmd
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py smoke tmmd
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py smoke_rlvr tmmd
```

## Training and evaluation

Time-LLM and MM-TSFlib minimize normalized MAE on the training split. After every epoch they
compute capped task-level dev sMAE and retain the best checkpoint. Context length is 512 and the
maximum forecast horizon is 256, covering all three benchmarks.

```bash
S=experiments/self_evolving/final_method/unified/scripts
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/learned/bin/python $S/time_llm.py train drcik
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/learned/bin/python $S/time_llm.py predict drcik
PYTHONPATH=$PWD .venv/learned/bin/python $S/time_llm.py score drcik

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD .venv/learned/bin/python $S/mm_tsflib.py train tmmd
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD .venv/learned/bin/python $S/mm_tsflib.py predict tmmd
PYTHONPATH=$PWD .venv/learned/bin/python $S/mm_tsflib.py score tmmd
```

PostTime first caches TimesFM-2.5 priors for every split. SFT teaches Gemma to emit a JSON
decision and revised trajectory. RLVR uses GRPO and the clipped improvement reward
`clip(0.5 + 0.5 * (1 - MAE(revision)/[MAE(prior)+1e-8]), 0, 1)`. Malformed or wrong-length
outputs receive zero reward. Prediction falls back to the TimesFM prior when parsing fails.
After each stage, all saved adapters are run on dev and the adapter with the lowest capped
task-level dev sMAE is copied to the stable `sft/` or `rlvr/` directory. The candidate scores
are retained in `<stage>_selection.json`.

```bash
S=experiments/self_evolving/final_method/unified/scripts
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py prior timesx
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py sft timesx
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py rlvr timesx
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py predict timesx
PYTHONPATH=$PWD .venv/learned/bin/python $S/posttime.py score timesx
```

Existing priors and predictions are resumed by task ID. To run every model and dataset
sequentially, use:

```bash
CUDA_VISIBLE_DEVICES=0 PY=.venv/learned/bin/python \
  bash experiments/self_evolving/final_method/unified/scripts/run_learned.sh
```

Restrict a run with `MODELS=time_llm,mm_tsflib` or `DATASETS=tmmd,timesx`. Separate concurrent
runs should use different model/dataset pairs.

### Four-GPU run

The parallel launcher treats each model/dataset pair as one job and dynamically assigns the next
job to the first available GPU. PostTime stages remain ordered inside a job. Logs are written to
`runs/learned_baselines/logs/<model>_<dataset>.log`.

```bash
GPUS=0,1,2,3 PY=.venv/learned/bin/python \
  bash experiments/self_evolving/final_method/unified/scripts/run_learned_parallel.sh
```

Monitor it with:

```bash
watch -n 2 nvidia-smi
tail -f runs/learned_baselines/logs/posttime_timesx.log
```

Completed training stages and predictions are reused. Set `FORCE=1` only when intentionally
retraining every checkpoint. `MODELS` and `DATASETS` accept the same comma-separated filters as
the sequential launcher.

## Outputs and metrics

```text
runs/learned_baselines/
  time_llm/<dataset>/best.pt
  mm_tsflib/<dataset>/best.pt
  posttime/<dataset>/prior.json
  posttime/<dataset>/sft/
  posttime/<dataset>/rlvr/
  <model>/<dataset>/predictions.json
  <model>/<dataset>/metrics.json
  metrics.json
```

Prediction files map task IDs to deterministic forecast arrays. Metric files contain dataset,
split, model, completed/expected task counts, sMAE, and sRMSE. Both errors divide the task MAE
or RMSE by the mean absolute target value, cap each task at 5, then average over tasks. Displayed
values are rounded to three decimals. Missing tasks remain visible through the completed and
expected counts.

| Model | Dr-CiK test | Time-MMD test | TimesX ID | TimesX OOD |
|---|---:|---:|---:|---:|
| Time-LLM | pending | pending | pending | pending |
| MM-TSFlib | pending | pending | pending | pending |
| PostTime SFT+RLVR | pending | pending | pending | pending |

Populate this table only from `runs/learned_baselines/metrics.json` after every completed count
equals its expected count.

## Paper text

> We evaluated three learned multimodal forecasting baselines using fixed, dataset-specific
> train/dev/test partitions. Time-LLM used a frozen GPT-2 backbone with patch reprogramming,
> while the MM-TSFlib port combined an Autoformer-style numerical encoder with attention-pooled
> frozen BERT representations. Our independent PostTime reproduction used a frozen TimesFM-2.5
> prior and a 4-bit Gemma-3-4B reviser trained by LoRA SFT followed by GRPO with a clipped
> improvement-ratio reward. Checkpoints were selected without test feedback, and final sMAE and
> sRMSE were averaged from capped task-level normalized errors.

Table note: *Time-LLM and MM-TSFlib results use task-interface ports. PostTime is an independent
reproduction because official training code and checkpoints were unavailable at implementation
time. Results therefore test the published recipes under a shared protocol rather than claiming
bitwise reproduction of the original papers.*

## References and limitations

- [Time-LLM official repository](https://github.com/KimMeen/Time-LLM)
- [MM-TSFlib official repository](https://github.com/AdityaLab/MM-TSFlib)
- [Time-MMD](https://arxiv.org/abs/2406.08627)
- [PostTime](https://arxiv.org/abs/2605.29401)
- [TimesFM](https://github.com/google-research/timesfm)

The task-interface ports can differ from the original repositories in preprocessing, batching,
and software version. PostTime's trace construction cannot reproduce unreleased proprietary
trace-generation data. The benchmark uses one declared configuration per method and does not
perform an unreported hyperparameter sweep.

Time-LLM source is Apache-2.0 and MM-TSFlib source is MIT. The checked-out Time-MMD and TimesX
dataset snapshots do not contain top-level license files; their upstream dataset cards and source
terms must therefore be reviewed before redistributing data. Dr-CiK use and redistribution remain
subject to its Hugging Face dataset card. This repository stores only adapters, split manifests,
and derived metrics—not copies of benchmark data.
