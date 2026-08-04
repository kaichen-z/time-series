# TimesFM 2.5 Backbone

The default numerical backbone is Google Research TimesFM 2.5 using the official
`timesfm` Python package and the `google/timesfm-2.5-200m-pytorch` checkpoint.

Official references:

- <https://github.com/google-research/timesfm>
- <https://huggingface.co/google/timesfm-2.5-200m-pytorch>
- <https://arxiv.org/abs/2310.10688>

## Installation

```bash
pip install -e '.[timesfm]'
```

The model is loaded lazily when the first task requests a baseline. The first online run
may download the checkpoint from Hugging Face. To prepare an offline run, download/cache
the checkpoint in advance and use:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --timesfm-cache-dir /path/to/model-cache \
  --timesfm-local-files-only
```

## Adapter behavior

The adapter follows the official TimesFM 2.5 API:

```text
TimesFM_2p5_200M_torch.from_pretrained(model_id)
    -> compile(ForecastConfig)
    -> forecast(horizon, [history_values])
    -> point forecast becomes immutable y_baseline
```

Default forecast flags are:

```text
max_context = 4096
max_horizon = 1024
normalize_inputs = true
use_continuous_quantile_head = true
force_flip_invariance = true
infer_is_positive = true
fix_quantile_crossing = true
```

TimesFM 2.5 does not require the old frequency indicator. The Dr-CiK frequency remains
available to the diagnosis and contextual agents but is not passed to the 2.5 backbone.

The current agent system uses the TimesFM point forecast as `y_baseline`; probabilistic
trajectories are still generated with the repository's residual uncertainty layer. A
future experiment should replace that layer with direct sampling or interpolation from
TimesFM's returned quantiles.

## Failure and ablation policy

The default behavior is fail-fast. Missing packages, missing checkpoints, invalid model
outputs, and inference failures raise a `BackboneUnavailableError`.

There are two explicit alternatives:

```bash
# Controlled baseline ablation
--backbone statistical

# Operational degraded mode; output is visibly marked
--backbone timesfm --allow-statistical-fallback
```

Never report a run whose method begins with `statistical_fallback:` as a TimesFM result.
