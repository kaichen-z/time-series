# TSFM runtime matrix

This page is the execution truth table for the 31 foundation cards in
`numerical_agent/datasets/forecast_method_dataset_v002.json`, as of the 2026-08-18
catalog cutoff. The checked-in runtime manifest is authoritative for machine-readable
bindings.

The states have deliberately narrow meanings:

- **Direct / verified at code level** means the card is connected to the in-process Chronos or
  TimesFM 2.5 adapter and that adapter has passed injected-backend tests. It does **not**
  mean that this repository has loaded the real checkpoint.
- **Worker / `experimental_unverified`** means the card is connected to an isolated
  worker adapter and has passed no-download tests with injected model backends. It does
  **not** mean that the locked environment has installed successfully or that the real
  checkpoint has passed smoke.
- **Unavailable** means the card cannot satisfy the generic local
  `history -> forecast` contract. The reason is specific to that card; it is not a
  judgment on the research idea.

There are exactly **4 direct**, **17 worker**, and **10 unavailable** cards. No real
official-checkpoint smoke result is checked in yet, for either direct or worker cards.
Consequently, none of the 31 cards is claimed as real-checkpoint verified or as an
experimental baseline result.

## All 31 foundation cards

| ID | Catalog card | Exact checkpoint/API | Execution binding | Status | License or exact unavailable reason |
|---|---|---|---|---|---|
| `method_tsfm_0001` | TimesFM 1.0 | `google/timesfm-1.0-200m-pytorch` | `legacy` in `timesfm_v1` | Worker: `experimental_unverified` | Apache-2.0 |
| `method_tsfm_0002` | Chronos T5 | `amazon/chronos-t5-base` | direct `chronos` | Direct / verified at code level | Apache-2.0 |
| `method_tsfm_0003` | Moirai 1.x | `Salesforce/moirai-1.1-R-base` | `uni2ts` in `uni2ts` | Worker: `experimental_unverified`; gated | `CC-BY-NC-4.0` acknowledgement required |
| `method_tsfm_0004` | Lag-Llama | `time-series-foundation-models/Lag-Llama` / `lag-llama.ckpt` | `legacy` in `lag_llama` | Worker: `experimental_unverified` | Apache-2.0 |
| `method_tsfm_0005` | MOMENT-1 | `AutonLab/MOMENT-1-large` | none | Unavailable: `forecast_head_requires_training` | The normal forecasting head is randomly initialized and requires training; only narrowly scoped reconstruction forecasting is zero-shot. |
| `method_tsfm_0006` | Granite Tiny Time Mixer R2 | `ibm-granite/granite-timeseries-ttm-r2`, model revision `512-96-ft-r2.1` | `granite` in `granite_tsfm` | Worker: `experimental_unverified` | Apache-2.0 |
| `method_tsfm_0007` | Timer | `thuml/timer-base-84m` | `transformer_generation` in `timer_legacy` | Worker: `experimental_unverified` | Apache-2.0 weights; MIT code |
| `method_tsfm_0008` | Time-MoE | `Maple728/TimeMoE-200M` | `transformer_generation` in `transformers_recent` | Worker: `experimental_unverified` | Apache-2.0 |
| `method_tsfm_0009` | ForecastPFN | `google-drive:1acp5thS7I4g_6Gw40wNFGnU1Sx14z0cU` | none | Unavailable: `official_checkpoint_missing` | The sole official checkpoint download currently returns HTTP 404. |
| `method_tsfm_0010` | TimeGPT-1 | Nixtla TimeGPT API | none | Unavailable: `no_public_local_weights` | Hosted proprietary API; no local public weights. |
| `method_tsfm_0011` | TEMPO | `Melady/TEMPO` / `TEMPO-80M_v1.pth` | `legacy` in `tempo_legacy` | Worker: `experimental_unverified` | Apache-2.0 weights; MIT code |
| `method_tsfm_0012` | UniTS | `mims-harvard/UniTS` | none | Unavailable: `no_generic_zero_shot_api` | The official path is task-YAML/DDP experiment code; arbitrary new-dataset zero-shot forecasting requires a specially trained version. |
| `method_tsfm_0013` | Sundial | `thuml/sundial-base-128m` | `transformer_generation` in `timer_legacy` | Worker: `experimental_unverified` | Apache-2.0 weights; MIT code |
| `method_tsfm_0014` | Toto 2.0 | `Datadog/Toto-2.0-22m` | `dedicated` in `toto2` | Worker: `experimental_unverified` | Apache-2.0 |
| `method_tsfm_0015` | Timer-S1 | `thuml/Timer-S1` | `transformer_generation` in `transformers_recent` | Worker: `experimental_unverified` | Apache-2.0; H100-class GPU recommended |
| `method_tsfm_0016` | Chronos-2 | `amazon/chronos-2` | direct `chronos` | Direct / verified at code level | Apache-2.0 |
| `method_tsfm_0017` | Moirai 2.0 | `Salesforce/moirai-2.0-R-small` | `uni2ts` in `uni2ts` | Worker: `experimental_unverified`; gated | `CC-BY-NC-4.0` acknowledgement required |
| `method_tsfm_0018` | Chronos-Bolt | `amazon/chronos-bolt-base` | direct `chronos` | Direct / verified at code level | Apache-2.0 |
| `method_tsfm_0019` | Moirai-MoE | `Salesforce/moirai-moe-1.0-R-small` | `uni2ts` in `uni2ts` | Worker: `experimental_unverified`; gated | `CC-BY-NC-4.0` acknowledgement required |
| `method_tsfm_0020` | FlowState | `ibm-research/flowstate`, revision `r1.1` | `granite` in `granite_tsfm` | Worker: `experimental_unverified`; gated | `research/non-commercial; official terms ambiguous` acknowledgement required |
| `method_tsfm_0021` | Xihe | Xihe research family | none | Unavailable: `no_public_local_weights` | No public consumable checkpoint. |
| `method_tsfm_0022` | Kairos | `mldi-lab/Kairos_50m` | `transformer_generation` in `kairos` | Worker: `experimental_unverified` | Apache-2.0 |
| `method_tsfm_0023` | TimeFound | TimeFound research model | none | Unavailable: `no_public_local_weights` | No public consumable checkpoint. |
| `method_tsfm_0024` | Reverso | Reverso research family | none | Unavailable: `no_public_local_weights` | No public consumable checkpoint. |
| `method_tsfm_0025` | Falcon-X | `Falcon-X` | none | Unavailable: `no_public_local_weights` | No public consumable checkpoint. |
| `method_tsfm_0026` | SEMPO | `mala-lab/SEMPO` | none | Unavailable: `dataset_specific_cli_only` | The release has dataset-specific CLI evaluation and repository-relative checkpoints, but no supported generic history-to-forecast API. |
| `method_tsfm_0027` | TiRex | `NX-AI/TiRex` | `dedicated` in `tirex` | Worker: `experimental_unverified`; gated | `NXAI Community License` acknowledgement required |
| `method_tsfm_0028` | TiRex-2 | TiRex-2 research model | none | Unavailable: `no_public_local_weights` | No public consumable checkpoint. |
| `method_tsfm_0029` | TabPFN-TS | `Prior-Labs/tabpfn_3` / `tabpfn-v3-regressor-v3_20260506_timeseries.ckpt` | `dedicated` in `tabpfn_ts`, forced `LOCAL` mode | Worker: `experimental_unverified`; gated | `TabPFN-3 Non-Commercial License; Apache-2.0 code` acknowledgement and upstream access acceptance required |
| `method_tsfm_0030` | PatchTST-FM | `ibm-research/patchtst-fm-r1` | `granite` in `granite_tsfm` | Worker: `experimental_unverified`; gated | `CC-BY-NC-SA-4.0` acknowledgement required |
| `method_tsfm_0031` | TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | direct `timesfm` | Direct / verified at code level | Apache-2.0 |

## Isolated environments

The 17 worker cards are split across exactly 11 reviewed Linux x86_64 lock sets.
Setup checks the stated CPython minor before it creates an environment.

| Environment key | CPython | Adapter family | Cards |
|---|---:|---|---|
| `timesfm_v1` | 3.11 | `legacy` | TimesFM 1.0 |
| `uni2ts` | 3.11 | `uni2ts` | Moirai 1.x, Moirai 2.0, Moirai-MoE |
| `lag_llama` | 3.10 | `legacy` | Lag-Llama |
| `granite_tsfm` | 3.11 | `granite` | Granite TTM R2, FlowState, PatchTST-FM |
| `timer_legacy` | 3.10 | `transformer_generation` | Timer, Sundial |
| `transformers_recent` | 3.11 | `transformer_generation` | Time-MoE, Timer-S1 |
| `tempo_legacy` | 3.10 | `legacy` | TEMPO |
| `toto2` | 3.12 | `dedicated` | Toto 2.0 |
| `kairos` | 3.11 | `transformer_generation` | Kairos |
| `tirex` | 3.11 | `dedicated` | TiRex |
| `tabpfn_ts` | 3.11 | `dedicated` | TabPFN-TS |

Each `.in` file records reviewed direct inputs, each `.txt` file is a complete
hash-checked PyPI closure, and the Kairos/Lag-Llama `.vcs` files pin their official
repositories to immutable commits. Validate the locks offline with:

```bash
.venv/bin/python scripts/validate_tsfm_locks.py
```

## Exact license acknowledgements

Gated cards are disabled unless the deployment explicitly supplies the exact relevant
identifier. Across all 17 workers, the five accepted acknowledgement strings are:

```text
CC-BY-NC-4.0
research/non-commercial; official terms ambiguous
NXAI Community License
TabPFN-3 Non-Commercial License; Apache-2.0 code
CC-BY-NC-SA-4.0
```

For a deployment whose operator has independently reviewed and accepted all applicable
terms, the comma-separated CLI value is:

```text
CC-BY-NC-4.0,research/non-commercial; official terms ambiguous,NXAI Community License,TabPFN-3 Non-Commercial License; Apache-2.0 code,CC-BY-NC-SA-4.0
```

This local acknowledgement only enables the broker gate. It does not accept upstream
terms, grant a license, or obtain access on the operator's behalf. In particular,
TabPFN-TS may still require an already accepted upstream account/credential or cached
weight. Keep credentials in the execution environment; never place tokens in the worker
JSON, shell command line, repository, report path, or smoke output.

## What smoke verification proves

The real-checkpoint smoke runner resolves an immutable 40--64 character checkpoint
revision before starting the broker, sends only a fixed 96-value hourly history with
horizon 4, and requires the worker to report the same post-load identity. A successful
report also requires package versions, actual device, latency, peak memory, exact output
length, and finite values. It stores none of the input values, forecast values,
credentials, deployment paths, or benchmark labels.

For loaders that do not return an independent model digest, the attestation is the exact
immutable revision passed to the loader, or the exact-revision local artifact path loaded
by the adapter, recorded only after loading succeeds. It is not an independent hash of
the in-memory model.

No such real-checkpoint smoke has been run in this repository yet. Adapter code and
no-download tests alone cannot change `experimental_unverified` into a verified baseline.
