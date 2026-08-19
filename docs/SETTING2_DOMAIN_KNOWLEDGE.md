# Setting 2: externally injected time-series domain knowledge

## What is injected

Setting 1 receives only the numeric history, horizon, frequency, and reusable self-generated
skills. With the explicit `--setting2-knowledge` flag, `statistics` and `combined` additionally
receive a source-backed knowledge selection from `evolving_loop/knowledge/time_series.json`.
The feature is opt-in so the current upstream defaults remain unchanged.

The library currently contains 90 falsifiable entries across 28 categories, grounded in 48
primary or authoritative sources:

- causal evaluation and residual diagnostics;
- naïve, seasonal-naïve, drift, mean, and exponential-smoothing baselines;
- Holt/local trend, damping, Theta-style blends, and long-horizon shrinkage;
- AR/differencing, seasonal differencing, decomposition, Fourier, and multiple seasonality;
- robust level/slope estimation, level shifts, and changepoint mixtures;
- direct historical analogue continuation, normalized-shape neighbors, and analogue abstention;
- variance stabilization, nonnegative/bounded support, and intermittent demand;
- forecast combination and validation-derived hypothesis weights;
- state-space, Bayesian structural, and Gaussian-process priors;
- TSFM transfer, complementarity, and long-horizon safeguards for `combined`.
- automatic ETS/ARIMA selection, direct-versus-recursive multi-step forecasting, and horizon-wise validation;
- robust STL/MSTL, PELT segmentation, changepoint-penalty ensembles, and regime mixtures;
- TSB obsolescence, terminal zero-run diagnostics, and temporal aggregation for intermittent demand;
- trimmed and horizon-dependent ensembles, disagreement-based abstention, and bagged decomposition;
- block-bootstrap trajectories, EnbPI, heteroscedastic conformal intervals, and proper scoring rules;
- executable structural lessons from N-BEATS, N-HiTS, DeepAR, TFT, and ES-RNN.

Each entry has a stable ID, a principle, use/avoid conditions, executable guidance, diagnostic
tags, priority, and one or more primary/authoritative sources. The knowledge is external to the
LLM and external to task documents; it never sees labels or ground-truth evidence.

## Research basis

The core statistical workflow follows Hyndman and Athanasopoulos, *Forecasting: Principles and
Practice* (3rd ed.): [rolling-origin validation](https://otexts.com/fpp3/tscv.html),
[transformations](https://otexts.com/fpp3/transformations.html),
[time-series components](https://otexts.com/fpp3/components.html),
[decomposition forecasting](https://otexts.com/fpp3/forecasting-decomposition.html),
[exponential smoothing](https://otexts.com/fpp3/ses.html),
[ARIMA and differencing](https://otexts.com/fpp3/arima.html),
[residual diagnostics](https://otexts.com/fpp3/diagnostics.html),
[forecast combinations](https://otexts.com/fpp3/combinations.html), and
[short-series constraints](https://otexts.com/fpp3/long-short-ts.html).

Specialized entries are grounded in the original or canonical sources:

- Assimakopoulos and Nikolopoulos, [Theta method](https://doi.org/10.1016/S0169-2070(00)00066-2);
- Syntetos and Boylan, [bias correction for intermittent demand](https://doi.org/10.1016/S0925-5273(00)00143-2);
- De Livera, Hyndman, and Snyder, [complex and multiple seasonality](https://doi.org/10.1198/jasa.2011.tm09771);
- Adams and MacKay, [Bayesian online changepoint detection](https://arxiv.org/abs/0710.3742);
- Durbin and Koopman, [state-space time-series methods](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001);
- Scott and Varian, [Bayesian structural time series](https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf);
- Rasmussen and Williams, [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/);
- Makridakis et al., [M4 Competition](https://doi.org/10.1016/j.ijforecast.2019.04.014);
- Farmer and Sidorowich, [local state-space analogue prediction](https://doi.org/10.1103/PhysRevLett.59.845);
- Lall and Sharma, [nearest-neighbor time-series resampling](https://doi.org/10.1029/95WR02966);
- Ansari et al., [Chronos](https://openreview.net/forum?id=gerNCVqqtR);
- Das et al., [TimesFM](https://arxiv.org/abs/2310.10688).
- Hyndman and Khandakar, [automatic ETS/ARIMA forecasting](https://www.jstatsoft.org/article/view/v027i03);
- Cleveland et al., [STL](https://doi.org/10.1080/01621459.1990.10475398), and Bandara et al., [MSTL](https://arxiv.org/abs/2107.13462);
- Killick et al., [PELT changepoint detection](https://arxiv.org/abs/1101.1438);
- Teunter et al., [TSB intermittent-demand forecasting](https://doi.org/10.1016/j.ejor.2011.05.018);
- Ben Taieb et al., [multi-step forecasting strategies](https://arxiv.org/abs/1108.3259);
- Xu and Xie, [EnbPI](https://proceedings.mlr.press/v139/xu21h.html), and Romano et al., [conformalized quantile regression](https://arxiv.org/abs/1905.03222);
- Gneiting and Raftery, [strictly proper scoring rules](https://doi.org/10.1198/016214506000001437);
- Oreshkin et al., [N-BEATS](https://openreview.net/forum?id=r1ecqn4YwB), Challu et al., [N-HiTS](https://arxiv.org/abs/2201.12886), Lim et al., [TFT](https://arxiv.org/abs/1912.09363), Salinas et al., [DeepAR](https://arxiv.org/abs/1704.04110), and Smyl, [ES-RNN](https://arxiv.org/abs/1902.01343).

The library deliberately stores operational, falsifiable implications rather than summaries of
model names. For example, it does not say merely “use Holt”; it states the numeric evidence needed,
the failure condition, an executable approximation, and the historical test that can reject it.

## How retrieval works

Before program generation, the host computes a deterministic profile using only the visible
history and task metadata:

- history length, horizon, and horizon/history ratio;
- declared seasonal period and autocorrelation at that period;
- lag-one autocorrelation and strongest empirical cycle candidates;
- robust recent slope, adjacent-window level shift, and slope change;
- median/MAD outlier fraction and early/recent difference-variance ratio;
- zero fraction, nonnegative support, and possible 0--100 bounds.
- long-history/hindcast availability, random-walk-like increments, volatility, count-like support,
  dynamic range, and independently supported multiple cycles.

Non-finite gaps are linearly interpolated only inside this diagnostic profile (with nearest-finite
boundary fill); the original history passed to generated forecasts is not modified.

These measurements produce tags such as `long_horizon`, `seasonality_supported`,
`weak_seasonal_evidence`, `recent_level_shift`, `outliers`, `intermittent`, and
`high_persistence`. The retriever ranks matching entries and returns at most ten, with a two-entry
cap per category before filling remaining slots. TSFM and neural-prior entries are available only in `combined`.
Thus the complete library is broad, while the per-task prompt remains small and relevant.

## How it realizes proposal811 Setting 2

The Coding Agent must generate structurally different executable hypotheses. For every hypothesis
it returns:

- a falsifiable assumption and failure condition;
- cited knowledge entry IDs;
- an initial confidence in `[0,1]`;
- executable `forecast(history, horizon, frequency)` code.

The Setting 2 candidate pool is explicitly additive: it retains the same unconditioned hypotheses
available to Setting 1 and adds a second set generated from the retrieved external knowledge. This
prevents knowledge injection from accidentally deleting a useful model-free hypothesis; the
existing rolling validation and Decision Agent still choose among all executed trajectories.

The host executes every program in the existing sandbox and evaluates it on the existing causal
rolling folds. It then converts relative hindcast error into normalized validation confidence and
ranks candidates by hindcast sMAPE. The Retrieval Agent receives the ranked numeric hypotheses,
their cited domain rules, prior confidence, and validation confidence; it then searches task
documents for evidence that distinguishes them. The Decision Agent receives the same fields but
can only choose an executed trajectory. This implements the requested chain:

`external statistical knowledge -> multiple assumptions -> executable skills -> historical
falsification and confidence ranking -> retrieval conditioned on assumptions -> constrained
decision`.

The `tsfm` variant remains a pure fixed-backbone ablation. The `combined` variant receives both the
same external knowledge and the unchanged Chronos candidate, so generated statistical programs can
complement rather than imitate the backbone.

## Evaluation contract

Knowledge value is isolated with `llm_only` versus `statistics` under identical task order, Codex
model, reasoning effort, number of generated programs, mutation count, validation folds, Retrieval
Agent, and Decision Agent. The official public-dev records can be loaded with
`--tasks-file hf://ServiceNow/Dr-CiK/public_dev`; `--task-manifest` preserves the exact frozen task
order. Result JSONL records the knowledge version, diagnostics, selected entries, hypothesis
citations, prior confidence, and validation confidence for later analysis.
