from __future__ import annotations

import pytest

from numerical_agent.evolution.identity import (
    IdentityError,
    identity_contract,
    validate_repair,
)


CURATED_EXAMPLES = {
    "sarima_auto": '''def sarima_auto(history, horizon, frequency):
    """Use for seasonal SARIMA selection."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    best = None
    best_aic = float("inf")
    for p in range(2):
        model = SARIMAX(history, order=(p, 0, 0), seasonal_order=(1, 0, 0, 7))
        result = model.fit(disp=False)
        if result.aic < best_aic:
            best_aic = result.aic
            best = result
    return list(best.forecast(steps=horizon))
''',
    "negative_binomial_dynamic_regression": '''def negative_binomial_dynamic_regression(history, horizon, frequency):
    """Use for overdispersed counts."""
    from scipy.optimize import minimize
    from scipy.special import gammaln
    import math
    def negative_log_likelihood(parameters):
        alpha = math.exp(parameters[-1])
        size = 1.0 / alpha
        log_likelihood = gammaln(history[-1] + size) - gammaln(size)
        return -log_likelihood
    fitted = minimize(negative_log_likelihood, [0.0])
    beta = fitted.x[0]
    previous = float(history[-1])
    forecasts = []
    for _ in range(horizon):
        previous = math.exp(math.log1p(previous) + beta)
        forecasts.append(previous)
    return forecasts
''',
    "smooth_transition_autoregression": '''def smooth_transition_autoregression(history, horizon, frequency):
    """Use for smoothly changing autoregressive regimes."""
    from scipy.optimize import least_squares
    from scipy.special import expit
    def residual_function(parameters):
        base = parameters[0]
        regime_difference = parameters[1]
        threshold = parameters[2]
        transition = expit(history[-1] - threshold)
        fitted = base + transition * regime_difference
        return [history[-1] - fitted]
    fitted = least_squares(residual_function, [0.0, 0.0, 0.0]).x
    base, regime_difference, threshold = fitted
    forecasts = []
    previous = history[-1]
    for _ in range(horizon):
        transition = expit(previous - threshold)
        previous = base + transition * regime_difference
        forecasts.append(previous)
    return forecasts
''',
    "scinet": '''def scinet(history, horizon, frequency):
    """Use for interleaved neural interactions."""
    import torch
    import torch.nn.functional as F
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = torch.nn.Parameter(torch.zeros(3))
            self.head = torch.nn.Parameter(torch.ones(1))
        def convolve(self, values):
            return F.conv1d(values, self.kernel.reshape(1, 1, 3))
        def interact(self, values, depth):
            return values if depth else self.interact(values, depth + 1)
        def forward(self, values):
            return self.interact(values, 0).sum() * self.head
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())
    loss = model(torch.tensor(history).reshape(1, 1, -1))
    loss.backward()
    optimizer.step()
    forecasts = []
    for _ in range(horizon):
        forecasts.append(float(model(torch.tensor(history).reshape(1, 1, -1))))
    return forecasts
''',
    "itransformer": '''def itransformer(history, horizon, frequency):
    """Use for inverted-variate attention."""
    import numpy as np
    def encode(embedded):
        query_matrix = np.eye(2)
        key_matrix = np.eye(2)
        value_matrix = np.eye(2)
        queries = embedded @ query_matrix
        keys = embedded @ key_matrix
        scores = queries @ keys.T
        attention = np.exp(scores)
        attention = attention / attention.sum(axis=1, keepdims=True)
        attended = attention @ (embedded @ value_matrix)
        return attended.ravel()
    feature = encode(np.asarray(history[-2:])[None, :])
    coefficients = np.linalg.solve(np.eye(2), feature)
    forecasts = []
    for _ in range(horizon):
        forecasts.append(float(coefficients[-1]))
    return forecasts
''',
    "ltsf_dlinear": '''def ltsf_dlinear(history, horizon, frequency):
    """Use for decomposed linear trend and remainder projections."""
    import numpy as np
    values = np.asarray(history)
    trend = np.convolve(values, np.ones(3) / 3, mode="same")
    remainder = values - trend
    weights_trend = np.linalg.lstsq(trend[:, None], values, rcond=None)[0]
    weights_remainder = np.linalg.lstsq(remainder[:, None], values, rcond=None)[0]
    forecast = trend[-1] * weights_trend + remainder[-1] * weights_remainder
    return [float(forecast[0])] * horizon
''',
}


@pytest.mark.parametrize("name", tuple(CURATED_EXAMPLES))
def test_curated_identity_contract_accepts_a_structural_implementation(name: str) -> None:
    source = CURATED_EXAMPLES[name]
    validate_repair(name, source, source, identity_contract(name, source))


def test_sarima_contract_rejects_dead_identity_words_around_a_naive_forecast() -> None:
    replacement = '''def sarima_auto(history, horizon, frequency):
    """Pretends to be SARIMA while returning a naive forecast."""
    sarimax = seasonal_order = aic = None
    return [float(history[-1])] * horizon
'''

    with pytest.raises(IdentityError, match="identity|structure"):
        validate_repair(
            "sarima_auto",
            CURATED_EXAMPLES["sarima_auto"],
            replacement,
            identity_contract("sarima_auto", CURATED_EXAMPLES["sarima_auto"]),
        )


def test_unknown_method_has_no_repair_contract_and_must_fork() -> None:
    parent = '''def custom_model(history, horizon, frequency):
    """Use for a custom optimized model."""
    from scipy.optimize import minimize
    result = minimize(lambda x: (x[0] - history[-1]) ** 2, [0.0])
    return [float(result.x[0])] * horizon
'''

    with pytest.raises(IdentityError, match="no verified identity contract|fork"):
        validate_repair("custom_model", parent, parent, identity_contract("custom_model", parent))

    payload = identity_contract("custom_model", parent).payload()
    assert payload["repair_allowed"] is False
    assert payload["mode"] == "fork_only"


def test_sarima_contract_rejects_a_complete_but_unreachable_scaffold() -> None:
    child = '''def sarima_auto(history, horizon, frequency):
    """Pretends to preserve SARIMA in dead code."""
    if False:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        best = None
        best_aic = float("inf")
        for p in range(2):
            model = SARIMAX(history, order=(p, 0, 0), seasonal_order=(1, 0, 0, 7))
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best = result
        forecast = best.forecast(steps=horizon)
    return [float(history[-1])] * horizon
'''

    with pytest.raises(IdentityError, match="SARIMA|output"):
        validate_repair(
            "sarima_auto",
            CURATED_EXAMPLES["sarima_auto"],
            child,
            identity_contract("sarima_auto", CURATED_EXAMPLES["sarima_auto"]),
        )


def test_sarima_contract_rejects_ignored_live_model_output() -> None:
    child = CURATED_EXAMPLES["sarima_auto"].replace(
        "return list(best.forecast(steps=horizon))",
        "ignored = best.forecast(steps=horizon)\n    return [float(history[-1])] * horizon",
    )

    with pytest.raises(IdentityError, match="output"):
        validate_repair(
            "sarima_auto",
            CURATED_EXAMPLES["sarima_auto"],
            child,
            identity_contract("sarima_auto", CURATED_EXAMPLES["sarima_auto"]),
        )


def test_sarima_contract_rejects_a_model_output_overwritten_by_naive() -> None:
    child = CURATED_EXAMPLES["sarima_auto"].replace(
        "return list(best.forecast(steps=horizon))",
        "forecast = best.forecast(steps=horizon)\n"
        "    forecast = [float(history[-1])] * horizon\n"
        "    return forecast",
    )

    with pytest.raises(IdentityError, match="output"):
        validate_repair(
            "sarima_auto",
            CURATED_EXAMPLES["sarima_auto"],
            child,
            identity_contract("sarima_auto", CURATED_EXAMPLES["sarima_auto"]),
        )


def test_sarima_contract_rejects_an_unconditional_constant_forecast() -> None:
    child = CURATED_EXAMPLES["sarima_auto"].replace(
        "return list(best.forecast(steps=horizon))",
        "best.forecast(steps=horizon)\n    return [0.0] * horizon",
    )

    with pytest.raises(IdentityError, match="output"):
        validate_repair(
            "sarima_auto",
            CURATED_EXAMPLES["sarima_auto"],
            child,
            identity_contract("sarima_auto", CURATED_EXAMPLES["sarima_auto"]),
        )


def test_sarima_contract_rejects_a_method_with_no_return() -> None:
    child = CURATED_EXAMPLES["sarima_auto"].replace(
        "return list(best.forecast(steps=horizon))",
        "best.forecast(steps=horizon)",
    )

    with pytest.raises(IdentityError, match="return"):
        validate_repair(
            "sarima_auto",
            CURATED_EXAMPLES["sarima_auto"],
            child,
            identity_contract("sarima_auto", CURATED_EXAMPLES["sarima_auto"]),
        )
