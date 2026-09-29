"""Forecasting methods evolved from a verified method catalog.

Each function takes (history, horizon, frequency) and returns exactly horizon finite floats,
or raises NotApplicable when the series does not meet its stated requirements.
"""


class NotApplicable(Exception):
    """Raised when the series does not meet the method's stated requirements."""


def naive_last(history, horizon, frequency):
    """Use when the latest observation is the best near-future estimate and no reliable trend or seasonality is present."""
    if len(history) < 1:
        raise NotApplicable(f"needs 1 point, got {len(history)}")
    return [float(history[-1])] * horizon


def naive_mean(history, horizon, frequency):
    """Use when the series is stationary around a constant mean without trend or seasonality."""
    import math
    import statistics
    if len(history) < 1:
        raise NotApplicable(f"needs at least 1 point, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if not all(math.isfinite(float(value)) for value in history):
        raise ValueError("history must contain only finite values")
    mean = float(statistics.fmean(history))
    return [mean] * horizon


def naive_drift(history, horizon, frequency):
    """Use when a series has a stable long-term linear trend and at least two observations."""
    if len(history) < 2:
        raise NotApplicable(f"needs at least 2 observations, got {len(history)}")
    drift = (float(history[-1]) - float(history[0])) / (len(history) - 1)
    return [float(history[-1]) + drift * step for step in range(1, horizon + 1)]


def seasonal_naive(history, horizon, frequency):
    """Use when observations have a known, stable seasonal cycle and little meaningful trend."""
    periods = {
        "1 minute": 1440,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    if frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    period = periods[frequency]
    if len(history) < period:
        raise NotApplicable(f"needs {period} points, got {len(history)}")
    return [float(history[-period + (step % period)]) for step in range(horizon)]


def simple_moving_average(history, horizon, frequency):
    """Use when recent observations fluctuate around a locally stable level and equal weighting over a short window is appropriate."""
    window_size = 3
    if len(history) < window_size:
        raise NotApplicable(f"needs {window_size} points, got {len(history)}")
    forecast = float(sum(history[-window_size:]) / window_size)
    return [forecast] * horizon


def ses(history, horizon, frequency):
    """Use when the series has no trend or seasonality and recent observations should receive more weight."""
    import numpy as np
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    if len(history) < 3:
        raise NotApplicable(f"needs 3 points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if not all(np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    fitted = SimpleExpSmoothing(
        values,
        initialization_method="estimated",
    ).fit(optimized=True)
    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if forecast.size != horizon or not np.all(np.isfinite(forecast)):
        raise RuntimeError("simple exponential smoothing produced an invalid forecast")
    return [float(value) for value in forecast]


def holt_linear_trend(history, horizon, frequency):
    """Use when the series has an approximately linear trend and no seasonality."""
    import numpy as np
    from statsmodels.tsa.holtwinters import Holt

    if len(history) < 2:
        raise NotApplicable(f"needs at least 2 points, got {len(history)}")
    if not np.all(np.isfinite(history)):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    fitted = Holt(values, exponential=False, damped_trend=False, initialization_method="estimated").fit(optimized=True)
    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if not np.all(np.isfinite(forecast)):
        raise ValueError("Holt linear trend produced non-finite forecasts")
    return [float(value) for value in forecast]


def holt_damped_trend(history, horizon, frequency):
    """Use when a non-seasonal series has a trend expected to weaken over the forecast horizon."""
    import numpy as np
    from statsmodels.tsa.holtwinters import Holt

    if len(history) < 3:
        raise NotApplicable(f"needs 3 points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")
    if horizon == 0:
        return []

    fitted = Holt(
        values,
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True)
    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if forecast.size != horizon or not np.all(np.isfinite(forecast)):
        raise ValueError("Holt damped trend produced an invalid forecast")
    return [float(value) for value in forecast]


def holt_winters_additive(history, horizon, frequency):
    """Use when the series has a stable trend and seasonal fluctuations of roughly constant absolute size."""
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported frequency of 1 hour, 1 day, 1 week, 1 month, or 1 quarter, got {frequency}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")

    model = ExponentialSmoothing(
        values,
        trend="add",
        seasonal="add",
        seasonal_periods=period,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if forecast.size != horizon or not np.all(np.isfinite(forecast)):
        raise FloatingPointError("Holt-Winters produced an invalid forecast")
    return [float(value) for value in forecast]


def holt_winters_multiplicative(history, horizon, frequency):
    """Use when a strictly positive seasonal series has fluctuations whose amplitude grows or shrinks with its level."""
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency!r}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if any(value <= 0 for value in history):
        raise NotApplicable("needs strictly positive values, got zero or negative values")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    model = ExponentialSmoothing(
        values,
        trend="add",
        seasonal="mul",
        seasonal_periods=period,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if forecast.shape != (horizon,) or not np.all(np.isfinite(forecast)):
        raise ValueError("Holt-Winters produced an invalid forecast")
    return forecast.tolist()


def holt_winters_damped(history, horizon, frequency):
    """Use when data have recurring seasonality and a trend expected to weaken over the forecast horizon."""
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal_periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in seasonal_periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    period = seasonal_periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    model = ExponentialSmoothing(
        values,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=period,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if not np.all(np.isfinite(forecast)):
        raise ValueError("forecast contains non-finite values")
    return [float(value) for value in forecast]


def ets_auto(history, horizon, frequency):
    """Use when a positive seasonal series may be described by an automatically selected ETS error-trend-season model."""
    import math
    import numpy as np
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    periods = {
        "1 second": 60,
        "1 minute": 60,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency.endswith("s"):
        normalized_frequency = normalized_frequency[:-1]
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if any((not math.isfinite(value)) or value <= 0 for value in history):
        raise NotApplicable("needs strictly positive finite values, got non-positive or non-finite values")

    values = np.asarray(history, dtype=float)
    best_result = None
    best_aic = float("inf")
    for error in ("add", "mul"):
        for trend in (None, "add", "mul"):
            damped_options = (False,) if trend is None else (False, True)
            for damped in damped_options:
                for seasonal in (None, "add", "mul"):
                    kwargs = {
                        "error": error,
                        "trend": trend,
                        "damped_trend": damped,
                        "seasonal": seasonal,
                        "initialization_method": "estimated",
                    }
                    if seasonal is not None:
                        kwargs["seasonal_periods"] = period
                    result = ETSModel(values, **kwargs).fit(disp=False)
                    aic = float(result.aic)
                    if aic < best_aic:
                        best_aic = aic
                        best_result = result

    forecast = np.asarray(best_result.forecast(horizon), dtype=float)
    if forecast.shape != (horizon,) or not np.all(np.isfinite(forecast)):
        raise ValueError("ETS forecast did not contain exactly horizon finite values")
    return [float(value) for value in forecast]


def theta_classic(history, horizon, frequency):
    """Use when a nonseasonal series is driven by a linear long-term trend plus locally weighted short-term movement."""
    import math
    import numpy as np
    from scipy.optimize import minimize_scalar

    if len(history) < 4:
        raise NotApplicable(f"needs at least 4 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if not all(math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    values = np.asarray(history, dtype=float)
    n = len(values)
    time = np.arange(n, dtype=float)
    centered_time = time - time.mean()
    slope = float(np.dot(centered_time, values - values.mean()) / np.dot(centered_time, centered_time))
    intercept = float(values.mean() - slope * time.mean())
    trend = intercept + slope * time
    theta_two = 2.0 * values - trend

    def ses_error(alpha):
        level = float(theta_two[0])
        error = 0.0
        for observation in theta_two[1:]:
            residual = float(observation) - level
            error += residual * residual
            level = alpha * float(observation) + (1.0 - alpha) * level
        return error

    result = minimize_scalar(ses_error, bounds=(0.0, 1.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"SES optimization failed: {result.message}")

    candidates = [(float(result.fun), float(result.x)), (ses_error(0.0), 0.0), (ses_error(1.0), 1.0)]
    alpha = min(candidates, key=lambda item: item[0])[1]
    level = float(theta_two[0])
    for observation in theta_two[1:]:
        level = alpha * float(observation) + (1.0 - alpha) * level

    future_time = np.arange(n, n + horizon, dtype=float)
    trend_forecast = intercept + slope * future_time
    forecast = 0.5 * (trend_forecast + level)
    output = [float(value) for value in forecast]
    if not all(math.isfinite(value) for value in output):
        raise ArithmeticError("Theta forecast produced non-finite values")
    return output


def theta_optimized(history, horizon, frequency):
    """Use when a non-seasonal series has a stable linear trend but its Theta weights and smoothing strength should be estimated from data."""
    import numpy as np
    from scipy.optimize import minimize

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise NotApplicable(f"needs finite history values, got {len(history) - int(np.isfinite(y).sum())} non-finite values")

    def trend_coefficients(values):
        x = np.arange(len(values), dtype=float)
        x_mean = x.mean()
        slope = np.dot(x - x_mean, values - values.mean()) / np.dot(x - x_mean, x - x_mean)
        return values.mean() - slope * x_mean, slope

    def objective(parameters):
        theta, alpha, trend_weight = parameters
        errors = []
        for end in range(3, len(y)):
            train = y[:end]
            intercept, slope = trend_coefficients(train)
            x = np.arange(end, dtype=float)
            trend = intercept + slope * x
            theta_line = theta * train + (1.0 - theta) * trend
            level = theta_line[0]
            for value in theta_line[1:]:
                level = alpha * value + (1.0 - alpha) * level
            trend_next = intercept + slope * end
            prediction = trend_weight * trend_next + (1.0 - trend_weight) * level
            errors.append(y[end] - prediction)
        errors = np.asarray(errors, dtype=float)
        return float(np.mean(errors * errors))

    result = minimize(
        objective,
        x0=np.array([2.0, 0.2, 0.5]),
        method="L-BFGS-B",
        bounds=((1.0, 5.0), (0.001, 0.999), (0.0, 1.0)),
    )
    if not result.success:
        raise RuntimeError(f"Theta optimization failed: {result.message}")

    theta, alpha, trend_weight = result.x
    intercept, slope = trend_coefficients(y)
    x = np.arange(len(y), dtype=float)
    trend = intercept + slope * x
    theta_line = theta * y + (1.0 - theta) * trend
    level = theta_line[0]
    for value in theta_line[1:]:
        level = alpha * value + (1.0 - alpha) * level

    future_x = np.arange(len(y), len(y) + horizon, dtype=float)
    future_trend = intercept + slope * future_x
    forecast = trend_weight * future_trend + (1.0 - trend_weight) * level
    if not np.all(np.isfinite(forecast)):
        raise RuntimeError("Theta optimization produced non-finite forecasts")
    return [float(value) for value in forecast]


def ar(history, horizon, frequency):
    """Use when a stationary series has a linear relationship with recent values and no dominant seasonality."""
    import numpy as np

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points, got {len(history)}")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")

    center = float(np.mean(values))
    scale = float(np.std(values))
    if scale == 0.0:
        scale = 1.0
    standardized = (values - center) / scale

    max_order = min(10, (len(values) - 2) // 3)
    best_order = None
    best_aic = float("inf")

    for order in range(1, max_order + 1):
        targets = standardized[max_order:]
        design = np.ones((len(targets), order + 1), dtype=float)
        for lag in range(1, order + 1):
            design[:, lag] = standardized[max_order - lag:len(standardized) - lag]
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        residuals = targets - design @ coefficients
        variance = max(float(np.mean(residuals ** 2)), np.finfo(float).tiny)
        aic = len(targets) * np.log(variance) + 2.0 * (order + 1)
        if aic < best_aic:
            best_aic = float(aic)
            best_order = order

    targets = standardized[best_order:]
    design = np.ones((len(targets), best_order + 1), dtype=float)
    for lag in range(1, best_order + 1):
        design[:, lag] = standardized[best_order - lag:len(standardized) - lag]
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]

    extended = standardized.tolist()
    forecasts = []
    for _ in range(horizon):
        lagged = np.asarray([extended[-lag] for lag in range(1, best_order + 1)])
        prediction = float(coefficients[0] + coefficients[1:] @ lagged)
        extended.append(prediction)
        forecasts.append(float(prediction * scale + center))

    if not np.all(np.isfinite(forecasts)):
        raise FloatingPointError("AR produced non-finite forecasts")
    return forecasts


def ma(history, horizon, frequency):
    """Use when a stationary series is driven by linear shocks whose effects disappear quickly."""
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    if len(history) < 5:
        raise NotApplicable(f"needs at least 5 points, got {len(history)}")
    if horizon < 1:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    if not all(np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    values = np.asarray(history, dtype=float)
    fitted = ARIMA(values, order=(0, 0, 1), trend="c").fit()
    forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float)
    if forecast.size != horizon or not np.all(np.isfinite(forecast)):
        raise ValueError("MA model produced an invalid forecast")
    return [float(value) for value in forecast]


def arma(history, horizon, frequency):
    """Use when the series is stationary, nonseasonal, and its values depend on both prior values and prior shocks."""
    import math
    from statsmodels.tsa.arima.model import ARIMA

    if len(history) < 5:
        raise NotApplicable(f"needs at least 5 points, got {len(history)}")
    if horizon < 1:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    if not all(math.isfinite(float(value)) for value in history):
        finite_count = sum(math.isfinite(float(value)) for value in history)
        raise NotApplicable(f"needs all history values to be finite, got {finite_count} finite values out of {len(history)}")

    model = ARIMA(
        [float(value) for value in history],
        order=(1, 0, 1),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    fitted = model.fit()
    forecast = [float(value) for value in fitted.forecast(steps=horizon)]
    if not all(math.isfinite(value) for value in forecast):
        raise ValueError("ARMA produced a non-finite forecast")
    return forecast


def arima_auto(history, horizon, frequency):
    """Use when a non-seasonal series can be made stationary with low-order differencing and its autoregressive and moving-average orders are unknown."""
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    if len(history) < 20:
        raise NotApplicable(f"needs 20 points, got {len(history)}")
    if horizon <= 0:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")

    best_result = None
    best_aicc = np.inf
    n = len(values)

    for d in range(3):
        for p in range(4):
            for q in range(4):
                result = ARIMA(
                    values,
                    order=(p, d, q),
                    trend="c" if d == 0 else "n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                parameter_count = len(result.params)
                denominator = n - parameter_count - 1
                if denominator > 0 and np.isfinite(result.aic):
                    aicc = result.aic + (2.0 * parameter_count * (parameter_count + 1)) / denominator
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_result = result

    if best_result is None:
        raise RuntimeError("ARIMA order search produced no finite information criterion")

    forecast = np.asarray(best_result.forecast(steps=horizon), dtype=float).reshape(-1)
    if forecast.size != horizon or not np.all(np.isfinite(forecast)):
        raise RuntimeError("ARIMA produced an invalid forecast")
    return forecast.tolist()


def sarima_auto(history, horizon, frequency):
    """Use when a series has known calendar seasonality and several complete cycles available for automatic SARIMA order selection."""
    import math
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported frequency with a known seasonal period, got {frequency!r}")
    period = periods[normalized_frequency]
    required = 4 * period
    if len(history) < required:
        raise NotApplicable(f"needs {required} points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    best_result = None
    best_aic = math.inf

    for p in range(3):
        for d in range(2):
            for q in range(3):
                for seasonal_p in range(2):
                    for seasonal_d in range(2):
                        for seasonal_q in range(2):
                            trend = "c" if d + seasonal_d == 0 else "n"
                            model = SARIMAX(
                                values,
                                order=(p, d, q),
                                seasonal_order=(seasonal_p, seasonal_d, seasonal_q, period),
                                trend=trend,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            )
                            result = model.fit(disp=False, maxiter=200)
                            if math.isfinite(float(result.aic)) and result.aic < best_aic:
                                best_aic = float(result.aic)
                                best_result = result

    if best_result is None:
        raise RuntimeError("SARIMA order search produced no finite information criterion")
    forecast = np.asarray(best_result.forecast(steps=horizon), dtype=float)
    if len(forecast) != horizon or not np.all(np.isfinite(forecast)):
        raise RuntimeError("SARIMA forecast was not finite or had the wrong length")
    return [float(value) for value in forecast]


def stl_naive(history, horizon, frequency):
    """Use when a series has a stable repeating seasonal pattern and at least two complete seasonal cycles."""
    import numpy as np
    from statsmodels.tsa.seasonal import STL

    periods = {
        "1 minute": 1440,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    if frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    period = periods[frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    result = STL(values, period=period, robust=False).fit()
    trend = result.trend
    seasonal_cycle = result.seasonal[-period:]
    drift = (trend[-1] - trend[0]) / (len(trend) - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    trend_forecast = trend[-1] + drift * steps
    seasonal_forecast = np.resize(seasonal_cycle, horizon)
    return [float(value) for value in trend_forecast + seasonal_forecast]


def classical_decomposition(history, horizon, frequency):
    """Use when a series has a stable seasonal pattern and at least two complete seasonal cycles."""
    import numpy as np
    from statsmodels.tsa.seasonal import seasonal_decompose

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
        "1 year": 1,
    }
    if frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    period = periods[frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    decomposition = seasonal_decompose(
        values,
        model="additive",
        period=period,
        extrapolate_trend="freq",
    )
    time = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(time, np.asarray(decomposition.trend), 1)
    future_time = np.arange(len(values), len(values) + horizon, dtype=float)
    future_trend = intercept + slope * future_time
    seasonal = np.asarray(decomposition.seasonal, dtype=float)
    future_seasonal = seasonal[np.arange(len(values), len(values) + horizon) % period]
    forecast = future_trend + future_seasonal
    return [float(value) for value in forecast]


def stl_ets(history, horizon, frequency):
    """Use when a sufficiently long seasonal series has a stable recurring pattern and a smoothly evolving seasonally adjusted level or trend."""
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import STL

    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    parts = str(frequency).strip().lower().split()
    units = {
        "second": 60,
        "seconds": 60,
        "minute": 60,
        "minutes": 60,
        "hour": 24,
        "hours": 24,
        "day": 7,
        "days": 7,
        "week": 52,
        "weeks": 52,
        "month": 12,
        "months": 12,
        "quarter": 4,
        "quarters": 4,
    }
    if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in units:
        raise NotApplicable(f"needs a supported frequency such as '1 hour' or '1 day', got {frequency!r}")

    interval = int(parts[0])
    base_period = units[parts[1]]
    if interval <= 0 or base_period % interval != 0 or base_period // interval < 2:
        raise NotApplicable(f"needs a frequency with an integral seasonal period of at least 2, got {frequency!r}")

    period = base_period // interval
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    if horizon == 0:
        return []

    decomposition = STL(values, period=period, robust=True).fit()
    adjusted = values - decomposition.seasonal
    ets = ExponentialSmoothing(
        adjusted,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    adjusted_forecast = np.asarray(ets.forecast(horizon), dtype=float)
    seasonal_forecast = np.resize(decomposition.seasonal[-period:], horizon)
    forecast = adjusted_forecast + seasonal_forecast
    if len(forecast) != horizon or not np.all(np.isfinite(forecast)):
        raise ValueError("STL-ETS produced a non-finite or incorrectly sized forecast")
    return [float(value) for value in forecast]


def stl_arima(history, horizon, frequency):
    """Use when a seasonal series has at least two cycles and its seasonally adjusted values are well modeled by ARIMA."""
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.forecasting.stl import STLForecast

    periods = {
        "1 minute": 60,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported seasonal frequency, got {frequency}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    fitted = STLForecast(
        values,
        ARIMA,
        model_kwargs={"order": (1, 1, 1)},
        period=period,
        robust=True,
    ).fit()
    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if forecast.size != horizon or not np.all(np.isfinite(forecast)):
        raise ValueError("STL-ARIMA produced a non-finite or incorrectly sized forecast")
    return [float(value) for value in forecast]


def croston(history, horizon, frequency):
    """Use when demand is intermittent, with exact zeros separating occasional positive observations."""
    import math

    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if len(history) < 2:
        raise NotApplicable(f"needs at least 2 points, got {len(history)}")
    if any(not math.isfinite(x) for x in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if any(x < 0 for x in history):
        raise NotApplicable("needs non-negative demand values, got negative values")
    if not any(x == 0 for x in history):
        raise NotApplicable(f"needs intermittent demand containing exact zeros, got 0 zeros in {len(history)} points")
    if not any(x > 0 for x in history):
        raise NotApplicable(f"needs at least one positive demand, got 0 in {len(history)} points")

    alpha = 0.1
    first = next(i for i, value in enumerate(history) if value > 0)
    demand_size = float(history[first])
    demand_interval = float(first + 1)
    previous = first

    for i in range(first + 1, len(history)):
        if history[i] > 0:
            interval = i - previous
            demand_size += alpha * (float(history[i]) - demand_size)
            demand_interval += alpha * (float(interval) - demand_interval)
            previous = i

    forecast = float(demand_size / demand_interval)
    return [forecast for _ in range(horizon)]


def croston_sba(history, horizon, frequency):
    """Use when demand is intermittent, nonnegative, and contains a meaningful fraction of exact zeros."""
    import math

    if horizon < 0:
        raise NotApplicable(f"needs a nonnegative horizon, got {horizon}")
    if len(history) < 4:
        raise NotApplicable(f"needs at least 4 points, got {len(history)}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if any(value < 0 for value in history):
        raise NotApplicable("needs nonnegative demand values, got negative values")

    zero_count = sum(value == 0 for value in history)
    if zero_count / len(history) < 0.2:
        raise NotApplicable(f"needs at least 20% exact zeros, got {zero_count} of {len(history)}")

    demand_indices = [index for index, value in enumerate(history) if value > 0]
    if len(demand_indices) < 2:
        raise NotApplicable(f"needs at least 2 nonzero demands, got {len(demand_indices)}")

    alpha = 0.1
    first_index = demand_indices[0]
    smoothed_demand = float(history[first_index])
    smoothed_interval = float(first_index + 1)
    previous_index = first_index

    for index in demand_indices[1:]:
        interval = float(index - previous_index)
        smoothed_demand += alpha * (float(history[index]) - smoothed_demand)
        smoothed_interval += alpha * (interval - smoothed_interval)
        previous_index = index

    forecast = (1.0 - alpha / 2.0) * smoothed_demand / smoothed_interval
    return [float(forecast) for _ in range(horizon)]


def tsb(history, horizon, frequency):
    """Use when demand is intermittent and the history contains both exact zeros and positive observations."""
    if len(history) < 5:
        raise NotApplicable(f"needs at least 5 points, got {len(history)}")
    if any(value < 0 for value in history):
        raise NotApplicable("needs non-negative demand values, got negative values")
    zero_count = sum(value == 0 for value in history)
    positive_count = sum(value > 0 for value in history)
    if zero_count == 0 or positive_count == 0:
        raise NotApplicable(f"needs both zero and positive demands, got {zero_count} zeros and {positive_count} positives")
    alpha_probability = 0.1
    alpha_size = 0.1
    probability = 1.0 if history[0] > 0 else 0.0
    size = float(next(value for value in history if value > 0))
    for value in history[1:]:
        occurrence = 1.0 if value > 0 else 0.0
        probability += alpha_probability * (occurrence - probability)
        if occurrence:
            size += alpha_size * (float(value) - size)
    forecast = float(probability * size)
    return [forecast for _ in range(horizon)]


def adida(history, horizon, frequency):
    """Use when nonnegative demand is intermittent and enough observations exist for temporal aggregation."""
    import math
    import numpy as np
    from scipy.optimize import minimize_scalar

    if horizon < 0:
        raise NotApplicable(f"needs a nonnegative horizon, got {horizon}")
    if len(history) == 0:
        raise NotApplicable(f"needs a nonempty history, got {len(history)} points")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if any(float(value) < 0.0 for value in history):
        raise NotApplicable("needs nonnegative intermittent demand, got negative values")

    nonzero_count = sum(float(value) > 0.0 for value in history)
    aggregation_level = 1 if nonzero_count == 0 else int(math.ceil(len(history) / nonzero_count))
    required_points = 3 * aggregation_level
    if len(history) < required_points:
        raise NotApplicable(f"needs {required_points} points for aggregation level {aggregation_level}, got {len(history)}")

    complete_length = (len(history) // aggregation_level) * aggregation_level
    values = np.asarray(history[-complete_length:], dtype=float)
    scale = float(np.max(values))
    if scale == 0.0:
        return [0.0] * horizon

    scaled = values / scale
    aggregated = scaled.reshape(-1, aggregation_level).sum(axis=1)

    def squared_error(alpha):
        level = float(aggregated[0])
        error = 0.0
        for observation in aggregated[1:]:
            difference = float(observation) - level
            error += difference * difference
            level = alpha * float(observation) + (1.0 - alpha) * level
        return error

    result = minimize_scalar(squared_error, bounds=(0.0, 1.0), method="bounded")
    alpha = float(result.x)
    level = float(aggregated[0])
    for observation in aggregated[1:]:
        level = alpha * float(observation) + (1.0 - alpha) * level

    forecast = (level / aggregation_level) * scale
    return [float(forecast)] * horizon


def imapa(history, horizon, frequency):
    """Use when forecasting nonnegative intermittent demand with enough observations for multiple temporal aggregation levels."""
    import math
    from scipy.optimize import minimize_scalar

    if horizon < 1:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    if len(history) < 4:
        raise NotApplicable(f"needs at least 4 points for two aggregation levels, got {len(history)}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if any(float(value) < 0.0 for value in history):
        raise NotApplicable("needs nonnegative intermittent demand, got negative values")

    values = [float(value) for value in history]
    max_level = len(values) // 2
    disaggregated_forecasts = []

    for level in range(1, max_level + 1):
        remainder = len(values) % level
        trimmed = values[remainder:]
        aggregated = [sum(trimmed[i:i + level]) for i in range(0, len(trimmed), level)]

        def squared_error(alpha):
            state = aggregated[0]
            error = 0.0
            for observation in aggregated[1:]:
                residual = observation - state
                error += residual * residual
                state = alpha * observation + (1.0 - alpha) * state
            return error

        result = minimize_scalar(squared_error, bounds=(0.0, 1.0), method="bounded")
        alpha = float(result.x)
        state = aggregated[0]
        for observation in aggregated[1:]:
            state = alpha * observation + (1.0 - alpha) * state
        disaggregated_forecasts.append(state / level)

    forecast = sum(disaggregated_forecasts) / len(disaggregated_forecasts)
    return [float(forecast) for _ in range(horizon)]


def local_level_kalman(history, horizon, frequency):
    """Use when a nonseasonal series has no trend but its underlying level drifts gradually over time."""
    if len(history) < 3:
        raise NotApplicable(f"needs at least 3 points, got {len(history)}")
    import numpy as np
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    values = np.asarray(history, dtype=float)
    model = UnobservedComponents(values, level="local level")
    result = model.fit(disp=False)
    forecast = np.asarray(result.forecast(steps=horizon), dtype=float)
    if forecast.shape != (horizon,) or not np.all(np.isfinite(forecast)):
        raise ValueError("local level Kalman forecast did not produce the required finite values")
    return [float(value) for value in forecast]


def local_linear_trend_kalman(history, horizon, frequency):
    """Use when a nonseasonal series has a slowly evolving level and trend."""
    import numpy as np
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    if len(history) < 3:
        raise NotApplicable(f"needs at least 3 points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    model = UnobservedComponents(values, level="local linear trend")
    fitted = model.fit(disp=False)
    forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float)
    return [float(value) for value in forecast]


def structural_time_series_bsm(history, horizon, frequency):
    """Use when a series has evolving level, trend, and a recurring seasonal pattern."""
    import numpy as np
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported seasonal frequency, got {frequency}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")

    model = UnobservedComponents(
        values,
        level=True,
        trend=True,
        seasonal=period,
        stochastic_level=True,
        stochastic_trend=True,
        stochastic_seasonal=True,
        irregular=True,
    )
    fitted = model.fit(disp=False)
    forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float)
    if forecast.shape != (horizon,) or not np.all(np.isfinite(forecast)):
        raise ValueError("structural model produced an invalid forecast")
    return forecast.tolist()


def robust_loess_trend(history, horizon, frequency):
    """Use when the series has a smooth nonseasonal trend and only a minority of observations are outliers."""
    import numpy as np

    if len(history) < 5:
        raise NotApplicable(f"needs at least 5 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if not all(np.isfinite(v) for v in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    y = np.asarray(history, dtype=float)
    n = len(y)
    x = np.arange(n, dtype=float)
    span = min(n, max(5, int(np.ceil(0.4 * n))))
    robust_weights = np.ones(n, dtype=float)

    for _ in range(4):
        fitted = np.empty(n, dtype=float)
        for i in range(n):
            distances = np.abs(x - x[i])
            bandwidth = np.partition(distances, span - 1)[span - 1]
            scaled = distances / bandwidth
            local_weights = np.where(scaled < 1.0, (1.0 - scaled ** 3) ** 3, 0.0)
            weights = local_weights * robust_weights
            design = np.column_stack((np.ones(n), x - x[i]))
            root_weights = np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                design * root_weights[:, None], y * root_weights, rcond=None
            )[0]
            fitted[i] = coefficients[0]

        residuals = y - fitted
        scale = np.median(np.abs(residuals - np.median(residuals)))
        if scale == 0.0:
            break
        scaled_residuals = residuals / (6.0 * scale)
        robust_weights = np.where(
            np.abs(scaled_residuals) < 1.0,
            (1.0 - scaled_residuals ** 2) ** 2,
            0.0,
        )

    endpoint = n - 1
    distances = np.abs(x - x[endpoint])
    bandwidth = np.partition(distances, span - 1)[span - 1]
    scaled = distances / bandwidth
    local_weights = np.where(scaled < 1.0, (1.0 - scaled ** 3) ** 3, 0.0)
    weights = local_weights * robust_weights
    centered_x = x - x[endpoint]
    design = np.column_stack((np.ones(n), centered_x))
    root_weights = np.sqrt(weights)
    intercept, slope = np.linalg.lstsq(
        design * root_weights[:, None], y * root_weights, rcond=None
    )[0]

    future_steps = np.arange(1, horizon + 1, dtype=float)
    return [float(value) for value in intercept + slope * future_steps]


def fourier_harmonic_regression(history, horizon, frequency):
    """Use when the series has a linear trend and a few stable sinusoidal seasonal patterns with a period implied by its frequency."""
    import numpy as np

    periods = {
        "1 minute": 1440,
        "5 minutes": 288,
        "10 minutes": 144,
        "15 minutes": 96,
        "30 minutes": 48,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if any(not np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    values = np.asarray(history, dtype=float)
    time = np.arange(len(values), dtype=float)
    harmonics = min(3, (period - 1) // 2)
    columns = [np.ones(len(values)), time]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * np.pi * harmonic * time / period
        columns.extend([np.sin(angle), np.cos(angle)])

    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]

    future_time = np.arange(len(values), len(values) + horizon, dtype=float)
    future_columns = [np.ones(horizon), future_time]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * np.pi * harmonic * future_time / period
        future_columns.extend([np.sin(angle), np.cos(angle)])

    future_design = np.column_stack(future_columns)
    return [float(value) for value in future_design @ coefficients]


def linear_trend_regression(history, horizon, frequency):
    """Use when the series has an approximately linear trend and no seasonality."""
    import numpy as np

    if len(history) < 2:
        raise NotApplicable(f"needs at least 2 points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    time = np.arange(len(values), dtype=float)
    design = np.column_stack((np.ones(len(values)), time))
    intercept, slope = np.linalg.lstsq(design, values, rcond=None)[0]
    future_time = np.arange(len(values), len(values) + horizon, dtype=float)
    return [float(intercept + slope * t) for t in future_time]


def polynomial_trend_regression(history, horizon, frequency):
    """Use when a non-seasonal series has a trend that is well approximated by a quadratic polynomial."""
    import numpy as np

    if len(history) < 3:
        raise NotApplicable(f"needs at least 3 points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    time = np.arange(len(values), dtype=float)
    center = time.mean()
    scale = max(center, 1.0)
    scaled_time = (time - center) / scale
    design = np.column_stack((np.ones(len(values)), scaled_time, scaled_time ** 2))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]

    future_time = np.arange(len(values), len(values) + horizon, dtype=float)
    scaled_future = (future_time - center) / scale
    future_design = np.column_stack((np.ones(horizon), scaled_future, scaled_future ** 2))
    forecast = future_design @ coefficients
    return [float(value) for value in forecast]


def piecewise_linear_trend(history, horizon, frequency):
    """Use when a nonseasonal series has sparse structural changes and its latest linear trend is expected to continue."""
    import numpy as np

    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if len(history) < 12:
        raise NotApplicable(f"needs at least 12 points, got {len(history)}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise NotApplicable("needs finite history values, got non-finite values")

    n = len(y)
    min_segment = 4
    max_segments = min(6, n // min_segment)
    costs = np.full((n + 1, n + 1), np.inf)

    for start in range(n):
        for end in range(start + min_segment, n + 1):
            x = np.arange(start, end, dtype=float)
            values = y[start:end]
            x_centered = x - x.mean()
            denominator = np.dot(x_centered, x_centered)
            slope = np.dot(x_centered, values - values.mean()) / denominator
            fitted = values.mean() + slope * x_centered
            residuals = values - fitted
            costs[start, end] = max(float(np.dot(residuals, residuals)), 0.0)

    dp = np.full((max_segments + 1, n + 1), np.inf)
    previous = np.full((max_segments + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for segments in range(1, max_segments + 1):
        for end in range(segments * min_segment, n + 1):
            first_start = (segments - 1) * min_segment
            last_start = end - min_segment
            candidates = dp[segments - 1, first_start:last_start + 1] + costs[first_start:last_start + 1, end]
            offset = int(np.argmin(candidates))
            dp[segments, end] = float(candidates[offset])
            previous[segments, end] = first_start + offset

    scale = max(float(np.dot(y - y.mean(), y - y.mean())), 1.0)
    floor = np.finfo(float).eps * scale
    candidate_segments = np.arange(2, max_segments + 1)
    bic = np.array([
        n * np.log(max(dp[segments, n] / n, floor)) + (3 * segments - 1) * np.log(n)
        for segments in candidate_segments
    ])
    best_segments = int(candidate_segments[int(np.argmin(bic))])
    final_start = int(previous[best_segments, n])

    final_x = np.arange(final_start, n, dtype=float)
    final_y = y[final_start:n]
    x_centered = final_x - final_x.mean()
    slope = np.dot(x_centered, final_y - final_y.mean()) / np.dot(x_centered, x_centered)
    intercept_at_mean = final_y.mean()
    future_x = np.arange(n, n + horizon, dtype=float)
    forecast = intercept_at_mean + slope * (future_x - final_x.mean())
    return [float(value) for value in forecast]


def seasonal_block_bootstrap(history, horizon, frequency):
    """Use when a seasonal series has at least two cycles and its historical cycles plausibly represent future fluctuations around a linear trend."""
    import numpy as np

    periods = {
        "1 minute": 1440,
        "5 minutes": 288,
        "10 minutes": 144,
        "15 minutes": 96,
        "30 minutes": 48,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = str(frequency).strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency!r}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("history must be a one-dimensional sequence of finite values")

    times = np.arange(values.size, dtype=float)
    centered_times = times - times.mean()
    slope = np.dot(centered_times, values - values.mean()) / np.dot(centered_times, centered_times)
    intercept = values.mean() - slope * times.mean()
    residuals = values - (intercept + slope * times)

    complete_length = (values.size // period) * period
    aligned_residuals = residuals[values.size - complete_length:]
    historical_blocks = aligned_residuals.reshape(-1, period)
    expected_bootstrap_block = historical_blocks.mean(axis=0)

    future_times = np.arange(values.size, values.size + horizon, dtype=float)
    future_residuals = np.resize(expected_bootstrap_block, horizon)
    forecast = intercept + slope * future_times + future_residuals
    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("forecast contains non-finite values")
    return [float(value) for value in forecast]


def nearest_neighbor_lag_analogue(history, horizon, frequency):
    """Use when recurring lag patterns are expected to have similar future continuations."""
    import math
    import numpy as np

    if not isinstance(horizon, int) or horizon < 1:
        raise NotApplicable(f"needs a positive integer horizon, got {horizon}")

    n = len(history)
    lag = max(2, min(24, int(math.sqrt(n))))
    needed = lag + horizon + 2
    if n < needed:
        raise NotApplicable(f"needs {needed} points, got {n}")
    if not all(math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    values = np.asarray(history, dtype=float)
    query = values[-lag:]
    endpoints = np.arange(lag, n - horizon + 1)
    lag_vectors = np.stack([values[t - lag:t] for t in endpoints])
    continuations = np.stack([values[t:t + horizon] for t in endpoints])
    distances = np.linalg.norm(lag_vectors - query, axis=1)
    neighbor_count = min(5, len(endpoints))
    nearest = np.argsort(distances, kind="mergesort")[:neighbor_count]
    forecast = np.mean(continuations[nearest], axis=0)
    return [float(value) for value in forecast]


def dtw_analogue_forecast(history, horizon, frequency):
    """Use when recurring episode shapes may predict their subsequent paths despite modest local timing distortions."""
    import math
    import numpy as np

    if horizon <= 0:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    pattern_length = max(8, min(24, 2 * horizon))
    needed = 2 * pattern_length + horizon
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    query = values[-pattern_length:]
    epsilon = np.finfo(float).eps

    def standardize(segment):
        center = float(np.mean(segment))
        scale = float(np.std(segment))
        return (segment - center) / max(scale, epsilon), max(scale, epsilon)

    def dtw_distance(left, right):
        length = len(left)
        window = max(1, int(math.ceil(0.2 * length)))
        costs = np.full((length + 1, length + 1), np.inf)
        costs[0, 0] = 0.0
        for i in range(1, length + 1):
            lower = max(1, i - window)
            upper = min(length, i + window)
            for j in range(lower, upper + 1):
                difference = left[i - 1] - right[j - 1]
                costs[i, j] = difference * difference + min(
                    costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1]
                )
        return math.sqrt(float(costs[length, length]) / length)

    query_shape, query_scale = standardize(query)
    candidate_count = len(values) - 2 * pattern_length - horizon + 1
    distances = []
    continuations = []

    for start in range(candidate_count):
        candidate = values[start:start + pattern_length]
        candidate_shape, candidate_scale = standardize(candidate)
        distance = dtw_distance(query_shape, candidate_shape)
        following = values[start + pattern_length:start + pattern_length + horizon]
        continuation = query[-1] + query_scale * (following - candidate[-1]) / candidate_scale
        distances.append(distance)
        continuations.append(continuation)

    neighbour_count = min(7, max(1, int(math.sqrt(candidate_count))))
    selected = np.argsort(np.asarray(distances))[:neighbour_count]
    selected_distances = np.asarray(distances)[selected]
    selected_paths = np.asarray(continuations)[selected]

    exact = selected_distances <= epsilon
    if np.any(exact):
        forecast = np.mean(selected_paths[exact], axis=0)
    else:
        weights = 1.0 / selected_distances
        weights = weights / np.sum(weights)
        forecast = np.sum(selected_paths * weights[:, None], axis=0)

    return [float(value) for value in forecast]


def poisson_dynamic_regression(history, horizon, frequency):
    """Use when nonnegative count observations have Poisson-like conditional variance and their log mean evolves with time and the recent count level."""
    import numpy as np
    import statsmodels.api as sm

    if horizon < 0:
        raise ValueError(f"horizon must be nonnegative, got {horizon}")
    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 count observations, got {len(history)}")
    if any(not np.isfinite(value) for value in history):
        raise NotApplicable("needs finite count observations, got non-finite values")
    if any(value < 0 or float(value).is_integer() is False for value in history):
        raise NotApplicable("needs nonnegative integer counts, got values outside the count domain")
    if not any(value > 0 for value in history):
        raise NotApplicable("needs at least one positive count, got all zeros")

    counts = np.asarray(history, dtype=float)
    times = np.arange(1, len(counts), dtype=float)
    time_center = times.mean()
    time_scale = times.std()
    design = np.column_stack((
        np.ones(len(times), dtype=float),
        (times - time_center) / time_scale,
        np.log1p(counts[:-1]),
    ))
    model = sm.GLM(counts[1:], design, family=sm.families.Poisson())
    fitted = model.fit()

    forecasts = []
    previous = counts[-1]
    for step in range(horizon):
        future_time = float(len(counts) + step)
        predictors = np.asarray([
            1.0,
            (future_time - time_center) / time_scale,
            np.log1p(previous),
        ])
        linear_mean = float(np.dot(predictors, fitted.params))
        prediction = float(np.exp(np.clip(linear_mean, -700.0, 700.0)))
        if not np.isfinite(prediction):
            raise FloatingPointError("Poisson dynamic regression produced a non-finite forecast")
        forecasts.append(prediction)
        previous = prediction

    return forecasts


def negative_binomial_dynamic_regression(history, horizon, frequency):
    """Use when nonnegative integer counts are overdispersed and their mean depends on recent counts and a time trend."""
    import math
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import gammaln

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points, got {len(history)}")
    if not isinstance(horizon, int) or horizon < 0:
        raise NotApplicable(f"needs a nonnegative integer horizon, got {horizon}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a nonempty frequency string, got {frequency!r}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite count observations, got non-finite values")
    if any(float(value) < 0.0 for value in history):
        raise NotApplicable(f"needs nonnegative counts, got minimum {min(history)}")
    if any(not float(value).is_integer() for value in history):
        raise NotApplicable("needs integer-valued counts, got non-integer observations")
    if not any(float(value) > 0.0 for value in history):
        raise NotApplicable(f"needs at least one positive count, got {len(history)} zeros")

    y_all = np.asarray(history, dtype=float)
    sample_mean = float(np.mean(y_all))
    sample_variance = float(np.var(y_all, ddof=1))
    if sample_variance <= sample_mean:
        raise NotApplicable(
            f"needs sample variance greater than sample mean, got variance {sample_variance} and mean {sample_mean}"
        )

    n = len(y_all)
    target = y_all[1:]
    time_scale = float(n - 1)
    times = (np.arange(1, n, dtype=float) - (n - 1) / 2.0) / time_scale
    lagged = np.log1p(y_all[:-1])
    design = np.column_stack((np.ones(n - 1), times, lagged))

    initial_beta = np.linalg.lstsq(design, np.log1p(target), rcond=None)[0]
    initial_alpha = max((sample_variance - sample_mean) / (sample_mean * sample_mean), 1e-6)
    initial = np.concatenate((initial_beta, [math.log(initial_alpha)]))

    def negative_log_likelihood(parameters):
        beta = parameters[:-1]
        alpha = np.exp(parameters[-1])
        size = 1.0 / alpha
        log_mean = np.clip(design @ beta, -30.0, 30.0)
        mean = np.exp(log_mean)
        log_total = np.log(size + mean)
        log_likelihood = (
            gammaln(target + size)
            - gammaln(size)
            - gammaln(target + 1.0)
            + size * (math.log(size) - log_total)
            + target * (log_mean - log_total)
        )
        return float(-np.sum(log_likelihood))

    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0)] * 3 + [(-18.0, 10.0)],
    )
    if not result.success:
        raise RuntimeError(f"negative-binomial optimization failed: {result.message}")
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("negative-binomial optimization produced non-finite parameters")

    beta = result.x[:-1]
    forecasts = []
    previous = float(y_all[-1])
    for step in range(horizon):
        future_index = n + step
        future_time = (future_index - (n - 1) / 2.0) / time_scale
        predictors = np.array([1.0, future_time, math.log1p(previous)])
        log_mean = float(np.clip(predictors @ beta, -30.0, 30.0))
        previous = float(math.exp(log_mean))
        forecasts.append(previous)

    return forecasts


def integer_autoregressive_inar(history, horizon, frequency):
    """Use when forecasting stable, nonnegative integer counts whose dependence is plausibly driven by binomial thinning and integer-valued innovations."""
    import math

    if len(history) < 3:
        raise NotApplicable(f"needs at least 3 points, got {len(history)}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite counts, got non-finite values")
    if any(float(value) < 0 or not float(value).is_integer() for value in history):
        raise NotApplicable("needs nonnegative integer counts, got negative or fractional values")
    if len(set(float(value) for value in history[:-1])) < 2:
        raise NotApplicable("needs varying lagged counts, got constant lagged counts")

    counts = [float(value) for value in history]
    scale = max(counts)
    scaled = [value / scale for value in counts]
    lagged = scaled[:-1]
    current = scaled[1:]
    lagged_mean = math.fsum(lagged) / len(lagged)
    current_mean = math.fsum(current) / len(current)
    denominator = math.fsum((value - lagged_mean) ** 2 for value in lagged)
    numerator = math.fsum(
        (lagged[i] - lagged_mean) * (current[i] - current_mean)
        for i in range(len(lagged))
    )
    thinning_probability = min(max(numerator / denominator, 0.0), 1.0 - 1e-12)
    stationary_mean = math.fsum(scaled) / len(scaled)
    innovation_mean = (1.0 - thinning_probability) * stationary_mean

    forecasts = []
    level = scaled[-1]
    for _ in range(horizon):
        level = thinning_probability * level + innovation_mean
        forecasts.append(float(level * scale))
    return forecasts


def gaussian_process_autoregression(history, horizon, frequency):
    """Use when a nonlinear autoregressive relationship is expected to vary smoothly across similar lag patterns."""
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")

    lag = min(12, max(2, len(values) // 4))
    if len(values) - lag < 3:
        raise NotApplicable(f"needs at least {lag + 3} points for lag {lag}, got {len(history)}")

    X = np.asarray([values[i - lag:i] for i in range(lag, len(values))])
    y = values[lag:]
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0.0] = 1.0
    X_scaled = (X - center) / scale

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(lag),
        length_scale_bounds=(1e-2, 1e2),
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        n_restarts_optimizer=0,
    )
    model.fit(X_scaled, y)

    extended = values.tolist()
    forecast = []
    for _ in range(horizon):
        lag_vector = np.asarray(extended[-lag:], dtype=float)
        prediction = float(model.predict(((lag_vector - center) / scale).reshape(1, -1))[0])
        if not np.isfinite(prediction):
            raise ValueError("Gaussian-process prediction was not finite")
        forecast.append(prediction)
        extended.append(prediction)

    return forecast


def support_vector_lag_regression(history, horizon, frequency):
    """Use when lagged values and calendar position have a stable nonlinear relationship with the target."""
    import math
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
    }
    if frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    period = periods[frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    lag_count = period
    features = []
    targets = []
    for t in range(lag_count, len(values)):
        angle = 2.0 * math.pi * (t % period) / period
        features.append(np.concatenate((values[t - lag_count:t], [math.sin(angle), math.cos(angle)])))
        targets.append(values[t])

    target_scaler = StandardScaler()
    scaled_targets = target_scaler.fit_transform(np.asarray(targets, dtype=float).reshape(-1, 1)).ravel()
    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))
    model.fit(np.asarray(features, dtype=float), scaled_targets)

    extended = values.tolist()
    forecast = []
    for t in range(len(values), len(values) + horizon):
        angle = 2.0 * math.pi * (t % period) / period
        feature = np.asarray(extended[-lag_count:] + [math.sin(angle), math.cos(angle)], dtype=float).reshape(1, -1)
        scaled_prediction = model.predict(feature).reshape(-1, 1)
        prediction = float(target_scaler.inverse_transform(scaled_prediction)[0, 0])
        if not math.isfinite(prediction):
            raise RuntimeError("support-vector regression produced a non-finite forecast")
        forecast.append(prediction)
        extended.append(prediction)
    return forecast


def kernel_ridge_lag_regression(history, horizon, frequency):
    """Use when a sufficiently long univariate series has smooth nonlinear relationships between recent lagged values and its next value."""
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler

    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points, got {len(history)}")
    if not all(np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    values = np.asarray(history, dtype=float)
    lag = min(24, max(3, int(np.sqrt(len(values)))))
    features = np.asarray([values[index - lag:index] for index in range(lag, len(values))])
    targets = values[lag:]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    target_mean = float(np.mean(targets))
    target_scale = float(np.std(targets))
    if target_scale == 0.0:
        target_scale = 1.0
    scaled_targets = (targets - target_mean) / target_scale

    model = KernelRidge(alpha=0.1, kernel="rbf", gamma=1.0 / lag)
    model.fit(scaled_features, scaled_targets)

    extended = values.tolist()
    forecast = []
    for _ in range(horizon):
        lagged = np.asarray(extended[-lag:], dtype=float).reshape(1, -1)
        prediction = float(model.predict(scaler.transform(lagged))[0] * target_scale + target_mean)
        if not np.isfinite(prediction):
            raise FloatingPointError("kernel ridge regression produced a non-finite forecast")
        forecast.append(prediction)
        extended.append(prediction)
    return forecast


def neural_network_autoregression(history, horizon, frequency):
    """Use when a sufficiently long series has stable nonlinear relationships among recent and seasonal observations."""
    import numpy as np
    from scipy.optimize import least_squares

    parts = frequency.lower().strip().split()
    if len(parts) != 2 or not parts[0].replace('.', '', 1).isdigit():
        raise NotApplicable(f"needs a supported regular frequency, got {frequency}")
    step = float(parts[0])
    unit = parts[1].rstrip('s')
    base_steps = {
        'second': 86400.0,
        'minute': 1440.0,
        'hour': 24.0,
        'day': 7.0,
        'week': 52.0,
        'month': 12.0,
        'quarter': 4.0,
    }
    if step <= 0 or unit not in base_steps:
        raise NotApplicable(f"needs a supported positive regular frequency, got {frequency}")
    cycles = base_steps[unit] / step
    period = int(round(cycles))
    if period < 2 or abs(cycles - period) > 1e-9:
        raise NotApplicable(f"needs a frequency that evenly divides a seasonal cycle, got {frequency}")

    short_lags = list(range(1, min(6, period) + 1))
    lags = sorted(set(short_lags + [period, 2 * period]))
    needed = max(48, 8 * period)
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise NotApplicable(f"needs finite history values, got {len(history)} points containing non-finite values")

    center = float(np.mean(values))
    scale = float(np.std(values))
    if scale == 0.0:
        scale = 1.0
    standardized = (values - center) / scale
    max_lag = max(lags)
    X = np.asarray([[standardized[t - lag] for lag in lags] for t in range(max_lag, len(values))], dtype=float)
    y = standardized[max_lag:]

    feature_count = X.shape[1]
    hidden_count = 6
    input_weight_count = feature_count * hidden_count
    total_parameters = input_weight_count + hidden_count + hidden_count + 1
    initial = 0.03 * np.sin(np.arange(total_parameters, dtype=float) + 1.0)
    ridge = 1e-3

    def unpack(parameters):
        offset = input_weight_count
        input_weights = parameters[:offset].reshape(feature_count, hidden_count)
        hidden_bias = parameters[offset:offset + hidden_count]
        offset += hidden_count
        output_weights = parameters[offset:offset + hidden_count]
        output_bias = parameters[-1]
        return input_weights, hidden_bias, output_weights, output_bias

    def residuals(parameters):
        input_weights, hidden_bias, output_weights, output_bias = unpack(parameters)
        hidden = np.tanh(X @ input_weights + hidden_bias)
        fitted = hidden @ output_weights + output_bias
        return np.concatenate((fitted - y, np.sqrt(ridge) * parameters))

    fitted_parameters = least_squares(residuals, initial, method='trf', max_nfev=2000).x
    input_weights, hidden_bias, output_weights, output_bias = unpack(fitted_parameters)
    extended = standardized.tolist()
    forecasts = []
    for _ in range(horizon):
        features = np.asarray([extended[-lag] for lag in lags], dtype=float)
        hidden = np.tanh(features @ input_weights + hidden_bias)
        prediction = float(hidden @ output_weights + output_bias)
        if not np.isfinite(prediction):
            raise FloatingPointError("neural network produced a non-finite forecast")
        prediction = float(np.clip(prediction, -8.0, 8.0))
        extended.append(prediction)
        forecasts.append(float(prediction * scale + center))
    return forecasts


def deepar(history, horizon, frequency):
    """Use when related series share autoregressive dynamics that can be learned with a Gaussian predictive likelihood."""
    import math
    import torch

    periods = {
        "1 minute": 60,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
        "1 year": 1,
    }
    if frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    period = periods[frequency]
    context = max(4, 2 * period)
    needed = context + 2
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    values = [float(value) for value in history]
    if not all(math.isfinite(value) for value in values):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon == 0:
        return []

    dtype = torch.float64
    magnitude = max(max(abs(value) for value in values), 1.0)
    scaled = [value / magnitude for value in values]
    center = sum(scaled) / len(scaled)
    variance = sum((value - center) ** 2 for value in scaled) / len(scaled)
    spread = max(math.sqrt(variance), 1e-6)
    normalized = torch.tensor([(value - center) / spread for value in scaled], dtype=dtype)

    first = max(0, len(values) - context - 512)
    starts = list(range(first, len(values) - context))
    inputs = torch.stack([normalized[start:start + context] for start in starts])
    targets = torch.stack([normalized[start + 1:start + context + 1] for start in starts])

    hidden_size = 12
    wx = torch.linspace(-0.25, 0.25, hidden_size, dtype=dtype).requires_grad_()
    wh_base = 0.35 * torch.eye(hidden_size, dtype=dtype)
    wh_base += 0.04 * torch.roll(torch.eye(hidden_size, dtype=dtype), 1, 1)
    wh = wh_base.requires_grad_()
    bh = torch.zeros(hidden_size, dtype=dtype, requires_grad=True)
    wmu = torch.linspace(0.12, -0.12, hidden_size, dtype=dtype).requires_grad_()
    bmu = torch.zeros(1, dtype=dtype, requires_grad=True)
    wsigma = torch.linspace(-0.05, 0.05, hidden_size, dtype=dtype).requires_grad_()
    bsigma = torch.zeros(1, dtype=dtype, requires_grad=True)

    parameters = [wx, wh, bh, wmu, bmu, wsigma, bsigma]
    optimizer = torch.optim.Adam(parameters, lr=0.025)

    for _ in range(100):
        optimizer.zero_grad()
        state = torch.zeros((inputs.shape[0], hidden_size), dtype=dtype)
        means = []
        log_scales = []
        for step in range(context):
            state = torch.tanh(inputs[:, step:step + 1] * wx + state @ wh + bh)
            means.append(state @ wmu + bmu)
            log_scales.append(torch.clamp(state @ wsigma + bsigma, -6.0, 3.0))
        mean_tensor = torch.stack(means, dim=1)
        log_scale_tensor = torch.stack(log_scales, dim=1)
        residual = targets - mean_tensor
        loss = (0.5 * residual.square() * torch.exp(-2.0 * log_scale_tensor) + log_scale_tensor).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()

    with torch.no_grad():
        state = torch.zeros((1, hidden_size), dtype=dtype)
        for value in normalized[-context:]:
            state = torch.tanh(value.reshape(1, 1) * wx + state @ wh + bh)

        forecasts = []
        finite_limit = 1.7976931348623157e308 / magnitude
        for _ in range(horizon):
            prediction = torch.clamp(state @ wmu + bmu, -20.0, 20.0)
            scaled_prediction = float(prediction.item()) * spread + center
            scaled_prediction = max(-finite_limit, min(finite_limit, scaled_prediction))
            forecast = scaled_prediction * magnitude
            if not math.isfinite(forecast):
                raise RuntimeError("DeepAR produced a non-finite forecast")
            forecasts.append(float(forecast))
            state = torch.tanh(prediction.reshape(1, 1) * wx + state @ wh + bh)

    return forecasts


def nbeats(history, horizon, frequency):
    """Use when many historical windows are available to learn shared nonlinear trend and seasonal basis expansions."""
    import math
    import torch

    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if horizon == 0:
        return []

    backcast_length = max(12, 2 * horizon)
    required = backcast_length + horizon + 31
    if len(history) < required:
        raise NotApplicable(f"needs {required} points for at least 32 training windows, got {len(history)}")

    values = [float(value) for value in history]
    if any(not math.isfinite(value) for value in values):
        raise NotApplicable("needs finite history values, got non-finite values")

    dtype = torch.float32
    series = torch.tensor(values, dtype=dtype)
    windows = series.unfold(0, backcast_length + horizon, 1)
    if windows.shape[0] > 512:
        indices = torch.linspace(0, windows.shape[0] - 1, 512).round().long()
        windows = windows[indices]

    inputs = windows[:, :backcast_length]
    targets = windows[:, backcast_length:]
    centers = inputs.mean(dim=1, keepdim=True)
    scales = inputs.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    inputs = (inputs - centers) / scales
    targets = (targets - centers) / scales

    trend_backcast_time = torch.linspace(-1.0, 0.0, backcast_length, dtype=dtype)
    trend_forecast_time = torch.linspace(0.0, 1.0, horizon, dtype=dtype)
    trend_backcast_basis = torch.stack([trend_backcast_time ** degree for degree in range(3)])
    trend_forecast_basis = torch.stack([trend_forecast_time ** degree for degree in range(3)])

    harmonics = min(6, max(1, horizon))
    seasonal_backcast_time = torch.arange(backcast_length, dtype=dtype) / backcast_length
    seasonal_forecast_time = torch.arange(horizon, dtype=dtype) / max(1, horizon)

    def seasonal_basis(time):
        rows = [torch.ones_like(time)]
        for harmonic in range(1, harmonics + 1):
            angle = 2.0 * math.pi * harmonic * time
            rows.extend([torch.cos(angle), torch.sin(angle)])
        return torch.stack(rows)

    seasonal_backcast_basis = seasonal_basis(seasonal_backcast_time)
    seasonal_forecast_basis = seasonal_basis(seasonal_forecast_time)
    width = min(128, max(32, 2 * backcast_length))

    def deterministic_parameter(rows, columns, offset):
        positions = torch.arange(rows * columns, dtype=dtype).reshape(rows, columns)
        values = 0.08 * torch.sin(0.173 * positions + offset) / math.sqrt(max(1, rows))
        return torch.nn.Parameter(values)

    class Block(torch.nn.Module):
        def __init__(self, kind, theta_size, offset):
            super().__init__()
            self.kind = kind
            dimensions = [backcast_length, width, width, width, width]
            self.weights = torch.nn.ParameterList([
                deterministic_parameter(dimensions[index], dimensions[index + 1], offset + index)
                for index in range(4)
            ])
            self.biases = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(width, dtype=dtype)) for _ in range(4)
            ])
            self.backcast_head = deterministic_parameter(width, theta_size, offset + 5.0)
            self.forecast_head = deterministic_parameter(width, theta_size, offset + 6.0)
            self.backcast_bias = torch.nn.Parameter(torch.zeros(theta_size, dtype=dtype))
            self.forecast_bias = torch.nn.Parameter(torch.zeros(theta_size, dtype=dtype))

        def forward(self, residual, backcast_basis, forecast_basis):
            hidden = residual
            for weight, bias in zip(self.weights, self.biases):
                hidden = torch.relu(hidden @ weight + bias)
            backcast_theta = hidden @ self.backcast_head + self.backcast_bias
            forecast_theta = hidden @ self.forecast_head + self.forecast_bias
            return backcast_theta @ backcast_basis, forecast_theta @ forecast_basis

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            seasonal_size = 1 + 2 * harmonics
            self.blocks = torch.nn.ModuleList([
                Block("trend", 3, 1.0),
                Block("seasonal", seasonal_size, 11.0),
                Block("trend", 3, 21.0),
                Block("seasonal", seasonal_size, 31.0),
            ])

        def forward(self, values):
            residual = values
            forecast = torch.zeros((values.shape[0], horizon), dtype=values.dtype, device=values.device)
            for block in self.blocks:
                if block.kind == "trend":
                    backcast_basis = trend_backcast_basis
                    forecast_basis = trend_forecast_basis
                else:
                    backcast_basis = seasonal_backcast_basis
                    forecast_basis = seasonal_forecast_basis
                backcast, block_forecast = block(residual, backcast_basis, forecast_basis)
                residual = residual - backcast
                forecast = forecast + block_forecast
            return forecast

    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(220):
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = torch.mean((prediction - targets) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    final_input = series[-backcast_length:]
    final_center = final_input.mean()
    final_scale = final_input.std(unbiased=False).clamp_min(1e-6)
    final_input = ((final_input - final_center) / final_scale).unsqueeze(0)
    with torch.no_grad():
        result = model(final_input).squeeze(0) * final_scale + final_center

    output = [float(value) for value in result.tolist()]
    if any(not math.isfinite(value) for value in output):
        raise FloatingPointError("N-BEATS produced a non-finite forecast")
    return output


def nhits(history, horizon, frequency):
    """Use when a sufficiently long series contains forecast structure operating at several temporal resolutions."""
    import numpy as np

    if horizon <= 0:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty frequency, got {frequency!r}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise NotApplicable(f"needs finite history values, got {len(history)} values with non-finite entries")

    input_size = max(16, 2 * horizon)
    input_size = 4 * ((input_size + 3) // 4)
    needed = input_size + horizon + 8
    if len(values) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    center = float(np.mean(values))
    scale = float(np.sqrt(np.mean((values - center) ** 2) + 1e-8))
    series = (values - center) / scale
    sample_count = len(series) - input_size - horizon + 1
    inputs = np.stack([series[i:i + input_size] for i in range(sample_count)])
    targets = np.stack([series[i + input_size:i + input_size + horizon] for i in range(sample_count)])

    def interpolate(coefficients, length):
        knot_count = coefficients.shape[1]
        if knot_count == 1:
            return np.repeat(coefficients, length, axis=1)
        knot_positions = np.linspace(0.0, length - 1.0, knot_count)
        output_positions = np.arange(length, dtype=float)
        return np.stack([np.interp(output_positions, knot_positions, row) for row in coefficients])

    residual_inputs = inputs.copy()
    residual_targets = targets.copy()
    current_residual = series[-input_size:].reshape(1, -1)
    current_forecast = np.zeros((1, horizon), dtype=float)

    for pooling_size in (4, 2, 1):
        pooled_inputs = residual_inputs.reshape(sample_count, input_size // pooling_size, pooling_size).mean(axis=2)
        current_pooled = current_residual.reshape(1, input_size // pooling_size, pooling_size).mean(axis=2)
        design = np.concatenate([np.ones((sample_count, 1)), pooled_inputs], axis=1)
        current_design = np.concatenate([np.ones((1, 1)), current_pooled], axis=1)

        backcast_knots = max(2, input_size // pooling_size)
        forecast_knots = max(1, (horizon + pooling_size - 1) // pooling_size)
        backcast_indices = np.linspace(0, input_size - 1, backcast_knots).round().astype(int)
        forecast_indices = np.linspace(0, horizon - 1, forecast_knots).round().astype(int)
        coefficients_target = np.concatenate([
            residual_inputs[:, backcast_indices],
            residual_targets[:, forecast_indices]
        ], axis=1)

        penalty = np.eye(design.shape[1]) * 1e-3
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ coefficients_target)
        fitted_coefficients = design @ weights
        current_coefficients = current_design @ weights

        fitted_backcast = interpolate(fitted_coefficients[:, :backcast_knots], input_size)
        fitted_forecast = interpolate(fitted_coefficients[:, backcast_knots:], horizon)
        predicted_backcast = interpolate(current_coefficients[:, :backcast_knots], input_size)
        predicted_forecast = interpolate(current_coefficients[:, backcast_knots:], horizon)

        residual_inputs = residual_inputs - fitted_backcast
        residual_targets = residual_targets - fitted_forecast
        current_residual = current_residual - predicted_backcast
        current_forecast = current_forecast + predicted_forecast

    result = current_forecast[0] * scale + center
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nhits produced non-finite forecasts")
    return [float(value) for value in result]


def temporal_fusion_transformer(history, horizon, frequency):
    """Use when stable observed and known-future calendar covariates can support multi-horizon forecasting."""
    import math
    import numpy as np
    import torch

    if horizon <= 0:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")

    parts = str(frequency).lower().strip().split()
    units = {"minute": 1440.0, "hour": 24.0, "day": 7.0, "week": 52.0, "month": 12.0, "quarter": 4.0, "year": 1.0}
    if len(parts) != 2:
        raise NotApplicable(f"needs a numeric supported frequency, got {frequency}")
    unit = parts[1].rstrip("s")
    numeric = parts[0].replace(".", "", 1)
    if unit not in units or not numeric.isdigit():
        raise NotApplicable(f"needs a supported minute-to-year frequency, got {frequency}")
    amount = float(parts[0])
    if amount <= 0:
        raise NotApplicable(f"needs a positive frequency interval, got {frequency}")

    needed = horizon + 12
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")
    if not all(math.isfinite(float(v)) for v in history):
        raise NotApplicable(f"needs finite history values, got {len(history)} points containing non-finite values")

    values = np.asarray(history, dtype=np.float64)
    mean = float(values.mean())
    scale = float(values.std())
    if scale == 0.0:
        scale = 1.0
    normalized = (values - mean) / scale
    n = len(values)
    period = max(1, int(round(units[unit] / amount)))
    context = min(24, n - horizon - 4)
    starts = np.arange(max(0, n - context - horizon - 255), n - context - horizon + 1)

    input_rows = []
    future_rows = []
    targets = []
    for start in starts:
        positions = np.arange(start, start + context)
        previous = np.maximum(positions - 1, 0)
        inputs = np.stack([
            normalized[positions],
            normalized[positions] - normalized[previous],
            np.sin(2.0 * np.pi * positions / period),
            np.cos(2.0 * np.pi * positions / period),
            positions / max(1.0, float(n - 1))
        ], axis=1)
        future_positions = np.arange(start + context, start + context + horizon)
        known_future = np.stack([
            np.sin(2.0 * np.pi * future_positions / period),
            np.cos(2.0 * np.pi * future_positions / period),
            future_positions / max(1.0, float(n + horizon - 1))
        ], axis=1)
        input_rows.append(inputs)
        future_rows.append(known_future)
        targets.append(normalized[future_positions])

    x = torch.tensor(np.stack(input_rows), dtype=torch.float64)
    future = torch.tensor(np.stack(future_rows), dtype=torch.float64)
    target = torch.tensor(np.stack(targets), dtype=torch.float64)

    class TFT(torch.nn.Module):
        def __init__(self, features, hidden):
            super().__init__()
            self.hidden = hidden
            self.offset = 0
            self.variable_embeddings = self.parameter((features, hidden))
            self.variable_weights = self.parameter((features, features))
            self.variable_bias = self.parameter((features,))
            self.wz = self.parameter((hidden, hidden))
            self.uz = self.parameter((hidden, hidden))
            self.bz = self.parameter((hidden,))
            self.wr = self.parameter((hidden, hidden))
            self.ur = self.parameter((hidden, hidden))
            self.br = self.parameter((hidden,))
            self.wn = self.parameter((hidden, hidden))
            self.un = self.parameter((hidden, hidden))
            self.bn = self.parameter((hidden,))
            self.query = self.parameter((3, hidden))
            self.key = self.parameter((hidden, hidden))
            self.value = self.parameter((hidden, hidden))
            self.gate = self.parameter((hidden * 3, hidden))
            self.gate_bias = self.parameter((hidden,))
            self.output = self.parameter((hidden, 1))
            self.output_bias = self.parameter((1,))

        def parameter(self, shape):
            count = int(np.prod(shape))
            index = torch.arange(self.offset, self.offset + count, dtype=torch.float64)
            self.offset += count
            data = (0.04 * torch.sin(index * 0.173 + 0.31)).reshape(shape)
            return torch.nn.Parameter(data)

        def forward(self, inputs, known_future):
            selection_logits = torch.matmul(inputs, self.variable_weights) + self.variable_bias
            selection = torch.softmax(selection_logits, dim=-1)
            embedded = inputs.unsqueeze(-1) * self.variable_embeddings
            selected = torch.sum(selection.unsqueeze(-1) * embedded, dim=2)
            state = torch.zeros(inputs.shape[0], self.hidden, dtype=inputs.dtype, device=inputs.device)
            states = []
            for step in range(inputs.shape[1]):
                current = selected[:, step, :]
                update = torch.sigmoid(current @ self.wz + state @ self.uz + self.bz)
                reset = torch.sigmoid(current @ self.wr + state @ self.ur + self.br)
                candidate = torch.tanh(current @ self.wn + (reset * state) @ self.un + self.bn)
                state = update * state + (1.0 - update) * candidate
                states.append(state)
            encoded = torch.stack(states, dim=1)
            keys = encoded @ self.key
            attended_values = encoded @ self.value
            outputs = []
            for step in range(known_future.shape[1]):
                query = known_future[:, step, :] @ self.query
                scores = torch.sum(keys * query.unsqueeze(1), dim=-1) / math.sqrt(self.hidden)
                attention = torch.softmax(scores, dim=1)
                context_vector = torch.sum(attention.unsqueeze(-1) * attended_values, dim=1)
                gate_input = torch.cat([context_vector, state, query], dim=1)
                gate_value = torch.sigmoid(gate_input @ self.gate + self.gate_bias)
                fused = gate_value * context_vector + (1.0 - gate_value) * state
                outputs.append((fused @ self.output + self.output_bias).squeeze(-1))
            return torch.stack(outputs, dim=1)

    model = TFT(5, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(160):
        optimizer.zero_grad()
        prediction = model(x, future)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    final_positions = np.arange(n - context, n)
    final_previous = np.maximum(final_positions - 1, 0)
    final_input = np.stack([
        normalized[final_positions],
        normalized[final_positions] - normalized[final_previous],
        np.sin(2.0 * np.pi * final_positions / period),
        np.cos(2.0 * np.pi * final_positions / period),
        final_positions / max(1.0, float(n - 1))
    ], axis=1)
    forecast_positions = np.arange(n, n + horizon)
    final_future = np.stack([
        np.sin(2.0 * np.pi * forecast_positions / period),
        np.cos(2.0 * np.pi * forecast_positions / period),
        forecast_positions / max(1.0, float(n + horizon - 1))
    ], axis=1)
    with torch.no_grad():
        forecast = model(
            torch.tensor(final_input[None, :, :], dtype=torch.float64),
            torch.tensor(final_future[None, :, :], dtype=torch.float64)
        )[0].cpu().numpy()
    result = forecast * scale + mean
    if not np.all(np.isfinite(result)):
        raise ValueError("temporal fusion transformer produced non-finite forecasts")
    return [float(value) for value in result]


def intervention_arima(history, horizon, frequency):
    """Use when a series has an identifiable isolated pulse, persistent level shift, or gradually decaying intervention alongside autocorrelated dynamics."""
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if len(history) < 16:
        raise NotApplicable(f"needs at least 16 points, got {len(history)}")
    if not isinstance(horizon, int) or horizon < 0:
        raise NotApplicable(f"needs a non-negative integer horizon, got {horizon}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty frequency string, got {frequency!r}")

    y = np.asarray(history, dtype=float)
    bad = int(np.count_nonzero(~np.isfinite(y)))
    if bad:
        raise NotApplicable(f"needs finite history values, got {bad} non-finite values")
    if horizon == 0:
        return []

    baseline = SARIMAX(
        y,
        order=(1, 1, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=500)
    innovations = np.asarray(baseline.resid, dtype=float)
    center = float(np.median(innovations[2:]))
    intervention_time = int(np.argmax(np.abs(innovations[2:] - center)) + 2)

    n = len(y)
    observed_time = np.arange(n)
    future_time = np.arange(n, n + horizon)
    candidates = []

    pulse_observed = (observed_time == intervention_time).astype(float)[:, None]
    pulse_future = np.zeros((horizon, 1), dtype=float)
    candidates.append((pulse_observed, pulse_future))

    shift_observed = (observed_time >= intervention_time).astype(float)[:, None]
    shift_future = np.ones((horizon, 1), dtype=float)
    candidates.append((shift_observed, shift_future))

    decay = 0.7
    temporary_observed = np.where(
        observed_time >= intervention_time,
        decay ** (observed_time - intervention_time),
        0.0,
    )[:, None]
    temporary_future = (decay ** (future_time - intervention_time))[:, None]
    candidates.append((temporary_observed, temporary_future))

    fitted = []
    for observed_x, future_x in candidates:
        result = SARIMAX(
            y,
            exog=observed_x,
            order=(1, 1, 1),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=500)
        fitted.append((float(result.aic), result, future_x))

    _, selected, selected_future = min(fitted, key=lambda item: item[0])
    forecast = np.asarray(
        selected.get_forecast(steps=horizon, exog=selected_future).predicted_mean,
        dtype=float,
    )
    if forecast.shape != (horizon,) or not np.all(np.isfinite(forecast)):
        raise ValueError("intervention ARIMA produced a non-finite or incorrectly shaped forecast")
    return [float(value) for value in forecast]


def outlier_adjusted_arima(history, horizon, frequency):
    """Use when isolated additive or innovation shocks contaminate an otherwise stable ARIMA series."""
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    if len(history) < 30:
        raise NotApplicable(f"needs 30 points, got {len(history)}")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    candidate_orders = [(p, 0, q) for p in range(3) for q in range(3)]
    fitted_candidates = [
        ARIMA(
            values,
            order=order,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit()
        for order in candidate_orders
    ]
    selected_index = min(
        range(len(fitted_candidates)),
        key=lambda i: float(fitted_candidates[i].bic),
    )
    order = candidate_orders[selected_index]
    cleaned = values.copy()
    detected_positions = set()
    maximum_outliers = max(1, len(values) // 20)

    for _ in range(maximum_outliers):
        fitted = ARIMA(
            cleaned,
            order=order,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit()
        innovations = np.asarray(fitted.resid, dtype=float)
        innovations = innovations - np.median(innovations)
        scale = 1.4826 * np.median(np.abs(innovations - np.median(innovations)))
        scale = max(float(scale), np.finfo(float).eps * max(1.0, float(np.max(np.abs(cleaned)))))

        ar = np.asarray(fitted.arparams, dtype=float)
        ma = np.asarray(fitted.maparams, dtype=float)
        n = len(cleaned)
        psi = np.zeros(n, dtype=float)
        pi = np.zeros(n, dtype=float)
        psi[0] = 1.0
        pi[0] = 1.0

        for k in range(1, n):
            ar_term = ar[k - 1] if k <= len(ar) else 0.0
            psi[k] = ar_term
            for j in range(1, min(len(ma), k) + 1):
                psi[k] += ma[j - 1] * psi[k - j]

            phi_term = -ar[k - 1] if k <= len(ar) else 0.0
            pi[k] = phi_term
            for j in range(1, min(len(ma), k) + 1):
                pi[k] -= ma[j - 1] * pi[k - j]

        best_score = 0.0
        best_kind = None
        best_position = None
        best_effect = None
        start = max(len(ar), len(ma)) + 2

        for position in range(start, n):
            if position in detected_positions:
                continue

            innovative_effect = innovations[position]
            innovative_score = abs(innovative_effect) / scale
            if innovative_score > best_score:
                best_score = innovative_score
                best_kind = "innovative"
                best_position = position
                best_effect = innovative_effect

            signature = pi[: n - position]
            denominator = float(np.dot(signature, signature))
            additive_effect = float(np.dot(signature, innovations[position:]) / denominator)
            additive_score = abs(additive_effect) * np.sqrt(denominator) / scale
            if additive_score > best_score:
                best_score = additive_score
                best_kind = "additive"
                best_position = position
                best_effect = additive_effect

        if best_score <= 3.5:
            break

        detected_positions.add(best_position)
        if best_kind == "additive":
            cleaned[best_position] -= best_effect
        else:
            cleaned[best_position:] -= best_effect * psi[: n - best_position]

    final_fit = ARIMA(
        cleaned,
        order=order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit()
    forecast = np.asarray(final_fit.forecast(steps=horizon), dtype=float).reshape(-1)
    if len(forecast) != horizon or not np.all(np.isfinite(forecast)):
        raise RuntimeError("ARIMA produced a non-finite or incorrectly sized forecast")
    return forecast.tolist()


def markov_switching_autoregression(history, horizon, frequency):
    """Use when a regularly sampled series appears to alternate between a small number of persistent autoregressive regimes."""
    import numpy as np
    from scipy.optimize import minimize

    if len(history) < 12:
        raise NotApplicable(f"needs at least 12 points, got {len(history)}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("history must contain only finite values")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return []

    lagged = y[:-1]
    current = y[1:]
    design = np.column_stack((np.ones(lagged.size), lagged))
    common_intercept, common_ar = np.linalg.lstsq(design, current, rcond=None)[0]
    common_ar = float(np.clip(common_ar, -0.95, 0.95))
    residuals = current - common_intercept - common_ar * lagged
    scale = max(float(np.std(residuals)), float(np.std(y)) * 0.05, 1e-8)
    low, high = np.quantile(y, [0.25, 0.75])

    def unpack(theta):
        intercepts = theta[:2]
        ar = 0.995 * np.tanh(theta[2:4])
        sigma = np.exp(theta[4:6])
        p00 = 1.0 / (1.0 + np.exp(-theta[6]))
        p11 = 1.0 / (1.0 + np.exp(-theta[7]))
        transition = np.array([[p00, 1.0 - p00], [1.0 - p11, p11]])
        return intercepts, ar, sigma, transition

    def stationary_probabilities(transition):
        p00 = transition[0, 0]
        p11 = transition[1, 1]
        p0 = (1.0 - p11) / (2.0 - p00 - p11)
        return np.array([p0, 1.0 - p0])

    def negative_log_likelihood(theta):
        intercepts, ar, sigma, transition = unpack(theta)
        probabilities = stationary_probabilities(transition)
        total = 0.0
        normalizer = np.sqrt(2.0 * np.pi)
        for t in range(1, y.size):
            predicted = probabilities @ transition
            means = intercepts + ar * y[t - 1]
            z = (y[t] - means) / sigma
            densities = np.exp(-0.5 * z * z) / (normalizer * sigma)
            weighted = predicted * densities
            likelihood = np.sum(weighted)
            total -= np.log(likelihood + np.finfo(float).tiny)
            probabilities = weighted / (likelihood + np.finfo(float).tiny)
        return float(total)

    persistence_logit = np.log(0.9 / 0.1)
    initial = np.array([
        low * (1.0 - common_ar),
        high * (1.0 - common_ar),
        np.arctanh(common_ar / 0.995),
        np.arctanh(common_ar / 0.995),
        np.log(scale),
        np.log(scale),
        persistence_logit,
        persistence_logit,
    ])
    bounds = [
        (None, None), (None, None),
        (-4.0, 4.0), (-4.0, 4.0),
        (-20.0, 20.0), (-20.0, 20.0),
        (-8.0, 8.0), (-8.0, 8.0),
    ]
    fitted = minimize(negative_log_likelihood, initial, method="L-BFGS-B", bounds=bounds)
    if not fitted.success:
        raise RuntimeError(f"Markov-switching autoregression fitting failed: {fitted.message}")

    intercepts, ar, sigma, transition = unpack(fitted.x)
    probabilities = stationary_probabilities(transition)
    normalizer = np.sqrt(2.0 * np.pi)
    for t in range(1, y.size):
        predicted = probabilities @ transition
        means = intercepts + ar * y[t - 1]
        z = (y[t] - means) / sigma
        densities = np.exp(-0.5 * z * z) / (normalizer * sigma)
        weighted = predicted * densities
        probabilities = weighted / np.sum(weighted)

    conditional_means = np.full(2, y[-1], dtype=float)
    forecasts = []
    for _ in range(horizon):
        next_probabilities = probabilities @ transition
        weighted_means = np.zeros(2, dtype=float)
        for previous_regime in range(2):
            for next_regime in range(2):
                weight = probabilities[previous_regime] * transition[previous_regime, next_regime]
                regime_mean = intercepts[next_regime] + ar[next_regime] * conditional_means[previous_regime]
                weighted_means[next_regime] += weight * regime_mean
        next_conditional_means = weighted_means / next_probabilities
        forecast = float(np.dot(next_probabilities, next_conditional_means))
        if not np.isfinite(forecast):
            raise FloatingPointError("forecast became non-finite")
        forecasts.append(forecast)
        probabilities = next_probabilities
        conditional_means = next_conditional_means

    return forecasts


def threshold_autoregression(history, horizon, frequency):
    """Use when an autoregressive series changes abruptly between two regimes according to its own recent level."""
    import numpy as np

    if len(history) < 20:
        raise NotApplicable(f"needs at least 20 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty frequency string, got {frequency!r}")
    if not all(np.isfinite(float(value)) for value in history):
        raise NotApplicable("needs only finite history values, got non-finite values")

    values = np.asarray(history, dtype=float)
    order = 2
    targets = values[order:]
    regressors = np.column_stack((np.ones(len(targets)), values[1:-1], values[:-2]))
    threshold_variable = values[1:-1]
    minimum_regime_size = order + 2
    unique_levels = np.unique(threshold_variable)

    if len(unique_levels) < 2:
        raise NotApplicable(f"needs at least 2 distinct threshold values, got {len(unique_levels)}")

    candidate_cut_points = (unique_levels[:-1] + unique_levels[1:]) / 2.0
    eligible_cut_points = []
    for cut_point in candidate_cut_points:
        lower_count = int(np.sum(threshold_variable <= cut_point))
        upper_count = len(threshold_variable) - lower_count
        if lower_count >= minimum_regime_size and upper_count >= minimum_regime_size:
            eligible_cut_points.append(float(cut_point))

    if not eligible_cut_points:
        raise NotApplicable(f"needs at least {minimum_regime_size} observations in each threshold regime, got no eligible partition")

    best_sse = np.inf
    best_cut_point = None
    best_lower_coefficients = None
    best_upper_coefficients = None

    for cut_point in eligible_cut_points:
        lower_mask = threshold_variable <= cut_point
        upper_mask = ~lower_mask
        lower_coefficients = np.linalg.lstsq(regressors[lower_mask], targets[lower_mask], rcond=None)[0]
        upper_coefficients = np.linalg.lstsq(regressors[upper_mask], targets[upper_mask], rcond=None)[0]
        lower_residuals = targets[lower_mask] - regressors[lower_mask] @ lower_coefficients
        upper_residuals = targets[upper_mask] - regressors[upper_mask] @ upper_coefficients
        sse = float(lower_residuals @ lower_residuals + upper_residuals @ upper_residuals)
        if sse < best_sse:
            best_sse = sse
            best_cut_point = cut_point
            best_lower_coefficients = lower_coefficients
            best_upper_coefficients = upper_coefficients

    extended = values.tolist()
    forecasts = []
    for _ in range(horizon):
        coefficients = best_lower_coefficients if extended[-1] <= best_cut_point else best_upper_coefficients
        forecast = float(coefficients[0] + coefficients[1] * extended[-1] + coefficients[2] * extended[-2])
        if not np.isfinite(forecast):
            raise FloatingPointError("threshold autoregression produced a non-finite forecast")
        forecasts.append(forecast)
        extended.append(forecast)

    return forecasts


def smooth_transition_autoregression(history, horizon, frequency):
    """Use when autoregressive behavior changes gradually as the series crosses an observable level."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.special import expit

    if len(history) < 40:
        raise NotApplicable(f"needs 40 points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    lag_order = 2
    target = values[lag_order:]
    design = np.column_stack((
        np.ones(len(values) - lag_order),
        values[1:-1],
        values[:-2],
    ))
    transition_variable = values[1:-1]
    scale = max(float(np.std(transition_variable)), np.finfo(float).eps * max(1.0, float(np.max(np.abs(values)))))

    best_parameters = None
    best_error = np.inf
    thresholds = np.quantile(transition_variable, [0.25, 0.5, 0.75])
    slopes = (0.5, 2.0, 10.0)

    for threshold in thresholds:
        for slope in slopes:
            transition = expit(slope * (transition_variable - threshold) / scale)
            augmented = np.column_stack((design, design * transition[:, None]))
            coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
            parameters = np.concatenate((coefficients, [np.log(slope), threshold]))
            residuals = target - augmented @ coefficients
            error = float(residuals @ residuals)
            if error < best_error:
                best_error = error
                best_parameters = parameters

    dimension = design.shape[1]
    lower = np.concatenate((
        np.full(2 * dimension, -np.inf),
        [np.log(0.01), float(np.min(transition_variable) - scale)],
    ))
    upper = np.concatenate((
        np.full(2 * dimension, np.inf),
        [np.log(100.0), float(np.max(transition_variable) + scale)],
    ))

    def residual_function(parameters):
        base = parameters[:dimension]
        regime_difference = parameters[dimension:2 * dimension]
        slope = np.exp(parameters[-2])
        threshold = parameters[-1]
        transition = expit(slope * (transition_variable - threshold) / scale)
        fitted = design @ base + transition * (design @ regime_difference)
        return target - fitted

    fitted_parameters = least_squares(
        residual_function,
        best_parameters,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=5000,
    ).x

    base = fitted_parameters[:dimension]
    regime_difference = fitted_parameters[dimension:2 * dimension]
    slope = float(np.exp(fitted_parameters[-2]))
    threshold = float(fitted_parameters[-1])
    extended = values.tolist()
    forecast = []

    for _ in range(horizon):
        predictors = np.array([1.0, extended[-1], extended[-2]])
        transition = float(expit(slope * (extended[-1] - threshold) / scale))
        prediction = float(predictors @ base + transition * (predictors @ regime_difference))
        if not np.isfinite(prediction):
            raise FloatingPointError("smooth-transition autoregression produced a non-finite forecast")
        forecast.append(prediction)
        extended.append(prediction)

    return forecast


def bats(history, horizon, frequency):
    """Use when a positive series has stable trend, one seasonality, and autocorrelated errors after transformation."""
    import math
    import numpy as np
    from scipy import optimize, special, stats
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    periods = {
        "1 minute": 60,
        "1 hour": 24,
        "hourly": 24,
        "1 day": 7,
        "daily": 7,
        "1 week": 52,
        "weekly": 52,
        "1 month": 12,
        "monthly": 12,
        "1 quarter": 4,
        "quarterly": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported single-season frequency, got {frequency}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in history):
        raise NotApplicable("needs finite positive history values, got non-positive or non-finite values")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    lambda_result = optimize.minimize_scalar(
        lambda lam: -stats.boxcox_llf(lam, values),
        bounds=(0.0, 1.0),
        method="bounded",
    )
    boxcox_lambda = float(lambda_result.x)
    transformed = special.boxcox(values, boxcox_lambda)

    ets_model = ETSModel(
        transformed,
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=period,
        initialization_method="estimated",
    )
    ets_result = ets_model.fit(disp=False)
    ets_forecast = np.asarray(ets_result.forecast(horizon), dtype=float)

    arma_model = SARIMAX(
        np.asarray(ets_result.resid, dtype=float),
        order=(1, 0, 1),
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    arma_result = arma_model.fit(disp=False)
    transformed_forecast = ets_forecast + np.asarray(arma_result.forecast(horizon), dtype=float)

    inverse_argument = 1.0 + boxcox_lambda * transformed_forecast
    inverse_argument = np.maximum(inverse_argument, np.finfo(float).tiny)
    log_forecast = np.log(inverse_argument) / boxcox_lambda
    log_forecast = np.clip(
        log_forecast,
        math.log(np.finfo(float).tiny),
        math.log(np.finfo(float).max) - 1.0,
    )
    return [float(value) for value in np.exp(log_forecast)]


def tbats(history, horizon, frequency):
    """Use when positive data contain stable, potentially multiple or non-integer seasonal cycles."""
    import math
    import numpy as np
    from scipy.optimize import minimize

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    if not isinstance(frequency, str):
        raise NotApplicable(f"needs a supported frequency, got {frequency!r}")

    parts = frequency.strip().lower().split()
    if len(parts) != 2 or not parts[0].replace(".", "", 1).isdigit():
        raise NotApplicable(f"needs a frequency such as '1 hour' or '1 day', got {frequency!r}")

    amount = float(parts[0])
    unit = parts[1].rstrip("s")
    supported = {"minute", "hour", "day", "week", "month", "quarter"}
    if amount <= 0.0 or unit not in supported:
        raise NotApplicable(f"needs a positive minute, hour, day, week, month, or quarter frequency, got {frequency!r}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("history must contain only finite values")
    if np.any(y <= 0.0):
        raise NotApplicable(f"needs strictly positive values, got {int(np.sum(y <= 0.0))} non-positive values")

    candidates = {
        "minute": [1440.0 / amount, 10080.0 / amount, 525960.0 / amount],
        "hour": [24.0 / amount, 168.0 / amount, 8766.0 / amount],
        "day": [7.0 / amount, 365.25 / amount],
        "week": [365.25 / (7.0 * amount)],
        "month": [12.0 / amount],
        "quarter": [4.0 / amount],
    }[unit]
    candidates = [p for p in candidates if p >= 2.0]
    if not candidates:
        raise NotApplicable(f"needs a sampling interval shorter than its seasonal cycle, got {frequency!r}")

    needed = int(math.ceil(2.0 * candidates[0]))
    if len(y) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    periods = [p for p in candidates if len(y) >= int(math.ceil(2.0 * p))]
    harmonic_budget = max(1, len(y) // (20 * len(periods)))
    harmonics = [max(1, min(5, int(math.floor(p / 2.0)), harmonic_budget)) for p in periods]

    n = len(y)
    t = np.arange(n, dtype=float)
    columns = [np.ones(n), t]
    angles = []
    for period, count in zip(periods, harmonics):
        period_angles = []
        for k in range(1, count + 1):
            theta = 2.0 * np.pi * k / period
            columns.extend([np.cos(theta * t), np.sin(theta * t)])
            period_angles.append(theta)
        angles.append(period_angles)
    design = np.column_stack(columns)
    log_y_sum = float(np.log(y).sum())

    def transform(lam):
        if abs(lam) < 1e-7:
            return np.log(y)
        return np.expm1(lam * np.log(y)) / lam

    def initialize(z):
        coef = np.linalg.lstsq(design, z, rcond=None)[0]
        level = float(coef[0])
        trend = float(coef[1])
        states = []
        offset = 2
        for count in harmonics:
            group = []
            for _ in range(count):
                group.append([float(coef[offset]), float(coef[offset + 1])])
                offset += 2
            states.append(group)
        return level, trend, states

    def run(params, retain=False):
        lam, alpha, beta, phi, rho = params[:5]
        gamma = params[5:].reshape(len(periods), 2)
        z = transform(lam)
        level, trend, states = initialize(z)
        initial_season = sum(pair[0] for group in states for pair in group)
        error_state = float(z[0] - level - initial_season)
        squared_error = error_state * error_state

        for i in range(1, n):
            base = level + phi * trend
            seasonal = 0.0
            for j, group in enumerate(states):
                for k, pair in enumerate(group):
                    c, s = pair
                    theta = angles[j][k]
                    rotated_c = c * math.cos(theta) + s * math.sin(theta)
                    rotated_s = -c * math.sin(theta) + s * math.cos(theta)
                    pair[0] = rotated_c
                    pair[1] = rotated_s
                    seasonal += rotated_c
            prediction = base + seasonal + rho * error_state
            innovation = float(z[i] - prediction)
            squared_error += innovation * innovation
            level = base + alpha * innovation
            trend = phi * trend + beta * innovation
            for j, group in enumerate(states):
                for pair in group:
                    pair[0] += gamma[j, 0] * innovation
                    pair[1] += gamma[j, 1] * innovation
            error_state = rho * error_state + innovation

        if retain:
            return lam, phi, rho, level, trend, states, error_state
        variance = max(squared_error / n, np.finfo(float).tiny)
        return n * math.log(variance) - 2.0 * (lam - 1.0) * log_y_sum

    initial = np.array([0.0, 0.2, 0.02, 0.98, 0.0] + [0.05, 0.0] * len(periods), dtype=float)
    bounds = [(-1.0, 2.0), (0.0001, 0.9999), (0.0, 0.5), (0.8, 1.0), (-0.95, 0.95)]
    bounds.extend([(-0.5, 0.5), (-0.5, 0.5)] * len(periods))
    fitted = minimize(run, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-10})
    if not fitted.success:
        raise RuntimeError(f"TBATS optimization failed: {fitted.message}")

    lam, phi, rho, level, trend, states, error_state = run(fitted.x, retain=True)
    transformed_forecast = []
    for _ in range(horizon):
        level = level + phi * trend
        trend = phi * trend
        seasonal = 0.0
        for j, group in enumerate(states):
            for k, pair in enumerate(group):
                c, s = pair
                theta = angles[j][k]
                pair[0] = c * math.cos(theta) + s * math.sin(theta)
                pair[1] = -c * math.sin(theta) + s * math.cos(theta)
                seasonal += pair[0]
        error_state = rho * error_state
        transformed_forecast.append(level + seasonal + error_state)

    z_forecast = np.asarray(transformed_forecast, dtype=float)
    if abs(lam) < 1e-7:
        result = np.exp(np.clip(z_forecast, -700.0, 700.0))
    else:
        inverse_base = np.maximum(1.0 + lam * z_forecast, np.finfo(float).tiny)
        result = np.exp(np.clip(np.log(inverse_base) / lam, -700.0, 700.0))
    return [float(value) for value in result]


def dynamic_harmonic_regression_arima(history, horizon, frequency):
    """Use when seasonal shapes repeat smoothly at fixed periods and the remaining errors have ARIMA-like autocorrelation."""
    import math
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    seasonal_candidates = {
        "15 minutes": (96.0, 672.0),
        "30 minutes": (48.0, 336.0),
        "1 hour": (24.0, 168.0),
        "1 day": (7.0, 365.25),
        "1 week": (52.1775,),
        "1 month": (12.0,),
        "1 quarter": (4.0,),
    }
    if frequency not in seasonal_candidates:
        raise NotApplicable(f"needs a supported seasonal frequency, got {frequency}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise NotApplicable("needs finite history values, got non-finite values")

    minimum = max(12, math.ceil(2 * seasonal_candidates[frequency][0]))
    if len(history) < minimum:
        raise NotApplicable(f"needs {minimum} points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if horizon == 0:
        return []

    periods = [
        period
        for period in seasonal_candidates[frequency]
        if len(history) >= math.ceil(2 * period)
    ]
    observed_time = np.arange(len(history), dtype=float)
    future_time = np.arange(len(history), len(history) + horizon, dtype=float)

    observed_terms = []
    future_terms = []
    for period in periods:
        harmonics = min(3, int(math.floor((period - 1.0) / 2.0)))
        for harmonic in range(1, harmonics + 1):
            observed_angle = 2.0 * np.pi * harmonic * observed_time / period
            future_angle = 2.0 * np.pi * harmonic * future_time / period
            observed_terms.extend((np.sin(observed_angle), np.cos(observed_angle)))
            future_terms.extend((np.sin(future_angle), np.cos(future_angle)))

    observed_exog = np.column_stack(observed_terms)
    future_exog = np.column_stack(future_terms)
    model = SARIMAX(
        y,
        exog=observed_exog,
        order=(1, 1, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = np.asarray(
        fitted.get_forecast(steps=horizon, exog=future_exog).predicted_mean,
        dtype=float,
    )
    if not np.all(np.isfinite(forecast)):
        raise ValueError("dynamic harmonic regression produced non-finite forecasts")
    return forecast.tolist()


def mstl_ets(history, horizon, frequency):
    """Use when a sufficiently long series contains several stable additive seasonal periods."""
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import STL

    periods_by_frequency = {
        "1 minute": (60, 1440, 10080),
        "5 minutes": (12, 288, 2016),
        "15 minutes": (96, 672),
        "30 minutes": (48, 336),
        "1 hour": (24, 168),
        "1 day": (7, 365),
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods_by_frequency:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    periods = periods_by_frequency[normalized_frequency]
    required = 2 * max(periods)
    if len(history) < required:
        raise NotApplicable(f"needs {required} points, got {len(history)}")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return []

    adjusted = np.asarray(history, dtype=float)
    seasonal_components = []
    for period in periods:
        decomposition = STL(adjusted, period=period, robust=True).fit()
        seasonal_components.append((period, decomposition.seasonal))
        adjusted = adjusted - decomposition.seasonal

    ets = ExponentialSmoothing(
        adjusted,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)
    forecast = np.asarray(ets.forecast(horizon), dtype=float)

    for period, component in seasonal_components:
        repetitions = (horizon + period - 1) // period
        forecast += np.tile(component[-period:], repetitions)[:horizon]

    if not np.all(np.isfinite(forecast)):
        raise ValueError("forecast contains non-finite values")
    return [float(value) for value in forecast]


def random_forest_lag_forecast(history, horizon, frequency):
    """Use when nonlinear lag, rolling, and calendar relationships are expected to recur within the historical target range."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    period = periods[normalized_frequency]
    needed = max(2 * period, 20)
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")
    if not all(np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    values = np.asarray(history, dtype=float)
    lag_count = min(period, 24)
    rolling_windows = sorted(set((3, min(7, lag_count), lag_count)))

    def features(series, target_index):
        lag_values = [series[-lag] for lag in range(1, lag_count + 1)]
        rolling_values = []
        for window in rolling_windows:
            segment = np.asarray(series[-window:], dtype=float)
            rolling_values.extend([
                float(np.mean(segment)),
                float(np.std(segment)),
                float(np.min(segment)),
                float(np.max(segment)),
            ])
        phase = 2.0 * np.pi * (target_index % period) / period
        calendar_values = [np.sin(phase), np.cos(phase)]
        return lag_values + rolling_values + calendar_values

    training_features = []
    training_targets = []
    for target_index in range(lag_count, len(values)):
        training_features.append(features(values[:target_index], target_index))
        training_targets.append(values[target_index])

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=0,
        n_jobs=1,
    )
    model.fit(np.asarray(training_features, dtype=float), np.asarray(training_targets, dtype=float))

    extended = values.tolist()
    forecast = []
    for step in range(horizon):
        target_index = len(values) + step
        prediction = float(model.predict(np.asarray([features(extended, target_index)], dtype=float))[0])
        if not np.isfinite(prediction):
            raise ValueError("random forest produced a non-finite forecast")
        forecast.append(prediction)
        extended.append(prediction)
    return forecast


def xgboost_lag_forecast(history, horizon, frequency):
    """Use when lag, rolling, and calendar features exhibit repeatable nonlinear threshold effects."""
    import math

    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if horizon == 0:
        return []
    if not isinstance(frequency, str):
        raise NotApplicable(f"needs a supported frequency string, got {frequency}")

    parts = frequency.strip().lower().split()
    if len(parts) != 2 or not parts[0].isdigit():
        raise NotApplicable(f"needs an integer frequency such as '1 hour' or '1 day', got {frequency}")

    step = int(parts[0])
    unit = parts[1].rstrip("s")
    cycle_sizes = {"minute": 1440, "hour": 24, "day": 7, "week": 52, "month": 12}
    if step <= 0 or unit not in cycle_sizes or cycle_sizes.get(unit, 0) % step != 0:
        raise NotApplicable(f"needs a positive minute, hour, day, week, or month frequency dividing its calendar cycle, got {frequency}")

    period = cycle_sizes[unit] // step
    required = max(24, 2 * period)
    if len(history) < required:
        raise NotApplicable(f"needs {required} points, got {len(history)}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    import numpy as np
    from xgboost import XGBRegressor

    values = [float(value) for value in history]
    lag_steps = sorted(set([1, 2, 3, period]))
    rolling_windows = sorted(set([3, 7, period]))
    max_lookback = max(lag_steps + rolling_windows)

    def make_features(series, time_index):
        row = [series[-lag] for lag in lag_steps]
        for window in rolling_windows:
            segment = np.asarray(series[-window:], dtype=float)
            row.extend([float(segment.mean()), float(segment.std(ddof=0)), float(segment.min()), float(segment.max())])
        angle = 2.0 * math.pi * (time_index % period) / period
        row.extend([math.sin(angle), math.cos(angle), float(time_index), math.log1p(time_index)])
        return row

    features = []
    targets = []
    for target_index in range(max_lookback, len(values)):
        features.append(make_features(values[:target_index], target_index))
        targets.append(values[target_index])

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=3.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=2.0,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(np.asarray(features, dtype=float), np.asarray(targets, dtype=float))

    extended = list(values)
    forecast = []
    for time_index in range(len(values), len(values) + horizon):
        row = np.asarray([make_features(extended, time_index)], dtype=float)
        prediction = float(model.predict(row)[0])
        if not math.isfinite(prediction):
            raise RuntimeError("xgboost_lag_forecast produced a non-finite prediction")
        forecast.append(prediction)
        extended.append(prediction)

    return forecast


def lightgbm_lag_forecast(history, horizon, frequency):
    """Use when a long series has nonlinear relationships among recent lags, seasonal lags, and calendar-position features."""
    import numpy as np
    from lightgbm import LGBMRegressor

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    if frequency not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    period = periods[frequency]
    required = 2 * period + 128
    if len(history) < required:
        raise NotApplicable(f"needs {required} points, got {len(history)}")
    if not np.all(np.isfinite(np.asarray(history, dtype=float))):
        raise ValueError("history must contain only finite values")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    short_lags = list(range(1, min(12, period) + 1))
    lags = sorted(set(short_lags + [period, 2 * period]))
    windows = sorted(set([3, min(12, period), period]))
    first_training_index = max(lags + windows)
    scale = float(max(1, len(values) - 1))

    def make_features(series, index):
        row = [float(series[index - lag]) for lag in lags]
        for window in windows:
            segment = np.asarray(series[index - window:index], dtype=float)
            row.extend([
                float(np.mean(segment)),
                float(np.std(segment)),
                float(np.min(segment)),
                float(np.max(segment)),
            ])
        row.extend([
            float(series[index - 1] - series[index - 2]),
            float(series[index - period] - series[index - period - 1]),
            float(index / scale),
            float(np.sin(2.0 * np.pi * index / period)),
            float(np.cos(2.0 * np.pi * index / period)),
            float(np.sin(4.0 * np.pi * index / period)),
            float(np.cos(4.0 * np.pi * index / period)),
        ])
        return row

    x_train = np.asarray([
        make_features(values, index)
        for index in range(first_training_index, len(values))
    ], dtype=float)
    y_train = values[first_training_index:]

    model = LGBMRegressor(
        boosting_type="gbdt",
        objective="regression_l2",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=0,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    extended = values.tolist()
    forecast = []
    for _ in range(horizon):
        features = np.asarray([make_features(extended, len(extended))], dtype=float)
        prediction = float(model.predict(features)[0])
        if not np.isfinite(prediction):
            raise ValueError("LightGBM produced a non-finite forecast")
        extended.append(prediction)
        forecast.append(prediction)
    return forecast


def bayesian_online_changepoint_forecast(history, horizon, frequency):
    """Use when abrupt changes separate locally stable levels and a constant changepoint hazard is credible."""
    import math
    import numpy as np

    intervals = {
        "1 minute": 60.0,
        "5 minutes": 300.0,
        "15 minutes": 900.0,
        "30 minutes": 1800.0,
        "1 hour": 3600.0,
        "6 hours": 21600.0,
        "12 hours": 43200.0,
        "1 day": 86400.0,
        "1 week": 604800.0,
    }
    if len(history) < 4:
        raise NotApplicable(f"needs at least 4 points, got {len(history)}")
    if frequency not in intervals:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    if not all(math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    values = np.asarray(history, dtype=float)
    prior_mean = float(np.mean(values))
    prior_kappa = 0.01
    prior_alpha = 2.0
    prior_beta = max(float(np.var(values, ddof=1)), 1e-6)
    expected_run = max(2.0, 30.0 * 86400.0 / intervals[frequency])
    hazard = 1.0 / expected_run

    run_prob = np.array([1.0])
    means = np.array([prior_mean])
    kappas = np.array([prior_kappa])
    alphas = np.array([prior_alpha])
    betas = np.array([prior_beta])

    for observation in values:
        degrees = 2.0 * alphas
        scale_squared = betas * (kappas + 1.0) / (alphas * kappas)
        log_predictive = np.array([
            math.lgamma((degree + 1.0) / 2.0)
            - math.lgamma(degree / 2.0)
            - 0.5 * math.log(degree * math.pi * scale)
            - 0.5 * (degree + 1.0) * math.log1p(
                (observation - mean) ** 2 / (degree * scale)
            )
            for degree, scale, mean in zip(degrees, scale_squared, means)
        ])
        log_previous = np.log(run_prob)
        growth_logs = log_previous + math.log1p(-hazard) + log_predictive
        change_terms = log_previous + math.log(hazard) + log_predictive
        change_max = float(np.max(change_terms))
        change_log = change_max + math.log(float(np.sum(np.exp(change_terms - change_max))))
        joint_logs = np.concatenate(([change_log], growth_logs))
        normalizer_max = float(np.max(joint_logs))
        run_prob = np.exp(joint_logs - normalizer_max)
        run_prob /= float(np.sum(run_prob))

        new_kappas = kappas + 1.0
        new_means = (kappas * means + observation) / new_kappas
        new_alphas = alphas + 0.5
        new_betas = betas + kappas * (observation - means) ** 2 / (2.0 * new_kappas)

        reset_kappa = prior_kappa + 1.0
        reset_mean = (prior_kappa * prior_mean + observation) / reset_kappa
        reset_alpha = prior_alpha + 0.5
        reset_beta = prior_beta + prior_kappa * (observation - prior_mean) ** 2 / (2.0 * reset_kappa)
        kappas = np.concatenate(([reset_kappa], new_kappas))
        means = np.concatenate(([reset_mean], new_means))
        alphas = np.concatenate(([reset_alpha], new_alphas))
        betas = np.concatenate(([reset_beta], new_betas))

    forecast_level = float(np.dot(run_prob, means))
    return [forecast_level for _ in range(horizon)]


def pelt_segment_then_forecast(history, horizon, frequency):
    """Use when abrupt level changes make the most recent stable regime more relevant than older observations."""
    import numpy as np

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points, got {len(history)}")
    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise NotApplicable(f"needs a non-negative integer horizon, got {horizon}")

    x = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(x)):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon == 0:
        return []

    n = len(x)
    min_segment = 4
    scale = float(np.max(np.abs(x)))
    z = x / scale if scale > 0.0 else x.copy()

    prefix = np.concatenate(([0.0], np.cumsum(z)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(z * z)))

    def segment_cost(start, end):
        length = end - start
        total = prefix[end] - prefix[start]
        squared = prefix_sq[end] - prefix_sq[start]
        return max(0.0, squared - total * total / length)

    variance = float(np.var(z))
    penalty = max(variance, np.finfo(float).eps) * 2.0 * np.log(n)
    objective = np.full(n + 1, np.inf)
    previous = np.full(n + 1, -1, dtype=int)
    objective[0] = -penalty
    candidates = [0]

    for end in range(min_segment, n + 1):
        eligible = [start for start in candidates if end - start >= min_segment]
        values = [
            objective[start] + segment_cost(start, end) + penalty
            for start in eligible
        ]
        best_index = int(np.argmin(values))
        best_start = eligible[best_index]
        objective[end] = values[best_index]
        previous[end] = best_start

        candidates = [
            start
            for start in eligible
            if objective[start] + segment_cost(start, end) <= objective[end]
        ]
        new_candidate = end - min_segment + 1
        if new_candidate >= min_segment and np.isfinite(objective[new_candidate]):
            candidates.append(new_candidate)

    latest_start = int(previous[n])
    latest = z[latest_start:]
    normalized_level = float(np.mean(latest))
    normalized_level = min(1.0, max(-1.0, normalized_level))
    level = float(normalized_level * scale) if scale > 0.0 else 0.0
    return [level for _ in range(horizon)]


def robust_stl_ets(history, horizon, frequency):
    """Use when a seasonal series has a locally smooth trend and seasonality with occasional contaminated observations."""
    import numpy as np
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
        "15 minutes": 96,
        "30 minutes": 48,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported seasonal frequency, got {frequency}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    decomposition = STL(values, period=period, robust=True).fit()
    weights = np.asarray(decomposition.weights, dtype=float)
    adjusted = weights * (values - decomposition.seasonal) + (1.0 - weights) * decomposition.trend

    model = ExponentialSmoothing(
        adjusted,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)
    adjusted_forecast = np.asarray(model.forecast(horizon), dtype=float)

    last_seasonal_cycle = np.asarray(decomposition.seasonal[-period:], dtype=float)
    seasonal_forecast = np.resize(last_seasonal_cycle, horizon)
    forecast = adjusted_forecast + seasonal_forecast
    if not np.all(np.isfinite(forecast)):
        raise ValueError("forecast contains non-finite values")
    return [float(value) for value in forecast]


def median_seasonal_profile_forecast(history, horizon, frequency):
    """Use when seasonality has a stable phase and occasional cycles may contain outliers."""
    import math
    import statistics

    periods = {
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    if frequency not in periods:
        raise NotApplicable(f"needs a supported frequency {tuple(periods)}, got {frequency}")
    period = periods[frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    values = [float(value) for value in history]
    profile = [statistics.median(values[position::period]) for position in range(period)]
    profile_center = statistics.median(profile)
    recent_level = statistics.median(values[-period:])
    forecast = [
        float(recent_level + profile[(len(values) + step) % period] - profile_center)
        for step in range(horizon)
    ]
    if any(not math.isfinite(value) for value in forecast):
        raise ArithmeticError("median seasonal profile produced non-finite forecasts")
    return forecast


def state_space_trigonometric_harmonics(history, horizon, frequency):
    """Use when a few fixed seasonal harmonics recur while their amplitudes drift gradually over time."""
    import numpy as np
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    periods = {
        "1 minute": 1440,
        "15 minutes": 96,
        "30 minutes": 48,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported regular frequency, got {frequency!r}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if not all(np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    harmonics = min(3, period // 2)
    model = UnobservedComponents(
        values,
        level="local level",
        stochastic_level=True,
        irregular=True,
        freq_seasonal=[{"period": period, "harmonics": harmonics}],
        stochastic_freq_seasonal=[True],
    )
    fitted = model.fit(disp=False)
    forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float).reshape(-1)
    if forecast.size != horizon:
        raise ValueError(f"expected {horizon} forecasts, got {forecast.size}")
    if not np.all(np.isfinite(forecast)):
        raise ValueError("state-space forecast produced non-finite values")
    return [float(value) for value in forecast]


def quantile_regression_forecast(history, horizon, frequency):
    """Use when conditional quantiles depend on recent lags and a stable recurring calendar cycle."""
    import math
    import numpy as np
    from sklearn.linear_model import QuantileRegressor
    from sklearn.preprocessing import StandardScaler

    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if not all(math.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    parts = frequency.strip().lower().split()
    supported_cycles = {
        "minute": 60,
        "hour": 24,
        "day": 7,
        "week": 52,
        "month": 12,
        "quarter": 4,
        "year": 1,
    }
    unit = parts[1].rstrip("s") if len(parts) == 2 else ""
    if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) <= 0 or unit not in supported_cycles:
        raise NotApplicable(f"needs a supported positive frequency, got {frequency!r}")

    step = int(parts[0])
    cycle = supported_cycles[unit]
    period = max(1, int(round(cycle / step)))
    lags = sorted(set([1, 2, 3] + ([period] if period > 1 else [])))
    max_lag = max(lags)
    needed = max_lag + 20
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    values = np.asarray(history, dtype=float)

    def predictors(series, time_index):
        lag_values = [series[-lag] for lag in lags]
        if period > 1:
            angle = 2.0 * math.pi * time_index / period
            calendar_values = [math.sin(angle), math.cos(angle)]
        else:
            calendar_values = [0.0, 1.0]
        return lag_values + calendar_values + [time_index / len(values)]

    rows = []
    targets = []
    for time_index in range(max_lag, len(values)):
        rows.append(predictors(values[:time_index], time_index))
        targets.append(values[time_index])

    design = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    scaler = StandardScaler()
    scaled_design = scaler.fit_transform(design)
    quantiles = (0.1, 0.5, 0.9)
    models = []
    for quantile in quantiles:
        model = QuantileRegressor(quantile=quantile, alpha=0.01, solver="highs")
        model.fit(scaled_design, target)
        models.append(model)

    extended = values.tolist()
    forecast = []
    for time_index in range(len(values), len(values) + horizon):
        row = np.asarray([predictors(extended, time_index)], dtype=float)
        scaled_row = scaler.transform(row)
        quantile_predictions = [float(model.predict(scaled_row)[0]) for model in models]
        central_prediction = quantile_predictions[1]
        if not math.isfinite(central_prediction):
            raise ValueError("quantile regression produced a non-finite forecast")
        forecast.append(central_prediction)
        extended.append(central_prediction)

    return forecast


def forecast_residual_bootstrap(history, horizon, frequency):
    """Use when autoregressive structure is stable and historical forecast errors represent future error dependence and scale."""
    import numpy as np

    if len(history) < 8:
        raise NotApplicable(f"needs at least 8 points, got {len(history)}")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        bad = int(np.size(y) - np.count_nonzero(np.isfinite(y)))
        raise NotApplicable(f"needs only finite history values, got {bad} non-finite values")
    if horizon == 0:
        return []

    n = y.size
    lag = min(12, max(1, n // 5))
    rows = n - lag
    design = np.empty((rows, lag + 1), dtype=float)
    design[:, 0] = 1.0
    targets = y[lag:]
    for row, t in enumerate(range(lag, n)):
        design[row, 1:] = y[t - lag:t][::-1]

    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    residuals = targets - design @ coefficients
    residuals = residuals - residuals.mean()
    residual_count = residuals.size
    block_length = max(1, int(round(np.sqrt(residual_count))))

    paths = np.empty((residual_count, horizon), dtype=float)
    for path_index in range(residual_count):
        simulated = y.tolist()
        for step in range(horizon):
            block_number = step // block_length
            within_block = step % block_length
            offset = block_number * (block_number + 1) // 2
            block_start = (path_index + offset) % residual_count
            innovation = residuals[(block_start + within_block) % residual_count]
            recent = np.asarray(simulated[-lag:][::-1], dtype=float)
            point_forecast = coefficients[0] + coefficients[1:] @ recent
            value = float(point_forecast + innovation)
            simulated.append(value)
            paths[path_index, step] = value

    forecast = np.median(paths, axis=0)
    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("forecast produced non-finite values")
    return forecast.astype(float).tolist()


def ltsf_dlinear(history, horizon, frequency):
    """Use when trend and residual dynamics can each be projected linearly over the requested horizon."""
    import numpy as np

    needed = max(12, horizon + 8)
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a nonnegative horizon, got {horizon}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a nonempty frequency string, got {frequency!r}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise NotApplicable("needs only finite history values, got non-finite values")
    if horizon == 0:
        return []

    kernel = min(25, len(values) if len(values) % 2 == 1 else len(values) - 1)
    half = kernel // 2
    padded = np.pad(values, (half, half), mode="edge")
    trend = np.convolve(padded, np.ones(kernel, dtype=float) / kernel, mode="valid")
    remainder = values - trend

    lookback = min(24, max(4, (len(values) - horizon) // 2))
    sample_count = len(values) - lookback - horizon + 1
    inputs_trend = np.stack([trend[i:i + lookback] for i in range(sample_count)])
    inputs_remainder = np.stack([remainder[i:i + lookback] for i in range(sample_count)])
    targets_trend = np.stack([trend[i + lookback:i + lookback + horizon] for i in range(sample_count)])
    targets_remainder = np.stack([remainder[i + lookback:i + lookback + horizon] for i in range(sample_count)])

    design_trend = np.column_stack((inputs_trend, np.ones(sample_count)))
    design_remainder = np.column_stack((inputs_remainder, np.ones(sample_count)))
    weights_trend = np.linalg.lstsq(design_trend, targets_trend, rcond=None)[0]
    weights_remainder = np.linalg.lstsq(design_remainder, targets_remainder, rcond=None)[0]

    last_trend = np.append(trend[-lookback:], 1.0)
    last_remainder = np.append(remainder[-lookback:], 1.0)
    forecast = last_trend @ weights_trend + last_remainder @ weights_remainder
    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("DLinear projection produced non-finite values")
    return [float(value) for value in forecast]


def informer(history, horizon, frequency):
    """Use when long, regularly sampled histories are driven by a small number of dominant long-range interactions."""
    import numpy as np

    parts = frequency.strip().lower().split()
    if len(parts) != 2 or parts[0] not in {"1", "2", "3", "4", "5", "6", "10", "12", "15", "20", "30"} or parts[1].rstrip("s") not in {"minute", "hour", "day", "week", "month", "quarter", "year"}:
        raise NotApplicable(f"needs a supported regular frequency, got {frequency}")

    step = int(parts[0])
    unit = parts[1].rstrip("s")
    cycle = {"minute": 1440, "hour": 24, "day": 7, "week": 52, "month": 12, "quarter": 4, "year": 1}[unit]
    period = max(1, int(round(cycle / step)))
    needed = max(32, 2 * period)
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return []

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("history must contain only finite values")

    n = len(y)
    t = np.arange(n, dtype=float)
    design = np.column_stack((np.ones(n), t))
    trend_coef = np.linalg.lstsq(design, y, rcond=None)[0]
    detrended = y - design @ trend_coef

    phase = np.arange(n) % period
    seasonal = np.zeros(period, dtype=float)
    for p in range(period):
        seasonal[p] = np.mean(detrended[phase == p])
    seasonal -= np.mean(seasonal)

    fitted_base = design @ trend_coef + seasonal[phase]
    residual = y - fitted_base
    scale = np.std(residual)
    if scale == 0.0:
        scale = max(np.std(y), 1.0)
    r = residual / scale

    def embedding(values, positions):
        positions = np.asarray(positions, dtype=float)
        phase_angle = 2.0 * np.pi * positions / period
        slow_angle = 2.0 * np.pi * positions / max(n, period)
        return np.column_stack((
            values,
            np.sin(phase_angle),
            np.cos(phase_angle),
            np.sin(2.0 * phase_angle),
            np.cos(2.0 * phase_angle),
            np.sin(slow_angle),
            np.cos(slow_angle),
            positions / max(1.0, n - 1.0)
        ))

    def projection(d, offset):
        rows = np.arange(1, d + 1, dtype=float)[:, None]
        cols = np.arange(1, d + 1, dtype=float)[None, :]
        return np.sin(rows * cols + offset) / np.sqrt(d)

    def probsparse_attention(x, causal):
        length, d = x.shape
        q = x @ projection(d, 0.17)
        k = x @ projection(d, 0.53)
        v = x @ projection(d, 0.89)
        logits = q @ k.T / np.sqrt(d)
        if causal:
            logits[np.triu_indices(length, 1)] = -np.inf
        sparsity = np.empty(length, dtype=float)
        for i in range(length):
            usable = logits[i, :i + 1] if causal else logits[i]
            sparsity[i] = np.max(usable) - np.mean(usable)
        query_count = min(length, max(1, int(np.ceil(np.log2(length + 1)))))
        selected = np.argpartition(sparsity, -query_count)[-query_count:]
        if causal:
            context = np.cumsum(v, axis=0) / np.arange(1, length + 1)[:, None]
        else:
            context = np.repeat(np.mean(v, axis=0, keepdims=True), length, axis=0)
        key_count = min(length, max(1, int(np.ceil(np.log2(length + 1)))))
        for i in selected:
            usable_count = i + 1 if causal else length
            row = logits[i, :usable_count]
            chosen_count = min(key_count, usable_count)
            chosen = np.argpartition(row, -chosen_count)[-chosen_count:]
            weights = np.exp(row[chosen] - np.max(row[chosen]))
            weights /= np.sum(weights)
            context[i] = weights @ v[chosen]
        return context

    encoder = embedding(r, t)
    encoded_residual = r.copy()
    for _ in range(2):
        encoder = np.tanh(encoder + probsparse_attention(encoder, True))
        if len(encoder) < 4:
            break
        smooth = encoder.copy()
        smooth[1:-1] = 0.25 * encoder[:-2] + 0.5 * encoder[1:-1] + 0.25 * encoder[2:]
        pair_count = len(smooth) // 2
        encoder = np.maximum(smooth[:2 * pair_count:2], smooth[1:2 * pair_count:2])
        encoded_residual = 0.5 * (encoded_residual[:2 * pair_count:2] + encoded_residual[1:2 * pair_count:2])

    future_t = np.arange(n, n + horizon, dtype=float)
    decoder = embedding(np.zeros(horizon, dtype=float), future_t)
    decoder = np.tanh(decoder + probsparse_attention(decoder, True))

    d = encoder.shape[1]
    queries = decoder @ projection(d, 1.31)
    keys = encoder @ projection(d, 1.73)
    cross_logits = queries @ keys.T / np.sqrt(d)
    key_count = min(len(encoder), max(1, int(np.ceil(np.log2(len(encoder) + 1)))))
    attended_residual = np.empty(horizon, dtype=float)
    for i in range(horizon):
        chosen = np.argpartition(cross_logits[i], -key_count)[-key_count:]
        weights = np.exp(cross_logits[i, chosen] - np.max(cross_logits[i, chosen]))
        weights /= np.sum(weights)
        attended_residual[i] = weights @ encoded_residual[chosen]

    future_base = trend_coef[0] + trend_coef[1] * future_t + seasonal[(np.arange(n, n + horizon) % period)]
    forecast = future_base + scale * attended_residual
    return [float(value) for value in forecast]


def autoformer(history, horizon, frequency):
    """Use when observations have stable periodic sub-series dependencies and a decomposable trend-seasonal structure."""
    import math
    import numpy as np
    import torch

    periods = {
        "15 minutes": 96,
        "30 minutes": 48,
        "1 hour": 24,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4,
    }
    key = str(frequency).strip().lower()
    if key not in periods:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")
    if horizon < 1:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")

    period = periods[key]
    needed = 4 * period
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("history must contain only finite scalar values")

    dtype = torch.float64
    series = torch.tensor(values, dtype=dtype)
    center = series.mean()
    scale = series.std(unbiased=False)
    if scale.item() == 0.0:
        scale = torch.tensor(1.0, dtype=dtype)
    normalized = (series - center) / scale

    context = 2 * period
    dimension = 12
    top_k = max(1, min(period, int(math.ceil(math.log(period + 1)))))

    correlations = []
    for lag in range(1, period + 1):
        left = normalized[lag:]
        right = normalized[:-lag]
        denominator = torch.sqrt(torch.sum(left * left) * torch.sum(right * right))
        correlation = torch.sum(left * right) / torch.clamp(denominator, min=1e-12)
        correlations.append(correlation)
    correlation_tensor = torch.stack(correlations)
    selected = torch.topk(correlation_tensor, k=top_k).indices + 1
    delay_weights = torch.softmax(correlation_tensor[selected - 1], dim=0)

    def deterministic_matrix(offset):
        indices = torch.arange(dimension * dimension, dtype=dtype).reshape(dimension, dimension)
        return torch.sin(indices * 0.173 + offset) / math.sqrt(dimension)

    value_projection = torch.cos(torch.arange(dimension, dtype=dtype) * 0.47 + 0.2)
    trend_projection = torch.sin(torch.arange(dimension, dtype=dtype) * 0.31 + 0.6)
    position_projection = torch.cos(torch.arange(dimension, dtype=dtype) * 0.23 + 1.1)
    feedforward = [
        (deterministic_matrix(0.4), deterministic_matrix(1.7)),
        (deterministic_matrix(2.3), deterministic_matrix(3.6)),
    ]
    width = min(25, context - (1 - context % 2))
    width = max(3, width if width % 2 == 1 else width - 1)

    def moving_average(tensor):
        padding = width // 2
        padded = torch.cat(
            [tensor[:, :1, :].expand(-1, padding, -1), tensor,
             tensor[:, -1:, :].expand(-1, padding, -1)],
            dim=1,
        )
        cumulative = torch.cat(
            [torch.zeros((tensor.shape[0], 1, tensor.shape[2]), dtype=dtype),
             torch.cumsum(padded, dim=1)],
            dim=1,
        )
        return (cumulative[:, width:, :] - cumulative[:, :-width, :]) / width

    def layer_normalize(tensor):
        mean = tensor.mean(dim=-1, keepdim=True)
        variance = ((tensor - mean) ** 2).mean(dim=-1, keepdim=True)
        return (tensor - mean) / torch.sqrt(variance + 1e-8)

    positions = torch.linspace(-1.0, 0.0, context, dtype=dtype).reshape(1, context, 1)

    def encode(windows):
        scalar_trend = moving_average(windows.unsqueeze(-1)).squeeze(-1)
        scalar_seasonal = windows - scalar_trend
        encoded = (
            scalar_seasonal.unsqueeze(-1) * value_projection
            + scalar_trend.unsqueeze(-1) * trend_projection
            + positions * position_projection
        )
        accumulated_trend = torch.zeros_like(encoded)

        for first, second in feedforward:
            aggregated = encoded.clone()
            for delay, weight in zip(selected.tolist(), delay_weights):
                aggregated = aggregated + weight * torch.roll(encoded, shifts=delay, dims=1)
            block_trend = moving_average(aggregated)
            seasonal = aggregated - block_trend
            transformed = torch.tanh(seasonal @ first) @ second
            decomposed = seasonal + transformed
            residual_trend = moving_average(decomposed)
            encoded = layer_normalize(decomposed - residual_trend)
            accumulated_trend = accumulated_trend + block_trend + residual_trend

        return torch.cat(
            [encoded[:, -1, :],
             encoded[:, -period:, :].mean(dim=1),
             accumulated_trend[:, -1, :],
             windows[:, -1:].clone(),
             windows[:, -period:].mean(dim=1, keepdim=True),
             torch.ones((windows.shape[0], 1), dtype=dtype)],
            dim=1,
        )

    first_target = max(context, len(history) - 512)
    target_indices = list(range(first_target, len(history)))
    training_windows = torch.stack([normalized[index - context:index] for index in target_indices])
    design = encode(training_windows)
    targets = normalized[target_indices].unsqueeze(1)
    penalty = torch.eye(design.shape[1], dtype=dtype) * 1e-3
    penalty[-1, -1] = 0.0
    coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ targets)

    extended = normalized.tolist()
    forecasts = []
    for _ in range(horizon):
        window = torch.tensor(extended[-context:], dtype=dtype).unsqueeze(0)
        prediction = (encode(window) @ coefficients).squeeze()
        if not torch.isfinite(prediction).item():
            raise FloatingPointError("autoformer produced a non-finite prediction")
        extended.append(float(prediction))
        forecasts.append(float(prediction * scale + center))

    if not all(math.isfinite(value) for value in forecasts):
        raise FloatingPointError("autoformer produced non-finite forecasts")
    return forecasts


def fedformer(history, horizon, frequency):
    """Use when the series has stable trend and seasonality concentrated in a small number of transferable frequency modes."""
    import math
    import numpy as np

    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if not isinstance(frequency, str):
        raise NotApplicable(f"needs a supported frequency string, got {frequency}")

    parts = frequency.strip().lower().split()
    if len(parts) != 2 or not parts[0].isdigit():
        raise NotApplicable(f"needs a frequency like '1 hour' or '1 day', got {frequency}")

    step = int(parts[0])
    unit = parts[1].rstrip("s")
    base_periods = {"minute": 1440, "hour": 24, "day": 7, "week": 52, "month": 12}
    if step <= 0 or unit not in base_periods or base_periods[unit] % step != 0:
        raise NotApplicable(f"needs a supported regular frequency, got {frequency}")

    period = base_periods[unit] // step
    if period < 2:
        raise NotApplicable(f"needs a frequency with at least 2 samples per seasonal cycle, got {frequency}")
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if not all(math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    n = values.size

    left = period // 2
    right = period - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    trend = np.convolve(padded, np.ones(period, dtype=float) / period, mode="valid")
    seasonal = values - trend

    trend_width = min(n, 2 * period)
    trend_times = np.arange(trend_width, dtype=float)
    trend_tail = trend[-trend_width:]
    trend_design = np.column_stack((np.ones(trend_width), trend_times))
    trend_coefficients = np.linalg.lstsq(trend_design, trend_tail, rcond=None)[0]
    future_trend_times = np.arange(trend_width, trend_width + horizon, dtype=float)
    future_trend = trend_coefficients[0] + trend_coefficients[1] * future_trend_times

    spectrum = np.fft.rfft(seasonal)
    magnitudes = np.abs(spectrum)
    candidate_indices = np.arange(1, spectrum.size)
    mode_count = min(candidate_indices.size, max(1, int(np.sqrt(n))))
    selected = candidate_indices[np.argsort(magnitudes[1:])[-mode_count:]]

    fit_times = np.arange(n, dtype=float)
    future_times = np.arange(n, n + horizon, dtype=float)
    fit_columns = [np.ones(n, dtype=float)]
    future_columns = [np.ones(horizon, dtype=float)]
    for index in selected:
        angular_frequency = 2.0 * np.pi * index / n
        fit_columns.extend((np.cos(angular_frequency * fit_times), np.sin(angular_frequency * fit_times)))
        future_columns.extend((np.cos(angular_frequency * future_times), np.sin(angular_frequency * future_times)))

    fit_design = np.column_stack(fit_columns)
    future_design = np.column_stack(future_columns)
    seasonal_coefficients = np.linalg.lstsq(fit_design, seasonal, rcond=None)[0]
    forecast = future_trend + future_design @ seasonal_coefficients
    return [float(value) for value in forecast]


def patchtst(history, horizon, frequency):
    """Use when a sufficiently long series has stable local patterns that can be learned from repeated temporal patches."""
    import math
    import numpy as np
    import torch

    if len(history) < 48:
        raise NotApplicable(f"needs at least 48 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty frequency, got {frequency!r}")

    values = np.asarray(history, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon == 0:
        return []

    mean = float(values.mean())
    scale = float(values.std())
    if scale < 1e-8:
        scale = 1.0
    normalized = (values - mean) / scale

    patch_length = min(16, max(4, len(values) // 12))
    stride = max(1, patch_length // 2)
    patches = np.stack([
        normalized[start:start + patch_length]
        for start in range(0, len(normalized) - patch_length + 1, stride)
    ])
    context_length = min(8, len(patches) - 1)
    sample_count = len(patches) - context_length
    if sample_count < 4:
        raise NotApplicable(f"needs at least 4 patch training samples, got {sample_count}")

    inputs = np.stack([
        patches[i:i + context_length] for i in range(sample_count)
    ])
    targets = np.stack([
        patches[i + context_length] for i in range(sample_count)
    ])
    if len(inputs) > 256:
        inputs = inputs[-256:]
        targets = targets[-256:]

    dtype = torch.float64
    x_train = torch.tensor(inputs, dtype=dtype)
    y_train = torch.tensor(targets, dtype=dtype)
    model_width = 32
    head_count = 4
    head_width = model_width // head_count
    feedforward_width = 64

    def parameter(shape, offset, amplitude):
        count = int(np.prod(shape))
        indices = torch.arange(count, dtype=dtype)
        data = amplitude * torch.sin(indices * 0.173 + offset)
        return data.reshape(shape).clone().detach().requires_grad_(True)

    input_weight = parameter((patch_length, model_width), 0.1, 0.08)
    input_bias = parameter((model_width,), 0.3, 0.01)
    query_weight = parameter((model_width, model_width), 0.5, 0.08)
    key_weight = parameter((model_width, model_width), 0.7, 0.08)
    value_weight = parameter((model_width, model_width), 0.9, 0.08)
    attention_weight = parameter((model_width, model_width), 1.1, 0.08)
    ff1_weight = parameter((model_width, feedforward_width), 1.3, 0.06)
    ff1_bias = parameter((feedforward_width,), 1.5, 0.01)
    ff2_weight = parameter((feedforward_width, model_width), 1.7, 0.06)
    ff2_bias = parameter((model_width,), 1.9, 0.01)
    output_weight = parameter((model_width, patch_length), 2.1, 0.08)
    output_bias = parameter((patch_length,), 2.3, 0.01)

    parameters = [input_weight, input_bias, query_weight, key_weight,
                  value_weight, attention_weight, ff1_weight, ff1_bias,
                  ff2_weight, ff2_bias, output_weight, output_bias]

    positions = torch.arange(context_length, dtype=dtype).reshape(-1, 1)
    dimensions = torch.arange(model_width, dtype=dtype).reshape(1, -1)
    rates = torch.exp(-math.log(10000.0) * (2.0 * torch.floor(dimensions / 2.0)) / model_width)
    angles = positions * rates
    positional_encoding = torch.where(
        (dimensions.to(torch.int64) % 2) == 0,
        torch.sin(angles),
        torch.cos(angles),
    )

    def layer_norm(tensor):
        center = tensor.mean(dim=-1, keepdim=True)
        variance = tensor.var(dim=-1, keepdim=True, unbiased=False)
        return (tensor - center) / torch.sqrt(variance + 1e-6)

    def forward(patch_tensor):
        hidden = patch_tensor @ input_weight + input_bias + positional_encoding
        normalized_hidden = layer_norm(hidden)
        batch_size = normalized_hidden.shape[0]
        query = (normalized_hidden @ query_weight).reshape(batch_size, context_length, head_count, head_width).transpose(1, 2)
        key = (normalized_hidden @ key_weight).reshape(batch_size, context_length, head_count, head_width).transpose(1, 2)
        value = (normalized_hidden @ value_weight).reshape(batch_size, context_length, head_count, head_width).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_width)
        attention = torch.softmax(scores, dim=-1)
        attended = (attention @ value).transpose(1, 2).reshape(batch_size, context_length, model_width)
        hidden = hidden + attended @ attention_weight
        feedforward_input = layer_norm(hidden)
        feedforward = torch.nn.functional.gelu(feedforward_input @ ff1_weight + ff1_bias)
        hidden = hidden + feedforward @ ff2_weight + ff2_bias
        return layer_norm(hidden[:, -1, :]) @ output_weight + output_bias

    optimizer = torch.optim.Adam(parameters, lr=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        prediction = forward(x_train)
        loss = torch.mean((prediction - y_train) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    extended = normalized.tolist()
    with torch.no_grad():
        while len(extended) < len(normalized) + horizon:
            end = len(extended)
            starts = [end - patch_length - (context_length - 1 - i) * stride for i in range(context_length)]
            context = np.stack([
                np.asarray(extended[start:start + patch_length], dtype=np.float64)
                for start in starts
            ])
            next_patch = forward(torch.tensor(context[None, :, :], dtype=dtype))[0].cpu().numpy()
            extended.extend(next_patch[-stride:].tolist())

    result = np.asarray(extended[len(normalized):len(normalized) + horizon], dtype=np.float64) * scale + mean
    if not np.all(np.isfinite(result)):
        raise RuntimeError("PatchTST produced non-finite forecasts")
    return [float(value) for value in result]


def itransformer(history, horizon, frequency):
    """Use when stable relationships among lag-derived variates can reveal persistent temporal structure."""
    import numpy as np

    if len(history) < 24:
        raise NotApplicable(f"needs 24 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon == 0:
        return []

    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.0:
        return [0.0] * horizon
    scaled = values / magnitude
    lookback = min(48, max(8, len(values) // 4))
    dimension = min(12, lookback)

    times = np.arange(lookback, dtype=float)[:, None]
    modes = np.arange(dimension, dtype=float)[None, :]
    projection = np.cos(np.pi * (times + 0.5) * modes / lookback)
    projection[:, 0] /= np.sqrt(2.0)
    projection *= np.sqrt(2.0 / lookback)

    indices = np.arange(dimension, dtype=float)
    grid = indices[:, None] + indices[None, :]
    query_matrix = np.sin(grid + 1.0) / np.sqrt(dimension)
    key_matrix = np.cos(1.3 * grid + 0.5) / np.sqrt(dimension)
    value_matrix = np.sin(0.7 * grid + 0.8) / np.sqrt(dimension)
    output_matrix = np.cos(1.1 * grid + 0.3) / np.sqrt(dimension)
    channel_encoding = np.sin((np.arange(3, dtype=float)[:, None] + 1.0) * (indices[None, :] + 1.0)) / np.sqrt(dimension)

    def encode(window):
        center = float(np.mean(window))
        spread = float(np.sqrt(np.mean((window - center) ** 2)))
        spread = max(spread, 1e-8)
        raw = (window - center) / spread
        differences = np.diff(raw, prepend=raw[0])
        padded = np.pad(raw, (1, 1), mode="edge")
        trend = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
        tokens = np.stack((raw, differences, trend))
        embedded = tokens @ projection + channel_encoding
        embedded = (embedded - embedded.mean(axis=1, keepdims=True)) / np.sqrt(embedded.var(axis=1, keepdims=True) + 1e-6)
        queries = embedded @ query_matrix
        keys = embedded @ key_matrix
        scores = queries @ keys.T / np.sqrt(dimension)
        scores -= scores.max(axis=1, keepdims=True)
        attention = np.exp(scores)
        attention /= attention.sum(axis=1, keepdims=True)
        attended = attention @ (embedded @ value_matrix)
        transformed = attended @ output_matrix + embedded
        transformed = (transformed - transformed.mean(axis=1, keepdims=True)) / np.sqrt(transformed.var(axis=1, keepdims=True) + 1e-6)
        feature = np.concatenate((transformed.ravel(), raw[-4:]))
        return feature, center, spread

    features = []
    targets = []
    for endpoint in range(lookback, len(scaled)):
        feature, center, spread = encode(scaled[endpoint - lookback:endpoint])
        features.append(feature)
        targets.append((scaled[endpoint] - center) / spread)

    design = np.asarray(features, dtype=float)
    design = np.column_stack((np.ones(len(design)), design))
    target = np.asarray(targets, dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * 1e-3
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)

    extended = scaled.tolist()
    forecasts = []
    finite_limit = np.finfo(float).max / magnitude
    for _ in range(horizon):
        feature, center, spread = encode(np.asarray(extended[-lookback:], dtype=float))
        normalized_prediction = float(np.dot(np.concatenate(([1.0], feature)), coefficients))
        normalized_prediction = float(np.clip(normalized_prediction, -8.0, 8.0))
        prediction = float(np.clip(center + spread * normalized_prediction, -finite_limit, finite_limit))
        extended.append(prediction)
        forecasts.append(float(prediction * magnitude))
    return forecasts


def timesnet(history, horizon, frequency):
    """Use when a series has a few stable, discoverable periods organizing both within-period and across-period variation."""
    import numpy as np

    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise NotApplicable(f"needs a non-negative integer horizon, got {horizon!r}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty sampling frequency, got {frequency!r}")
    if len(history) < 16:
        raise NotApplicable(f"needs 16 points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise NotApplicable("needs finite history, got non-finite values")

    n = len(values)
    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.0:
        magnitude = 1.0
    scaled = values / magnitude

    time = np.arange(n, dtype=float)
    centered_time = time - time.mean()
    slope = np.dot(centered_time, scaled - scaled.mean()) / np.dot(centered_time, centered_time)
    detrended = scaled - (scaled.mean() + slope * centered_time)
    spectrum = np.abs(np.fft.rfft(detrended))
    order = np.argsort(spectrum[1:], kind="stable")[::-1] + 1

    periods = []
    strengths = []
    for index in order:
        period = int(round(n / float(index)))
        if 2 <= period <= n // 2 and period not in periods:
            periods.append(period)
            strengths.append(float(spectrum[index]))
        if len(periods) == 3:
            break

    if len(periods) < 2:
        raise NotApplicable(f"needs at least 2 discoverable periods, got {len(periods)}")

    strengths = np.asarray(strengths, dtype=float)
    weights = np.exp(strengths - strengths.max())
    weights /= weights.sum()
    lag = min(n // 2, max(8, 2 * max(periods)))

    def inception_features(window):
        result = [window[-1], window.mean(), window.std()]
        if len(window) > 1:
            result.append(window[-1] - window[-2])
        else:
            result.append(0.0)

        for period, weight in zip(periods, weights):
            padding = (-len(window)) % period
            aligned = np.pad(window, (padding, 0), mode="edge")
            tensor = aligned.reshape(-1, period)
            for kernel_size in (1, 3, 5):
                radius = kernel_size // 2
                padded = np.pad(tensor, ((radius, radius), (radius, radius)), mode="edge")
                windows = np.lib.stride_tricks.sliding_window_view(
                    padded, (kernel_size, kernel_size)
                )
                branch = windows.mean(axis=(-2, -1))
                result.extend(weight * np.asarray([
                    branch[-1, -1],
                    branch[-1].mean(),
                    branch[-1].std(),
                    branch[:, -1].mean(),
                    branch[:, -1].std(),
                    branch.mean(),
                    branch.std(),
                ]))
        return np.asarray(result, dtype=float)

    design = np.vstack([
        inception_features(scaled[index - lag:index])
        for index in range(lag, n)
    ])
    targets = scaled[lag:]
    feature_mean = design.mean(axis=0)
    feature_scale = design.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    standardized = (design - feature_mean) / feature_scale
    target_mean = targets.mean()
    centered_targets = targets - target_mean
    ridge = 1e-3
    dual = np.linalg.solve(
        standardized @ standardized.T + ridge * np.eye(len(standardized)),
        centered_targets,
    )

    extended = scaled.tolist()
    forecast = []
    for _ in range(horizon):
        features = inception_features(np.asarray(extended[-lag:], dtype=float))
        standardized_features = (features - feature_mean) / feature_scale
        prediction = target_mean + standardized_features @ standardized.T @ dual
        prediction = float(np.tanh(prediction))
        value = float(prediction * magnitude)
        if not np.isfinite(value):
            raise FloatingPointError("TimesNet produced a non-finite forecast")
        extended.append(prediction)
        forecast.append(value)

    return forecast


def tsmixer(history, horizon, frequency):
    """Use when an evenly spaced series has enough data for dense time-and-feature mixing to learn its multistep dynamics."""
    import numpy as np
    import torch

    if horizon < 1:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    if len(history) < horizon + 16:
        raise NotApplicable(f"needs at least {horizon + 16} points, got {len(history)}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty frequency string, got {frequency!r}")
    if not all(np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    values = torch.tensor(history, dtype=torch.float64)
    lookback = min(24, (len(history) - horizon) // 2)
    sample_count = len(history) - lookback - horizon + 1
    x = torch.stack([values[i:i + lookback] for i in range(sample_count)])
    y = torch.stack([values[i + lookback:i + lookback + horizon] for i in range(sample_count)])

    center = x.mean()
    scale = x.std(unbiased=False)
    if scale.item() == 0.0:
        scale = torch.tensor(1.0, dtype=torch.float64)
    x = (x - center) / scale
    y = (y - center) / scale

    features = 8
    hidden_features = 16
    hidden_time = 2 * lookback

    def parameter(shape, phase):
        size = int(np.prod(shape))
        indices = torch.arange(size, dtype=torch.float64)
        data = 0.05 * torch.sin(indices + phase) / np.sqrt(max(1, shape[0]))
        return torch.nn.Parameter(data.reshape(shape))

    class Mixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_weight = parameter((features,), 0.5)
            self.input_bias = parameter((features,), 1.0)
            self.time_1 = torch.nn.ParameterList()
            self.time_2 = torch.nn.ParameterList()
            self.feature_1 = torch.nn.ParameterList()
            self.feature_2 = torch.nn.ParameterList()
            for block in range(2):
                self.time_1.append(parameter((lookback, hidden_time), 2.0 + block))
                self.time_2.append(parameter((hidden_time, lookback), 4.0 + block))
                self.feature_1.append(parameter((features, hidden_features), 6.0 + block))
                self.feature_2.append(parameter((hidden_features, features), 8.0 + block))
            self.output_weight = parameter((lookback * features, horizon), 10.0)
            self.output_bias = torch.nn.Parameter(torch.zeros(horizon, dtype=torch.float64))

        def normalize(self, tensor):
            mean = tensor.mean(dim=-1, keepdim=True)
            variance = ((tensor - mean) ** 2).mean(dim=-1, keepdim=True)
            return (tensor - mean) / torch.sqrt(variance + 1e-8)

        def forward(self, tensor):
            mixed = tensor.unsqueeze(-1) * self.input_weight + self.input_bias
            for block in range(2):
                time_input = self.normalize(mixed).transpose(1, 2)
                time_update = torch.tanh(time_input @ self.time_1[block]) @ self.time_2[block]
                mixed = mixed + time_update.transpose(1, 2)
                feature_input = self.normalize(mixed)
                feature_update = torch.tanh(feature_input @ self.feature_1[block]) @ self.feature_2[block]
                mixed = mixed + feature_update
            return mixed.reshape(mixed.shape[0], -1) @ self.output_weight + self.output_bias

    model = Mixer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(300):
        optimizer.zero_grad()
        prediction = model(x)
        loss = torch.mean((prediction - y) ** 2)
        loss.backward()
        optimizer.step()

    final_input = ((values[-lookback:] - center) / scale).unsqueeze(0)
    with torch.no_grad():
        forecast = model(final_input).squeeze(0) * scale + center
    result = forecast.cpu().numpy().astype(float)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("TSMixer produced non-finite forecasts")
    return result.tolist()


def tide(history, horizon, frequency):
    """Use when dense nonlinear transformations of history and reliable known future time covariates can support a direct long-horizon forecast."""
    import math
    import torch

    if horizon <= 0:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    needed = 2 * horizon + 16
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")
    if not all(math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    key = frequency.strip().lower()
    cycles_by_frequency = {
        "1 minute": (60.0, 1440.0, 10080.0),
        "5 minutes": (12.0, 288.0, 2016.0),
        "15 minutes": (4.0, 96.0, 672.0),
        "30 minutes": (2.0, 48.0, 336.0),
        "1 hour": (24.0, 168.0, 8766.0),
        "1 day": (7.0, 365.2425),
        "1 week": (52.1775,),
        "1 month": (12.0,),
        "1 quarter": (4.0,),
        "1 year": (),
    }
    if key not in cycles_by_frequency:
        raise NotApplicable(f"needs a supported frequency, got {frequency}")

    torch.set_default_dtype(torch.float64)
    values = torch.tensor([float(value) for value in history], dtype=torch.float64)
    n = len(history)
    center = values.mean()
    scale = values.std(unbiased=False)
    if float(scale) <= 1e-12:
        scale = torch.tensor(1.0, dtype=torch.float64)
    normalized = (values - center) / scale

    lookback = min(64, max(16, n // 3))
    last_start = n - lookback - horizon
    starts = list(range(last_start + 1))
    if len(starts) > 256:
        stride = max(1, len(starts) // 256)
        starts = starts[-256 * stride::stride]

    cycles = cycles_by_frequency[key]
    covariate_width = 2 + 2 * len(cycles)

    def time_covariates(indices):
        index_tensor = torch.tensor(indices, dtype=torch.float64)
        denominator = float(max(1, n + horizon - 1))
        columns = [index_tensor / denominator, (index_tensor / denominator) ** 2]
        for cycle in cycles:
            angle = index_tensor * (2.0 * math.pi / cycle)
            columns.extend((torch.sin(angle), torch.cos(angle)))
        return torch.stack(columns, dim=1)

    all_covariates = time_covariates(range(n + horizon))
    past_targets = torch.stack([normalized[start:start + lookback] for start in starts])
    past_covariates = torch.stack([all_covariates[start:start + lookback] for start in starts])
    future_covariates = torch.stack([
        all_covariates[start + lookback:start + lookback + horizon] for start in starts
    ])
    targets = torch.stack([
        normalized[start + lookback:start + lookback + horizon] for start in starts
    ])

    class ResidualBlock(torch.nn.Module):
        def __init__(self, input_width, output_width, hidden_width):
            super().__init__()
            self.dense1 = torch.nn.Linear(input_width, hidden_width)
            self.dense2 = torch.nn.Linear(hidden_width, output_width)
            self.skip = torch.nn.Linear(input_width, output_width) if input_width != output_width else None

        def forward(self, tensor):
            residual = tensor if self.skip is None else self.skip(tensor)
            return residual + self.dense2(torch.nn.functional.relu(self.dense1(tensor)))

    class TiDE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            projected_width = 4
            hidden_width = 64
            self.covariate_projection = ResidualBlock(covariate_width, projected_width, 16)
            encoder_width = lookback * (1 + projected_width) + horizon * projected_width
            self.encoder1 = ResidualBlock(encoder_width, hidden_width, hidden_width)
            self.encoder2 = ResidualBlock(hidden_width, hidden_width, hidden_width)
            self.decoder = ResidualBlock(hidden_width, horizon * 8, hidden_width)
            self.temporal_decoder = ResidualBlock(8 + projected_width, 1, 16)
            self.linear_residual = torch.nn.Linear(lookback, horizon)
            self.initialize_deterministically()

        def initialize_deterministically(self):
            offset = 1
            with torch.no_grad():
                for module in self.modules():
                    if isinstance(module, torch.nn.Linear):
                        count = module.weight.numel()
                        positions = torch.arange(offset, offset + count, dtype=torch.float64)
                        weights = torch.sin(positions * 0.6180339887498949)
                        weights = weights.reshape_as(module.weight) / math.sqrt(max(1, module.in_features))
                        module.weight.copy_(weights)
                        module.bias.zero_()
                        offset += count

        def forward(self, past_y, past_x, future_x):
            projected_past = self.covariate_projection(past_x)
            projected_future = self.covariate_projection(future_x)
            encoded_input = torch.cat([
                past_y,
                projected_past.flatten(start_dim=1),
                projected_future.flatten(start_dim=1),
            ], dim=1)
            hidden = self.encoder2(self.encoder1(encoded_input))
            decoded = self.decoder(hidden).reshape(-1, horizon, 8)
            temporal_input = torch.cat([decoded, projected_future], dim=2)
            nonlinear = self.temporal_decoder(temporal_input).squeeze(-1)
            return nonlinear + self.linear_residual(past_y)

    model = TiDE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        prediction = model(past_targets, past_covariates, future_covariates)
        loss = torch.mean((prediction - targets) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        forecast = model(
            normalized[-lookback:].reshape(1, lookback),
            all_covariates[n - lookback:n].reshape(1, lookback, covariate_width),
            all_covariates[n:n + horizon].reshape(1, horizon, covariate_width),
        ).reshape(horizon)
        forecast = forecast * scale + center

    result = [float(value) for value in forecast]
    if len(result) != horizon or not all(math.isfinite(value) for value in result):
        raise RuntimeError("TiDE produced a non-finite or incorrectly sized forecast")
    return result


def scinet(history, horizon, frequency):
    """Use when an evenly sampled series contains complementary patterns visible at multiple interleaved resolutions."""
    import math
    import torch
    import torch.nn.functional as F

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise NotApplicable(f"needs a non-negative integer horizon, got {horizon}")
    if len(history) < 16:
        raise NotApplicable(f"needs at least 16 points, got {len(history)}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")

    parts = frequency.strip().lower().split()
    valid_units = {"second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks"}
    numeric_frequency = len(parts) == 2 and parts[0].replace(".", "", 1).isdigit()
    if not numeric_frequency or parts[1] not in valid_units or float(parts[0]) <= 0.0:
        raise NotApplicable(f"needs a positive fixed frequency from seconds through weeks, got {frequency}")
    if horizon == 0:
        return []

    dtype = torch.float64
    values = torch.tensor([float(value) for value in history], dtype=dtype)
    center = values.mean()
    scale = values.std(unbiased=False)
    if scale.item() == 0.0:
        scale = torch.tensor(1.0, dtype=dtype)
    normalized = (values - center) / scale

    lookback = 1
    maximum_lookback = min(32, len(history) // 2)
    while lookback * 2 <= maximum_lookback:
        lookback *= 2
    levels = min(3, int(math.log2(lookback)))

    class SCINetModel(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.depth = depth
            self.kernels = torch.nn.Parameter(torch.zeros(depth, 4, 3, dtype=dtype))
            initial_head = torch.zeros(width, dtype=dtype)
            initial_head[-1] = 1.0
            self.head = torch.nn.Parameter(initial_head)
            self.bias = torch.nn.Parameter(torch.zeros((), dtype=dtype))

        def convolve(self, sequence, kernel):
            padded = F.pad(sequence.unsqueeze(1), (1, 1), mode="replicate")
            return F.conv1d(padded, kernel.reshape(1, 1, 3)).squeeze(1)

        def interact(self, sequence, depth):
            if depth == self.depth or sequence.shape[1] < 2:
                return sequence
            even = sequence[:, 0::2]
            odd = sequence[:, 1::2]
            phi, psi, rho, tau = self.kernels[depth]
            even_scaled = even * torch.exp(torch.clamp(self.convolve(odd, phi), -5.0, 5.0))
            odd_scaled = odd * torch.exp(torch.clamp(self.convolve(even, psi), -5.0, 5.0))
            even_updated = even_scaled + self.convolve(odd_scaled, rho)
            odd_updated = odd_scaled - self.convolve(even_scaled, tau)
            even_updated = self.interact(even_updated, depth + 1)
            odd_updated = self.interact(odd_updated, depth + 1)
            merged = torch.empty_like(sequence)
            merged[:, 0::2] = even_updated
            merged[:, 1::2] = odd_updated
            return merged

        def forward(self, sequence):
            encoded = self.interact(sequence, 0)
            return encoded.matmul(self.head) + self.bias

    windows = normalized.unfold(0, lookback + 1, 1)
    predictors = windows[:, :-1].contiguous()
    targets = windows[:, -1].contiguous()
    model = SCINetModel(lookback, levels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(250):
        optimizer.zero_grad()
        predictions = model(predictors)
        loss = F.mse_loss(predictions, targets)
        if not math.isfinite(loss.item()):
            raise FloatingPointError("SCINet optimization produced a non-finite loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.kernels.clamp_(-1.0, 1.0)
            model.head.clamp_(-5.0, 5.0)
            model.bias.clamp_(-5.0, 5.0)

    generated = normalized.tolist()
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            window = torch.tensor(generated[-lookback:], dtype=dtype).unsqueeze(0)
            next_value = model(window).item()
            if not math.isfinite(next_value):
                raise FloatingPointError("SCINet produced a non-finite forecast")
            generated.append(next_value)

    result = [float(value * scale.item() + center.item()) for value in generated[-horizon:]]
    if any(not math.isfinite(value) for value in result):
        raise FloatingPointError("SCINet produced a non-finite forecast")
    return result


def timemixer(history, horizon, frequency):
    """Use when fine and coarse sampling scales contain complementary stable seasonal and trend patterns."""
    import numpy as np

    if len(history) < 32:
        raise NotApplicable(f"needs at least 32 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("history must contain only finite values")
    if horizon == 0:
        return []

    trend_forecasts = []
    seasonal_forecasts = []
    trend_errors = []
    seasonal_errors = []

    for scale in (1, 2, 4):
        usable = (len(y) // scale) * scale
        values = y[-usable:].reshape(-1, scale).mean(axis=1)
        count = len(values)

        window = min(7, count if count % 2 == 1 else count - 1)
        window = max(3, window)
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        trend = np.convolve(padded, np.ones(window) / window, mode="valid")
        seasonal = values - trend

        centered = seasonal - seasonal.mean()
        maximum_period = min(24, count // 3)
        periods = np.arange(2, maximum_period + 1)
        correlations = np.array([
            np.dot(centered[:-period], centered[period:]) /
            max(np.dot(centered, centered), np.finfo(float).tiny)
            for period in periods
        ])
        period = int(periods[np.argmax(correlations)])

        t = np.arange(count, dtype=float)
        future_t = count - 1 + np.arange(1, horizon + 1, dtype=float) / scale
        t_center = t.mean()
        t_scale = max(t.std(), 1.0)
        normalized_t = (t - t_center) / t_scale
        normalized_future = (future_t - t_center) / t_scale

        trend_design = np.column_stack((np.ones(count), normalized_t))
        trend_coefficients = np.linalg.lstsq(trend_design, trend, rcond=None)[0]
        fitted_trend = trend_design @ trend_coefficients
        future_trend = np.column_stack((np.ones(horizon), normalized_future)) @ trend_coefficients

        angle = 2.0 * np.pi * t / period
        future_angle = 2.0 * np.pi * future_t / period
        seasonal_design = np.column_stack((
            np.ones(count), np.sin(angle), np.cos(angle)
        ))
        seasonal_coefficients = np.linalg.lstsq(seasonal_design, seasonal, rcond=None)[0]
        fitted_seasonal = seasonal_design @ seasonal_coefficients
        future_seasonal = np.column_stack((
            np.ones(horizon), np.sin(future_angle), np.cos(future_angle)
        )) @ seasonal_coefficients

        trend_forecasts.append(future_trend)
        seasonal_forecasts.append(future_seasonal)
        trend_errors.append(np.mean((trend - fitted_trend) ** 2))
        seasonal_errors.append(np.mean((seasonal - fitted_seasonal) ** 2))

    epsilon = np.finfo(float).eps * max(1.0, float(np.var(y)))
    trend_weights = 1.0 / (np.asarray(trend_errors) + epsilon)
    seasonal_weights = 1.0 / (np.asarray(seasonal_errors) + epsilon)
    trend_weights /= trend_weights.sum()
    seasonal_weights /= seasonal_weights.sum()

    forecast = (
        np.average(np.asarray(trend_forecasts), axis=0, weights=trend_weights) +
        np.average(np.asarray(seasonal_forecasts), axis=0, weights=seasonal_weights)
    )
    if not np.all(np.isfinite(forecast)):
        raise ValueError("timemixer produced non-finite forecasts")
    return [float(value) for value in forecast]


def samformer(history, horizon, frequency):
    """Use when nonlinear cross-channel lag relationships may benefit from a compact attention model trained for flat-minimum generalization."""
    import math
    import numpy as np
    import torch

    if len(history) < 24:
        raise NotApplicable(f"needs 24 points, got {len(history)}")
    if horizon < 1:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")

    values = np.asarray(history, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise NotApplicable("needs finite history values, got non-finite values")

    torch.set_default_dtype(torch.float64)
    series = torch.as_tensor(values, dtype=torch.float64)
    center = series.mean()
    scale = series.std(unbiased=False).clamp_min(1e-8)
    normalized = (series - center) / scale

    window = min(16, max(6, len(history) // 8))
    starts = max(0, len(history) - window - 512)
    x = torch.stack([normalized[i:i + window] for i in range(starts, len(history) - window)])
    y = torch.stack([normalized[i + window] for i in range(starts, len(history) - window)])

    width = 6
    hidden = 12

    def parameter(shape, scale_factor, phase):
        count = int(np.prod(shape))
        indices = torch.arange(count, dtype=torch.float64)
        data = torch.sin((indices + phase) * 0.371) * scale_factor
        return data.reshape(shape).clone().requires_grad_(True)

    input_weight = parameter((width,), 0.12, 1.0)
    position = parameter((window, width), 0.04, 2.0)
    wq = parameter((width, width), 0.10, 3.0)
    wk = parameter((width, width), 0.10, 4.0)
    wv = parameter((width, width), 0.10, 5.0)
    wo = parameter((width, width), 0.10, 6.0)
    w1 = parameter((width, hidden), 0.08, 7.0)
    b1 = parameter((hidden,), 0.01, 8.0)
    w2 = parameter((hidden, width), 0.08, 9.0)
    b2 = parameter((width,), 0.01, 10.0)
    head = parameter((width,), 0.08, 11.0)
    bias = parameter((1,), 0.0, 12.0)
    parameters = [input_weight, position, wq, wk, wv, wo, w1, b1, w2, b2, head, bias]

    def layer_norm(tensor):
        mean = tensor.mean(dim=-1, keepdim=True)
        variance = ((tensor - mean) ** 2).mean(dim=-1, keepdim=True)
        return (tensor - mean) / torch.sqrt(variance + 1e-6)

    def model(batch):
        tokens = batch.unsqueeze(-1) * input_weight + position
        tokens = layer_norm(tokens)
        queries = tokens @ wq
        keys = tokens @ wk
        values_tensor = tokens @ wv
        scores = (queries @ keys.transpose(-1, -2)) / math.sqrt(width)
        attended = torch.softmax(scores, dim=-1) @ values_tensor
        state = layer_norm(tokens + attended @ wo)
        feed_forward = torch.nn.functional.gelu(state @ w1 + b1) @ w2 + b2
        state = layer_norm(state + feed_forward)
        return state.mean(dim=1) @ head + bias[0]

    def objective(batch, targets):
        errors = model(batch) - targets
        regularization = sum((item * item).mean() for item in parameters)
        return (errors * errors).mean() + 1e-5 * regularization

    learning_rate = 0.008
    rho = 0.025
    for _ in range(70):
        for item in parameters:
            item.grad = None
        first_loss = objective(x, y)
        first_loss.backward()
        gradient_norm = torch.sqrt(sum((item.grad * item.grad).sum() for item in parameters))
        perturbations = []
        with torch.no_grad():
            multiplier = rho / (gradient_norm + 1e-12)
            for item in parameters:
                perturbation = item.grad * multiplier
                item.add_(perturbation)
                perturbations.append(perturbation)
        for item in parameters:
            item.grad = None
        second_loss = objective(x, y)
        second_loss.backward()
        with torch.no_grad():
            for item, perturbation in zip(parameters, perturbations):
                item.sub_(perturbation)
                item.sub_(learning_rate * item.grad.clamp(-2.0, 2.0))
                item.clamp_(-5.0, 5.0)

    generated = normalized.tolist()
    forecasts = []
    with torch.no_grad():
        for _ in range(horizon):
            batch = torch.tensor(generated[-window:], dtype=torch.float64).unsqueeze(0)
            prediction = model(batch)[0].clamp(-12.0, 12.0)
            value = prediction * scale + center
            if not torch.isfinite(value):
                raise RuntimeError("SAMformer produced a non-finite forecast")
            forecasts.append(float(value.item()))
            generated.append(float(prediction.item()))
    return forecasts


def auto_ces(history, horizon, frequency):
    """Use when a sufficiently long series has stable oscillatory level or trend dynamics."""
    import numpy as np
    from scipy.optimize import least_squares

    if len(history) < 16:
        raise NotApplicable(f"needs 16 points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise NotApplicable(f"needs finite history, got {len(history)} points with non-finite values")

    n = len(y)
    center = float(np.mean(y))
    scale = max(float(np.ptp(y)), float(np.std(y)), abs(center) * 0.01, 1e-8)
    minimum = float(np.min(y))
    maximum = float(np.max(y))
    omega_low = 2.0 * np.pi / n
    omega_high = np.pi - 1e-6

    spectrum = np.abs(np.fft.rfft(y - center))
    spectrum[0] = 0.0
    peak_index = int(np.argmax(spectrum))
    peak_omega = 2.0 * np.pi * max(1, peak_index) / n
    peak_omega = min(max(peak_omega, omega_low), omega_high)
    omega_starts = sorted(set((peak_omega, np.pi / 6.0, np.pi / 3.0, 2.0 * np.pi / 3.0)))

    best_ic = np.inf
    best_specification = None
    best_parameters = None

    for use_trend in (False, True):
        for adaptive_level in (False, True):
            for damped in (False, True):
                lower = [0.001]
                upper = [0.999]
                if adaptive_level:
                    lower.append(0.001)
                    upper.append(0.5)
                if use_trend:
                    lower.append(0.001)
                    upper.append(0.5)
                lower.append(omega_low)
                upper.append(omega_high)
                if damped:
                    lower.append(0.5)
                    upper.append(0.9999)
                lower.extend([minimum - 4.0 * scale, -4.0 * scale, -4.0 * scale])
                upper.extend([maximum + 4.0 * scale, 4.0 * scale, 4.0 * scale])
                if use_trend:
                    lower.extend([-2.0 * scale, -2.0 * scale])
                    upper.extend([2.0 * scale, 2.0 * scale])

                lower = np.asarray(lower, dtype=float)
                upper = np.asarray(upper, dtype=float)

                def unpack(parameters):
                    index = 0
                    alpha = parameters[index]
                    index += 1
                    beta = parameters[index] if adaptive_level else 0.0
                    index += int(adaptive_level)
                    gamma = parameters[index] if use_trend else 0.0
                    index += int(use_trend)
                    omega = parameters[index]
                    index += 1
                    rho = parameters[index] if damped else 1.0
                    index += int(damped)
                    mean_state = parameters[index]
                    real_state = parameters[index + 1]
                    imaginary_state = parameters[index + 2]
                    index += 3
                    if use_trend:
                        real_trend = parameters[index]
                        imaginary_trend = parameters[index + 1]
                    else:
                        real_trend = 0.0
                        imaginary_trend = 0.0
                    return alpha, beta, gamma, omega, rho, mean_state, complex(real_state, imaginary_state), complex(real_trend, imaginary_trend)

                def residuals(parameters):
                    alpha, beta, gamma, omega, rho, mean_state, state, trend = unpack(parameters)
                    rotation = rho * complex(np.cos(omega), np.sin(omega))
                    errors = np.empty(n, dtype=float)
                    for time in range(n):
                        predicted_state = rotation * (state + trend)
                        error = y[time] - (mean_state + predicted_state.real)
                        errors[time] = error
                        mean_state += beta * error
                        state = predicted_state + alpha * error
                        if use_trend:
                            trend = rotation * trend + gamma * alpha * error
                    return errors

                candidate_sse = np.inf
                candidate_parameters = None
                for omega_start in omega_starts:
                    start = [0.2]
                    if adaptive_level:
                        start.append(0.05)
                    if use_trend:
                        start.append(0.05)
                    start.append(min(max(omega_start, omega_low), omega_high))
                    if damped:
                        start.append(0.95)
                    start.extend([center, float(y[0] - center), 0.0])
                    if use_trend:
                        start.extend([0.0, 0.0])
                    result = least_squares(
                        residuals,
                        np.asarray(start, dtype=float),
                        bounds=(lower, upper),
                        method="trf",
                        max_nfev=5000,
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-10,
                    )
                    if not result.success:
                        raise RuntimeError(f"complex exponential smoothing optimization failed: {result.message}")
                    sse = float(np.dot(result.fun, result.fun))
                    if sse < candidate_sse:
                        candidate_sse = sse
                        candidate_parameters = result.x.copy()

                parameter_count = len(candidate_parameters)
                variance = max(candidate_sse / n, np.finfo(float).tiny)
                aic = n * np.log(variance) + 2.0 * parameter_count
                aicc = aic + (2.0 * parameter_count * (parameter_count + 1.0)) / (n - parameter_count - 1.0)
                if aicc < best_ic:
                    best_ic = aicc
                    best_specification = (use_trend, adaptive_level, damped)
                    best_parameters = candidate_parameters

    use_trend, adaptive_level, damped = best_specification
    index = 0
    alpha = best_parameters[index]
    index += 1
    beta = best_parameters[index] if adaptive_level else 0.0
    index += int(adaptive_level)
    gamma = best_parameters[index] if use_trend else 0.0
    index += int(use_trend)
    omega = best_parameters[index]
    index += 1
    rho = best_parameters[index] if damped else 1.0
    index += int(damped)
    mean_state = best_parameters[index]
    state = complex(best_parameters[index + 1], best_parameters[index + 2])
    index += 3
    trend = complex(best_parameters[index], best_parameters[index + 1]) if use_trend else 0.0j
    rotation = rho * complex(np.cos(omega), np.sin(omega))

    for observation in y:
        predicted_state = rotation * (state + trend)
        error = observation - (mean_state + predicted_state.real)
        mean_state += beta * error
        state = predicted_state + alpha * error
        if use_trend:
            trend = rotation * trend + gamma * alpha * error

    forecast = []
    for _ in range(horizon):
        state = rotation * (state + trend)
        if use_trend:
            trend = rotation * trend
        forecast.append(float(mean_state + state.real))

    if not np.all(np.isfinite(forecast)):
        raise RuntimeError("complex exponential smoothing produced non-finite forecasts")
    return forecast


def auto_mfles(history, horizon, frequency):
    """Use when an additive series has a stable trend and one or more recurring seasonal patterns."""
    import numpy as np
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    if not isinstance(frequency, str):
        raise NotApplicable(f"needs a supported frequency string, got {frequency!r}")
    text = frequency.strip().lower().replace("_", " ").replace("-", " ")
    parts = text.split()
    aliases = {"s": "second", "sec": "second", "secs": "second", "seconds": "second", "min": "minute", "mins": "minute", "minutes": "minute", "h": "hour", "hr": "hour", "hrs": "hour", "hours": "hour", "d": "day", "days": "day", "w": "week", "weeks": "week", "mo": "month", "mos": "month", "months": "month", "q": "quarter", "quarters": "quarter"}
    if len(parts) == 1:
        amount = 1.0
        unit = aliases.get(parts[0], parts[0])
    elif len(parts) == 2:
        try:
            amount = float(parts[0])
        except ValueError:
            amount = -1.0
        unit = aliases.get(parts[1], parts[1].rstrip("s"))
    else:
        amount = -1.0
        unit = ""
    supported = {"second", "minute", "hour", "day", "week", "month", "quarter"}
    if amount <= 0.0 or unit not in supported:
        raise NotApplicable(f"needs a supported positive frequency, got {frequency!r}")
    cycles = {
        "second": [60.0, 3600.0, 86400.0, 604800.0],
        "minute": [60.0, 1440.0, 10080.0],
        "hour": [24.0, 168.0, 8765.82],
        "day": [7.0, 365.2425],
        "week": [52.1775],
        "month": [12.0],
        "quarter": [4.0]
    }
    candidate_periods = [cycle / amount for cycle in cycles[unit] if cycle / amount >= 2.0]
    if not candidate_periods:
        raise NotApplicable(f"needs a frequency with a seasonal period of at least 2 points, got {frequency!r}")
    needed = int(np.ceil(2.0 * candidate_periods[0]))
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")
    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise NotApplicable(f"needs finite history values, got {len(history)} points containing non-finite values")
    n = y.size
    periods = [period for period in candidate_periods if n >= int(np.ceil(2.0 * period))]
    future_index = np.arange(n, n + horizon, dtype=float)
    observed_index = np.arange(n, dtype=float)
    center = (n - 1.0) / 2.0
    scale = max(float(n - 1), 1.0)
    trend_x = np.column_stack((np.ones(n), (observed_index - center) / scale))
    trend_future_x = np.column_stack((np.ones(horizon), (future_index - center) / scale))
    fitted = np.full(n, float(np.median(y)))
    forecast = np.full(horizon, float(np.median(y)))
    shrinkage = 0.5
    ridge = 1.0e-6
    for _ in range(4):
        residual = y - fitted
        trend_penalty = np.diag([ridge, ridge])
        trend_coef = np.linalg.solve(trend_x.T @ trend_x + trend_penalty, trend_x.T @ residual)
        fitted += shrinkage * (trend_x @ trend_coef)
        forecast += shrinkage * (trend_future_x @ trend_coef)
        for period in periods:
            harmonics = min(5, max(1, int(np.floor(period / 2.0))))
            observed_columns = []
            future_columns = []
            for harmonic in range(1, harmonics + 1):
                angle = 2.0 * np.pi * harmonic * observed_index / period
                future_angle = 2.0 * np.pi * harmonic * future_index / period
                observed_columns.extend((np.sin(angle), np.cos(angle)))
                future_columns.extend((np.sin(future_angle), np.cos(future_angle)))
            seasonal_x = np.column_stack(observed_columns)
            seasonal_future_x = np.column_stack(future_columns)
            residual = y - fitted
            penalty = ridge * np.eye(seasonal_x.shape[1])
            seasonal_coef = np.linalg.solve(seasonal_x.T @ seasonal_x + penalty, seasonal_x.T @ residual)
            fitted += shrinkage * (seasonal_x @ seasonal_coef)
            forecast += shrinkage * (seasonal_future_x @ seasonal_coef)
    residual = y - fitted
    alpha_grid = np.linspace(0.05, 0.95, 19)
    best_sse = np.inf
    best_level = float(residual[0])
    for alpha in alpha_grid:
        level = float(residual[0])
        sse = 0.0
        for value in residual[1:]:
            error = float(value) - level
            sse += error * error
            level = alpha * float(value) + (1.0 - alpha) * level
        if sse < best_sse:
            best_sse = sse
            best_level = level
    forecast += best_level
    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("MFLES produced non-finite forecasts")
    return [float(value) for value in forecast]


def dynamic_theta(history, horizon, frequency):
    """Use when local level and curvature evolve gradually and enough observations exist to estimate adaptive Theta components."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    y = np.asarray(history, dtype=float)
    n = len(y)
    if n < 8:
        raise NotApplicable(f"needs at least 8 points, got {n}")
    if not np.all(np.isfinite(y)):
        raise NotApplicable(f"needs finite observations, got {np.count_nonzero(~np.isfinite(y))} non-finite values")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return []

    _ = frequency
    window = min(n, max(6, int(round(3.0 * np.sqrt(n)))))
    trend = np.empty(n, dtype=float)
    for end in range(n):
        start = max(0, end - window + 1)
        segment = y[start:end + 1]
        if len(segment) == 1:
            trend[end] = segment[0]
        else:
            x = np.arange(len(segment), dtype=float)
            design = np.column_stack((np.ones(len(segment)), x))
            coefficients = np.linalg.lstsq(design, segment, rcond=None)[0]
            trend[end] = coefficients[0] + coefficients[1] * x[-1]

    differences = np.diff(y)
    scale = max(float(np.median(np.abs(differences))), np.finfo(float).eps)
    level_signal = np.zeros(n, dtype=float)
    curvature_signal = np.zeros(n, dtype=float)
    if n > 1:
        level_signal[1:] = np.minimum(np.abs(differences) / scale, 10.0)
    if n > 2:
        curvature_signal[2:] = np.minimum(np.abs(np.diff(y, n=2)) / scale, 10.0)

    def simulate(parameters, keep_state=False):
        alpha_base, alpha_response, theta_base, theta_response = parameters
        alpha = 0.02 + 0.96 * expit(alpha_base)
        theta = 1.0 + 7.0 * expit(theta_base)
        component = theta * y[0] + (1.0 - theta) * trend[0]
        squared_error = 0.0
        last_theta = theta
        for t in range(1, n):
            alpha = 0.02 + 0.96 * expit(
                alpha_base + alpha_response * level_signal[t - 1]
            )
            theta = 1.0 + 7.0 * expit(
                theta_base + theta_response * curvature_signal[t - 1]
            )
            prediction = (1.0 - 1.0 / theta) * trend[t] + component / theta
            error = y[t] - prediction
            squared_error += error * error
            theta_observation = theta * y[t] + (1.0 - theta) * trend[t]
            component = alpha * theta_observation + (1.0 - alpha) * component
            last_theta = theta
        if keep_state:
            return component, last_theta
        return squared_error / ((n - 1) * scale * scale)

    initial = np.array([-1.4, 0.25, -1.8, 0.25], dtype=float)
    bounds = [(-6.0, 6.0), (-3.0, 3.0), (-6.0, 6.0), (-3.0, 3.0)]
    fitted = minimize(simulate, initial, method="L-BFGS-B", bounds=bounds)
    if not fitted.success:
        raise RuntimeError(f"dynamic Theta optimization failed: {fitted.message}")

    component, last_theta = simulate(fitted.x, keep_state=True)
    last_trend = trend[-1]
    smoothed_level = (1.0 - 1.0 / last_theta) * last_trend + component / last_theta

    final_segment = y[-window:]
    x = np.arange(len(final_segment), dtype=float)
    design = np.column_stack((np.ones(len(final_segment)), x))
    intercept, slope = np.linalg.lstsq(design, final_segment, rcond=None)[0]

    forecasts = []
    for step in range(1, horizon + 1):
        decay = 0.8 ** step
        future_theta = 1.0 + 7.0 * expit(
            fitted.x[2] + fitted.x[3] * curvature_signal[-1] * decay
        )
        future_trend = intercept + slope * (x[-1] + step)
        value = smoothed_level + (1.0 - 1.0 / future_theta) * (
            future_trend - last_trend
        )
        forecasts.append(float(value))

    if not np.all(np.isfinite(forecasts)):
        raise FloatingPointError("dynamic Theta produced non-finite forecasts")
    return forecasts


def croston_optimized(history, horizon, frequency):
    """Use when nonnegative demand is intermittent and future arrival gaps and positive demand sizes are expected to resemble history."""
    import math
    import numpy as np
    from scipy.optimize import minimize

    if horizon < 0:
        raise NotApplicable(f"needs a nonnegative horizon, got {horizon}")
    if len(history) < 3:
        raise NotApplicable(f"needs at least 3 points, got {len(history)}")
    if any(not math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite observations, got non-finite values")
    if any(float(value) < 0.0 for value in history):
        raise NotApplicable("needs nonnegative observations, got negative values")
    positive_count = sum(float(value) > 0.0 for value in history)
    if positive_count < 2:
        raise NotApplicable(f"needs at least 2 positive observations, got {positive_count}")

    values = np.asarray(history, dtype=float)
    first = int(np.flatnonzero(values > 0.0)[0])

    def objective(parameters):
        size_alpha, interval_alpha = parameters
        size = values[first]
        interval = float(first + 1)
        last_positive = first
        squared_error = 0.0
        for index in range(first + 1, len(values)):
            forecast = size / interval
            error = values[index] - forecast
            squared_error += error * error
            if values[index] > 0.0:
                gap = float(index - last_positive)
                size += size_alpha * (values[index] - size)
                interval += interval_alpha * (gap - interval)
                last_positive = index
        return squared_error / float(len(values) - first - 1)

    result = minimize(
        objective,
        x0=np.array([0.1, 0.1], dtype=float),
        method="L-BFGS-B",
        bounds=((1e-6, 1.0), (1e-6, 1.0)),
    )
    if not result.success:
        raise RuntimeError(f"Croston parameter optimization failed: {result.message}")

    size_alpha, interval_alpha = result.x
    size = values[first]
    interval = float(first + 1)
    last_positive = first
    for index in range(first + 1, len(values)):
        if values[index] > 0.0:
            gap = float(index - last_positive)
            size += size_alpha * (values[index] - size)
            interval += interval_alpha * (gap - interval)
            last_positive = index

    forecast = float(size / interval)
    return [forecast for _ in range(horizon)]


def seasonal_window_average(history, horizon, frequency):
    """Use when seasonal phase is stable and the two most recent matching cycles have a steady mean."""
    import math

    periods = {
        "1 minute": 1440,
        "5 minutes": 288,
        "10 minutes": 144,
        "15 minutes": 96,
        "30 minutes": 48,
        "1 hour": 24,
        "2 hours": 12,
        "3 hours": 8,
        "4 hours": 6,
        "6 hours": 4,
        "8 hours": 3,
        "12 hours": 2,
        "1 day": 7,
        "1 week": 52,
        "1 month": 12,
        "1 quarter": 4
    }
    normalized_frequency = frequency.strip().lower()
    if normalized_frequency not in periods:
        raise NotApplicable(f"needs a supported seasonal frequency, got {frequency!r}")
    period = periods[normalized_frequency]
    if len(history) < 2 * period:
        raise NotApplicable(f"needs {2 * period} points, got {len(history)}")
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError(f"horizon must be a non-negative integer, got {horizon!r}")
    if any(not math.isfinite(value) for value in history):
        raise ValueError("history must contain only finite values")

    forecasts = []
    n = len(history)
    for step in range(horizon):
        phase = (n + step) % period
        latest = n - 1 - ((n - 1 - phase) % period)
        forecasts.append(float((history[latest] + history[latest - period]) / 2.0))
    return forecasts


def film_legendre_memory(history, horizon, frequency):
    """Use when a long, regularly sampled series is smooth and can be compressed into low-order trend and spectral components."""
    import numpy as np

    if len(history) < 16:
        raise NotApplicable(f"needs at least 16 points, got {len(history)}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a regular sampling frequency, got {frequency!r}")

    y = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("history must contain only finite values")
    if horizon == 0:
        return []

    n = y.size
    total = n + horizon
    observed_x = -1.0 + 2.0 * np.arange(n) / (total - 1)
    future_x = -1.0 + 2.0 * np.arange(n, total) / (total - 1)

    legendre_degree = min(12, max(3, int(np.sqrt(n))))
    observed_basis = np.polynomial.legendre.legvander(observed_x, legendre_degree)
    memory_coefficients = np.linalg.lstsq(observed_basis, y, rcond=None)[0]
    trend_fit = observed_basis @ memory_coefficients
    trend_forecast = np.polynomial.legendre.legvander(future_x, legendre_degree) @ memory_coefficients

    residual = y - trend_fit
    spectrum = np.fft.rfft(residual)
    candidate_bins = np.arange(1, spectrum.size)
    component_count = min(max(1, legendre_degree // 2), candidate_bins.size)
    selected_bins = candidate_bins[np.argsort(np.abs(spectrum[candidate_bins]))[-component_count:]]
    selected_bins = np.sort(selected_bins)

    observed_t = np.arange(n, dtype=float)
    future_t = np.arange(n, total, dtype=float)
    observed_columns = [np.ones(n)]
    future_columns = [np.ones(horizon)]
    for spectral_bin in selected_bins:
        angular_frequency = 2.0 * np.pi * spectral_bin / n
        observed_columns.extend([
            np.cos(angular_frequency * observed_t),
            np.sin(angular_frequency * observed_t),
        ])
        future_columns.extend([
            np.cos(angular_frequency * future_t),
            np.sin(angular_frequency * future_t),
        ])

    fourier_projection = np.column_stack(observed_columns)
    fourier_coefficients = np.linalg.lstsq(fourier_projection, residual, rcond=None)[0]
    spectral_forecast = np.column_stack(future_columns) @ fourier_coefficients
    forecast = trend_forecast + spectral_forecast

    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("FILM projection produced non-finite forecasts")
    return [float(value) for value in forecast]


def auto_tbats(history, horizon, frequency):
    """Use when a sufficiently long series has evolving trend and one or more possibly non-integer seasonal cycles."""
    import math
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import boxcox_normmax

    parts = frequency.strip().lower().split()
    if len(parts) != 2 or not parts[0].replace(".", "", 1).isdigit():
        raise NotApplicable(f"needs a supported numeric frequency, got {frequency}")
    quantity = float(parts[0])
    unit = parts[1][:-1] if parts[1].endswith("s") else parts[1]
    if quantity <= 0 or unit not in ("minute", "hour", "day", "week", "month", "quarter"):
        raise NotApplicable(f"needs a supported positive frequency, got {frequency}")

    if unit == "minute":
        periods = [1440.0 / quantity, 10080.0 / quantity]
    elif unit == "hour":
        periods = [24.0 / quantity, 168.0 / quantity]
    elif unit == "day":
        periods = [7.0 / quantity, 365.25 / quantity]
    elif unit == "week":
        periods = [52.1775 / quantity]
    elif unit == "month":
        periods = [12.0 / quantity]
    else:
        periods = [4.0 / quantity]
    periods = [p for p in periods if p >= 2.0]
    if not periods:
        raise NotApplicable(f"needs a frequency with a seasonal period of at least 2 observations, got {frequency}")

    needed = int(math.ceil(2.0 * max(periods)))
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")
    if not all(math.isfinite(float(value)) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if max(history) == min(history):
        raise NotApplicable("needs a varying history, got constant values")
    if horizon == 0:
        return []

    original = np.asarray(history, dtype=float)
    n = original.size
    harmonics = [(period, harmonic) for period in periods for harmonic in range(1, min(3, int(period // 2)) + 1)]
    angles = [2.0 * math.pi * harmonic / period for period, harmonic in harmonics]

    def transform(values, lam):
        if lam is None:
            return values.copy()
        if abs(lam) < 1e-8:
            return np.log(values)
        return (np.power(values, lam) - 1.0) / lam

    def initial_regression(values, trend_mode):
        time = np.arange(n, dtype=float)
        columns = [np.ones(n, dtype=float)]
        if trend_mode:
            columns.append(time)
        for angle in angles:
            columns.append(np.cos(angle * time))
            columns.append(np.sin(angle * time))
        design = np.column_stack(columns)
        return np.linalg.lstsq(design, values, rcond=None)[0]

    def run(parameters, values, trend_mode, p, q, coefficients, forecast_steps):
        position = 0
        alpha = 1.0 / (1.0 + math.exp(-float(np.clip(parameters[position], -35.0, 35.0))))
        position += 1
        if trend_mode:
            beta = 1.0 / (1.0 + math.exp(-float(np.clip(parameters[position], -35.0, 35.0))))
            position += 1
        else:
            beta = 0.0
        gammas = []
        for _ in angles:
            gamma_one = 0.25 * math.tanh(float(parameters[position]))
            gamma_two = 0.25 * math.tanh(float(parameters[position + 1]))
            position += 2
            gammas.append((gamma_one, gamma_two))
        if trend_mode == 2:
            logistic = 1.0 / (1.0 + math.exp(-float(np.clip(parameters[position], -35.0, 35.0))))
            phi = 0.80 + 0.195 * logistic
            position += 1
        elif trend_mode == 1:
            phi = 1.0
        else:
            phi = 0.0
        ar_parameters = [0.98 * math.tanh(float(parameters[position + index])) for index in range(p)]
        position += p
        ma_parameters = [0.98 * math.tanh(float(parameters[position + index])) for index in range(q)]

        coefficient_position = 1
        if trend_mode:
            slope = float(coefficients[1])
            coefficient_position = 2
            level = float(coefficients[0]) - phi * slope
        else:
            slope = 0.0
            level = float(coefficients[0])
        seasonal_states = []
        for angle in angles:
            cosine_coefficient = float(coefficients[coefficient_position])
            sine_coefficient = float(coefficients[coefficient_position + 1])
            coefficient_position += 2
            cosine = math.cos(angle)
            sine = math.sin(angle)
            seasonal_states.append([
                cosine * cosine_coefficient - sine * sine_coefficient,
                sine * cosine_coefficient + cosine * sine_coefficient,
            ])

        past_disturbances = [0.0] * p
        past_innovations = [0.0] * q
        squared_error = 0.0
        forecasts = []
        total_steps = len(values) + forecast_steps
        for time_index in range(total_steps):
            predicted_level = level + (phi * slope if trend_mode else 0.0)
            rotated_states = []
            seasonal_total = 0.0
            for state, angle in zip(seasonal_states, angles):
                cosine = math.cos(angle)
                sine = math.sin(angle)
                rotated_one = cosine * state[0] + sine * state[1]
                rotated_two = -sine * state[0] + cosine * state[1]
                rotated_states.append([rotated_one, rotated_two])
                seasonal_total += rotated_one
            arma_prediction = sum(ar_parameters[index] * past_disturbances[index] for index in range(p))
            arma_prediction += sum(ma_parameters[index] * past_innovations[index] for index in range(q))
            conditional_mean = predicted_level + seasonal_total + arma_prediction

            if time_index < len(values):
                innovation = float(values[time_index] - conditional_mean)
                disturbance = arma_prediction + innovation
                squared_error += innovation * innovation
            else:
                innovation = 0.0
                disturbance = arma_prediction
                forecasts.append(float(np.clip(conditional_mean, -1e300, 1e300)))

            level = predicted_level + alpha * disturbance
            if trend_mode:
                slope = phi * slope + beta * disturbance
            seasonal_states = [
                [state[0] + gamma[0] * disturbance, state[1] + gamma[1] * disturbance]
                for state, gamma in zip(rotated_states, gammas)
            ]
            if p:
                past_disturbances = [disturbance] + past_disturbances[:-1]
            if q:
                past_innovations = [innovation] + past_innovations[:-1]
        return squared_error, forecasts

    transformation_options = [None]
    if np.all(original > 0.0):
        estimated_lambda = float(np.clip(boxcox_normmax(original, method="mle"), -2.0, 2.0))
        transformation_options.append(estimated_lambda)

    best = None
    for lam in transformation_options:
        values = transform(original, lam)
        for trend_mode in (0, 1, 2):
            coefficients = initial_regression(values, trend_mode)
            for p, q in ((0, 0), (1, 0), (0, 1), (1, 1)):
                initial_parameters = [-1.38629436112]
                if trend_mode:
                    initial_parameters.append(-2.94443897917)
                initial_parameters.extend([0.0] * (2 * len(angles)))
                if trend_mode == 2:
                    initial_parameters.append(1.20397280433)
                initial_parameters.extend([0.0] * (p + q))
                initial_parameters = np.asarray(initial_parameters, dtype=float)

                def objective(parameters):
                    squared_error, _ = run(parameters, values, trend_mode, p, q, coefficients, 0)
                    return n * math.log(max(squared_error / n, np.finfo(float).tiny))

                result = minimize(objective, initial_parameters, method="L-BFGS-B", options={"maxiter": 250})
                squared_error, _ = run(result.x, values, trend_mode, p, q, coefficients, 0)
                parameter_count = len(result.x) + len(coefficients) + 1
                aic = n * (math.log(2.0 * math.pi) + 1.0 + math.log(max(squared_error / n, np.finfo(float).tiny)))
                if lam is not None:
                    aic -= 2.0 * (lam - 1.0) * float(np.log(original).sum())
                aic += 2.0 * parameter_count
                if math.isfinite(aic) and (best is None or aic < best[0]):
                    best = (aic, lam, trend_mode, p, q, coefficients, result.x)

    if best is None:
        raise RuntimeError("TBATS configuration search produced no finite likelihood")

    _, lam, trend_mode, p, q, coefficients, parameters = best
    fitted_values = transform(original, lam)
    _, transformed_forecasts = run(parameters, fitted_values, trend_mode, p, q, coefficients, horizon)
    forecasts = []
    for value in transformed_forecasts:
        if lam is None:
            forecast = value
        elif abs(lam) < 1e-8:
            forecast = math.exp(float(np.clip(value, -700.0, 700.0)))
        else:
            inner = max(lam * value + 1.0, np.finfo(float).tiny)
            forecast = math.exp(float(np.clip(math.log(inner) / lam, -700.0, 700.0)))
        forecasts.append(float(forecast))
    if len(forecasts) != horizon or not all(math.isfinite(value) for value in forecasts):
        raise RuntimeError("TBATS forecast was not finite")
    return forecasts


def crossformer(history, horizon, frequency):
    """Use when long series have stable dependencies across temporal segments and lag-derived variables."""
    import numpy as np

    if not isinstance(horizon, int) or horizon < 0:
        raise NotApplicable(f"needs a non-negative integer horizon, got {horizon}")
    if len(history) < 16:
        raise NotApplicable(f"needs 16 points, got {len(history)}")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty frequency string, got {frequency!r}")
    if any(not np.isfinite(value) for value in history):
        raise NotApplicable("needs finite history values, got non-finite values")
    if horizon == 0:
        return []

    values = np.asarray(history, dtype=float)
    center = float(np.mean(values))
    scale = float(np.std(values))
    if scale == 0.0:
        return [float(center) for _ in range(horizon)]

    segment_length = min(8, max(4, int(np.sqrt(len(values)))))
    normalized = (values - center) / scale
    padding = (-len(normalized)) % segment_length
    if padding:
        normalized = np.concatenate((np.full(padding, normalized[0]), normalized))
    segments = normalized.reshape(-1, segment_length)

    def softmax(scores):
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exponentials = np.exp(np.clip(shifted, -60.0, 0.0))
        return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

    def cross_time_attention(tokens):
        scores = tokens @ tokens.T / np.sqrt(tokens.shape[1])
        return tokens + softmax(scores) @ tokens

    def routed_cross_dimension_attention(tokens):
        dimensions = tokens.T
        route_count = min(4, dimensions.shape[0])
        positions = np.arange(dimensions.shape[1], dtype=float) + 0.5
        routes = np.vstack([
            np.cos(np.pi * route * positions / dimensions.shape[1])
            for route in range(route_count)
        ])
        routes /= np.maximum(np.linalg.norm(routes, axis=1, keepdims=True), 1e-12)
        dimension_norms = np.maximum(np.linalg.norm(dimensions, axis=1, keepdims=True), 1e-12)
        dimension_keys = dimensions / dimension_norms
        gathering = softmax(routes @ dimension_keys.T / np.sqrt(dimensions.shape[1]))
        routed_values = gathering @ dimensions
        dispatching = softmax(dimensions @ routed_values.T / np.sqrt(dimensions.shape[1]))
        return tokens + 0.5 * (dispatching @ routed_values).T

    def encoder_block(tokens):
        return routed_cross_dimension_attention(cross_time_attention(tokens))

    def hierarchical_encode(tokens):
        base = encoder_block(tokens)
        contexts = []
        level = base
        while level.shape[0] >= 4 and len(contexts) < 3:
            if level.shape[0] % 2:
                level = np.vstack((level[0], level))
            level = 0.5 * (level[0::2] + level[1::2])
            level = encoder_block(level)
            contexts.append(level[-1])
        if contexts:
            base = base + np.mean(np.vstack(contexts), axis=0)
        return base

    generated = []
    working_segments = segments.copy()
    required_segments = (horizon + segment_length - 1) // segment_length
    lower = float(np.min(normalized) - 2.0 * np.std(normalized))
    upper = float(np.max(normalized) + 2.0 * np.std(normalized))

    for step in range(required_segments):
        encoded = hierarchical_encode(working_segments)
        trend = working_segments[-1] - working_segments[-2]
        query = working_segments[-1] + trend / (step + 2.0)
        temporal_weights = softmax((query.reshape(1, -1) @ encoded.T) / np.sqrt(segment_length))
        temporal_context = (temporal_weights @ encoded)[0]
        decoder_tokens = np.vstack((working_segments[-1], query, temporal_context))
        decoded = routed_cross_dimension_attention(cross_time_attention(decoder_tokens))[-1]
        next_segment = np.clip(0.5 * query + 0.5 * decoded, lower, upper)
        generated.extend(next_segment.tolist())
        working_segments = np.vstack((working_segments, next_segment))

    forecast = np.asarray(generated[:horizon]) * scale + center
    return [float(value) for value in forecast]


def micn_multiscale_convolution(history, horizon, frequency):
    """Use when a long regular series has a stable trend and recurring patterns expressed at several temporal scales."""
    import numpy as np

    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if len(history) == 0:
        raise NotApplicable("needs non-empty history, got 0 points")
    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise NotApplicable("needs finite history values, got non-finite values")

    parts = frequency.strip().lower().split()
    if len(parts) != 2 or not parts[0].isdigit():
        raise NotApplicable(f"needs a supported regular frequency, got {frequency!r}")
    quantity = int(parts[0])
    unit = parts[1].rstrip("s")
    cycle_lengths = {
        "second": 60,
        "minute": 60,
        "hour": 24,
        "day": 7,
        "week": 52,
        "month": 12,
        "quarter": 4,
    }
    if quantity <= 0 or unit not in cycle_lengths:
        raise NotApplicable(f"needs a supported positive regular frequency, got {frequency!r}")
    cycle = cycle_lengths[unit]
    if cycle % quantity != 0:
        raise NotApplicable(f"needs a frequency dividing its recurring cycle exactly, got {frequency!r}")
    period = cycle // quantity
    if period < 2:
        raise NotApplicable(f"needs at least 2 observations per recurring cycle, got {period}")
    required = 3 * period
    if len(values) < required:
        raise NotApplicable(f"needs {required} points, got {len(values)}")
    if horizon == 0:
        return []

    left = period // 2
    right = period - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    trend = np.convolve(padded, np.ones(period) / period, mode="valid")
    seasonal = values - trend

    trend_window = min(len(values), 3 * period)
    trend_x = np.arange(trend_window, dtype=float)
    trend_y = trend[-trend_window:]
    trend_design = np.column_stack((np.ones(trend_window), trend_x))
    trend_coef = np.linalg.lstsq(trend_design, trend_y, rcond=None)[0]
    future_x = np.arange(trend_window, trend_window + horizon, dtype=float)
    future_trend = trend_coef[0] + trend_coef[1] * future_x

    phases = np.arange(len(values)) % period
    phase_profile = np.array([
        seasonal[phases == phase].mean() for phase in range(period)
    ])
    phase_profile -= phase_profile.mean()
    future_phase = phase_profile[(np.arange(len(values), len(values) + horizon) % period)]

    branch_forecasts = []
    for scale in (1, 2, 3, 4):
        if scale > period:
            continue
        offset = len(seasonal) % scale
        coarse = seasonal[offset:].reshape(-1, scale).mean(axis=1)
        kernel = max(1, int(round(period / scale)))
        rows = len(coarse) - kernel
        if rows <= 0:
            raise RuntimeError("multiscale convolution received insufficient coarse observations")

        design = np.vstack([
            coarse[index - kernel:index][::-1]
            for index in range(kernel, len(coarse))
        ])
        target = coarse[kernel:]
        center = coarse.mean()
        centered_design = design - center
        centered_target = target - center
        gram = centered_design.T @ centered_design
        ridge = 1e-6 * (np.trace(gram) / kernel + 1.0)
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(kernel),
            centered_design.T @ centered_target,
        )
        coefficient_norm = np.sum(np.abs(coefficients))
        if coefficient_norm > 0.98:
            coefficients *= 0.98 / coefficient_norm

        coarse_horizon = (horizon + scale - 1) // scale
        state = list(coarse[-kernel:] - center)
        coarse_future = []
        for _ in range(coarse_horizon):
            prediction = float(np.dot(coefficients, np.asarray(state[-kernel:][::-1])))
            state.append(prediction)
            coarse_future.append(center + prediction)
        branch_forecasts.append(np.repeat(coarse_future, scale)[:horizon])

    future_seasonal = np.mean(np.vstack([future_phase] + branch_forecasts), axis=0)
    seasonal_limit = max(float(np.max(np.abs(seasonal))), np.finfo(float).eps)
    future_seasonal = np.clip(future_seasonal, -seasonal_limit, seasonal_limit)
    forecast = future_trend + future_seasonal
    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("multiscale convolution produced non-finite forecasts")
    return [float(value) for value in forecast]


def nonstationary_transformer(history, horizon, frequency):
    """Use when local shifts in level and scale are predictable from recent observations and can inform attention-based forecasting."""
    import math
    import torch

    if len(history) < 12:
        raise NotApplicable(f"needs at least 12 points, got {len(history)}")
    if horizon < 0:
        raise NotApplicable(f"needs a non-negative horizon, got {horizon}")

    dtype = torch.float64
    series = torch.tensor(history, dtype=dtype)
    if not bool(torch.isfinite(series).all()):
        raise NotApplicable("needs finite history values, got non-finite values")

    window = min(16, max(4, len(history) // 3))
    dimension = 8
    inputs = series.unfold(0, window, 1)[:-1]
    targets = series[window:]
    global_mean = series.mean()
    global_scale = series.std(unbiased=False).clamp_min(1e-8)

    positions = torch.arange(window, dtype=dtype).unsqueeze(1)
    channels = torch.arange(dimension, dtype=dtype).unsqueeze(0)
    positional = torch.sin((positions + 1.0) / torch.pow(10000.0, channels / dimension))

    embedding_weight = torch.linspace(0.5, 1.5, dimension, dtype=dtype, requires_grad=True)
    query_weight = torch.eye(dimension, dtype=dtype, requires_grad=True)
    key_weight = torch.eye(dimension, dtype=dtype, requires_grad=True)
    value_weight = torch.eye(dimension, dtype=dtype, requires_grad=True)
    output_weight = torch.linspace(-0.2, 0.2, dimension, dtype=dtype, requires_grad=True)
    output_bias = torch.zeros((), dtype=dtype, requires_grad=True)
    tau_weight = torch.zeros(3, dtype=dtype, requires_grad=True)
    delta_weight = torch.zeros((3, window), dtype=dtype, requires_grad=True)

    parameters = [embedding_weight, query_weight, key_weight, value_weight,
                  output_weight, output_bias, tau_weight, delta_weight]
    optimizer = torch.optim.Adam(parameters, lr=0.015)

    means = inputs.mean(dim=1, keepdim=True)
    scales = inputs.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
    normalized = (inputs - means) / scales
    factors = torch.cat(((means - global_mean) / global_scale,
                         scales / global_scale,
                         torch.ones_like(means)), dim=1)

    for _ in range(180):
        optimizer.zero_grad()
        embedded = normalized.unsqueeze(2) * embedding_weight + positional
        queries = embedded @ query_weight
        keys = embedded @ key_weight
        values = embedded @ value_weight
        tau = torch.nn.functional.softplus(factors @ tau_weight).view(-1, 1, 1) + 0.1
        delta = (factors @ delta_weight).unsqueeze(1)
        scores = tau * (queries @ keys.transpose(1, 2)) / math.sqrt(dimension) + delta
        attended = torch.softmax(scores, dim=2) @ values
        normalized_prediction = 10.0 * torch.tanh((attended[:, -1, :] @ output_weight + output_bias) / 10.0)
        prediction = means[:, 0] + scales[:, 0] * normalized_prediction
        loss = torch.mean(((prediction - targets) / global_scale) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()

    extended = series.tolist()
    forecasts = []
    with torch.no_grad():
        for _ in range(horizon):
            current = torch.tensor(extended[-window:], dtype=dtype).view(1, window)
            mean = current.mean(dim=1, keepdim=True)
            scale = current.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
            normalized_current = (current - mean) / scale
            factor = torch.cat(((mean - global_mean) / global_scale,
                                scale / global_scale,
                                torch.ones_like(mean)), dim=1)
            embedded = normalized_current.unsqueeze(2) * embedding_weight + positional
            queries = embedded @ query_weight
            keys = embedded @ key_weight
            values = embedded @ value_weight
            tau = torch.nn.functional.softplus(factor @ tau_weight).view(1, 1, 1) + 0.1
            delta = (factor @ delta_weight).unsqueeze(1)
            scores = tau * (queries @ keys.transpose(1, 2)) / math.sqrt(dimension) + delta
            attended = torch.softmax(scores, dim=2) @ values
            normalized_prediction = 10.0 * torch.tanh((attended[:, -1, :] @ output_weight + output_bias) / 10.0)
            prediction = mean[0, 0] + scale[0, 0] * normalized_prediction[0]
            value = float(prediction.item())
            if not math.isfinite(value):
                raise FloatingPointError("nonstationary_transformer produced a non-finite forecast")
            forecasts.append(value)
            extended.append(value)

    return forecasts


def pyraformer(history, horizon, frequency):
    """Use when long-range dependencies can be captured through a hierarchy of local and cross-scale summaries."""
    import numpy as np

    if len(history) < 64:
        raise NotApplicable(f"needs 64 points, got {len(history)}")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not isinstance(frequency, str) or not frequency.strip():
        raise NotApplicable(f"needs a non-empty frequency string, got {frequency!r}")

    values = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")
    if horizon == 0:
        return []

    center = float(np.mean(values))
    scale = float(np.std(values))
    if scale == 0.0:
        scale = 1.0
    z = (values - center) / scale
    n = len(z)
    depth = int(np.ceil(np.log2(n))) + 1
    fanout = 4

    def make_features(t, cumulative, sequence):
        query = float(sequence[t - 1])
        previous = float(sequence[t - 2])
        recent4 = (cumulative[t] - cumulative[max(0, t - 4)]) / min(4, t)
        recent8 = (cumulative[t] - cumulative[max(0, t - 8)]) / min(8, t)
        features = [query, query - previous, recent4, query - recent8]

        for level in range(depth):
            width = 1 << level
            nodes = []
            for offset in range(fanout):
                end = t - offset * width
                if end <= 0:
                    break
                start = max(0, end - width)
                nodes.append((cumulative[end] - cumulative[start]) / (end - start))

            node_values = np.asarray(nodes, dtype=float)
            distances = np.arange(len(node_values), dtype=float)
            logits = -0.7 * distances - 0.5 * (node_values - query) ** 2
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
            attended = float(np.dot(weights, node_values))
            dispersion = float(np.sqrt(np.dot(weights, (node_values - attended) ** 2)))
            features.extend([attended, dispersion, float(node_values[0]), len(nodes) / fanout])

        return np.asarray(features, dtype=float)

    cumulative = np.concatenate((np.array([0.0]), np.cumsum(z)))
    rows = []
    targets = []
    for t in range(32, n):
        rows.append(make_features(t, cumulative, z))
        targets.append(z[t])

    design = np.vstack(rows)
    target = np.asarray(targets, dtype=float)
    column_center = np.mean(design, axis=0)
    column_scale = np.std(design, axis=0)
    column_scale[column_scale == 0.0] = 1.0
    normalized = (design - column_center) / column_scale
    augmented = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(augmented.shape[1]) * 1e-3
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)

    work = z.tolist()
    cumulative_work = cumulative.tolist()
    limit = max(8.0, 2.0 * float(np.max(np.abs(z))) + 2.0)
    forecast = []
    for _ in range(horizon):
        t = len(work)
        feature = make_features(t, np.asarray(cumulative_work), work)
        row = (feature - column_center) / column_scale
        prediction = float(coefficients[0] + np.dot(row, coefficients[1:]))
        if not np.isfinite(prediction):
            raise FloatingPointError("pyraformer produced a non-finite prediction")
        prediction = float(np.clip(prediction, -limit, limit))
        work.append(prediction)
        cumulative_work.append(cumulative_work[-1] + prediction)
        value = center + scale * prediction
        if not np.isfinite(value):
            raise FloatingPointError("pyraformer produced a non-finite forecast")
        forecast.append(float(value))

    return forecast


def lightts_sampling_mlp(history, horizon, frequency):
    """Use when interval and contiguous sampling preserve the important medium-to-long-range patterns in the series."""
    import numpy as np

    if horizon < 1:
        raise NotApplicable(f"needs a positive horizon, got {horizon}")
    needed = horizon + 24
    if len(history) < needed:
        raise NotApplicable(f"needs {needed} points, got {len(history)}")

    values = np.asarray(history, dtype=float)
    lookback = 16
    hidden_width = 12

    def sampled_views(window):
        center = np.mean(window)
        scale = max(float(np.std(window)), np.finfo(float).eps)
        normalized = (window - center) / scale
        blocks = normalized.reshape(4, 4)
        interval_view = np.concatenate((blocks.T.reshape(-1), blocks.mean(axis=0), blocks.std(axis=0)))
        contiguous_view = np.concatenate((blocks.reshape(-1), blocks.mean(axis=1), blocks.std(axis=1)))
        return interval_view, contiguous_view, center, scale

    feature_size = 24
    indices = np.arange(1, feature_size + 1, dtype=float)[:, None]
    units = np.arange(1, hidden_width + 1, dtype=float)[None, :]
    interval_weights = np.sin(indices * units) / np.sqrt(feature_size)
    contiguous_weights = np.cos(indices * (units + 0.5)) / np.sqrt(feature_size)
    interval_bias = np.sin(np.arange(1, hidden_width + 1, dtype=float))
    contiguous_bias = np.cos(np.arange(1, hidden_width + 1, dtype=float))

    rows = []
    targets = []
    last_start = len(values) - lookback - horizon
    for start in range(last_start + 1):
        window = values[start:start + lookback]
        interval_view, contiguous_view, center, scale = sampled_views(window)
        interval_projection = np.tanh(interval_view @ interval_weights + interval_bias)
        contiguous_projection = np.tanh(contiguous_view @ contiguous_weights + contiguous_bias)
        rows.append(np.concatenate((interval_projection, contiguous_projection, np.ones(1))))
        future = values[start + lookback:start + lookback + horizon]
        targets.append((future - center) / scale)

    design = np.asarray(rows, dtype=float)
    response = np.asarray(targets, dtype=float)
    penalty = 1e-3 * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ response)

    interval_view, contiguous_view, center, scale = sampled_views(values[-lookback:])
    interval_projection = np.tanh(interval_view @ interval_weights + interval_bias)
    contiguous_projection = np.tanh(contiguous_view @ contiguous_weights + contiguous_bias)
    final_features = np.concatenate((interval_projection, contiguous_projection, np.ones(1)))
    forecast = center + scale * (final_features @ coefficients)
    return [float(value) for value in forecast]
