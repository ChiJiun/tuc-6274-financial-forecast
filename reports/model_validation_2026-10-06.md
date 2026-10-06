# Model Validation — 2026-10-06

## Data

- Source: MOPS OTC historical monthly revenue files under `data/raw/mops_monthly/`
- Model window: 2013-01 to 2026-09
- Observations: 165 months
- Latest actual month: 2026-09, NT$6.126bn
- 2026 Jan–Sep revenue: NT$42.304bn
- 2025 Jan–Sep revenue: NT$21.215bn
- YTD YoY: +99.4%

## Primary annual-origin backtest

Expanding training window. Each test year is forecast recursively from the prior year-end. 2021–2025 are 12-month tests; 2026 tests Jan–Sep because only nine months are observed.

| Model | Mean MAPE | Mean WAPE | Mean absolute annual-total error |
|---|---:|---:|---:|
| Damped Holt-Winters (multiplicative seasonality) | 19.14% | **19.99%** | 18.38% |
| Gradient Boosting | 20.77% | 21.63% | 20.89% |
| Random Forest | 21.73% | 22.68% | 21.95% |
| Ridge | 22.87% | 23.70% | 22.97% |
| Seasonal Naive | 26.02% | 26.31% | 24.82% |

By year, the lowest-WAPE model was:

- 2021: Ridge
- 2022: Gradient Boosting
- 2023: Damped Holt-Winters
- 2024: Ridge
- 2025: Damped Holt-Winters
- 2026 Jan–Sep: Damped Holt-Winters

Damped Holt-Winters wins 3 of 6 annual-origin folds and has the lowest average WAPE when all folds are included.

### Sensitivity to evaluation window

Using only complete 2021–2025 calendar-year folds:

| Model | Mean WAPE |
|---|---:|
| Gradient Boosting | 18.10% |
| Damped Holt-Winters | 18.80% |
| Random Forest | 19.11% |
| Ridge | 21.09% |
| Seasonal Naive | 21.60% |

Using the more recent 2023–2025 folds:

| Model | Mean WAPE |
|---|---:|
| Damped Holt-Winters | **17.53%** |
| Gradient Boosting | 19.99% |
| Random Forest | 20.92% |
| Seasonal Naive | 24.72% |
| Ridge | 24.81% |

A simple recency-weighted score over 2021–2026 also ranks Damped Holt-Winters first at 20.85% WAPE.

Conclusion: the exact winner depends slightly on the evaluation window, but trend/seasonal exponential-smoothing models are the most stable on the recent regime.

## Additional 12-month rolling-origin robustness test

To reduce dependence on only six annual folds, 12-month forecasts were also evaluated from 20 quarterly origins from 2020-12 through 2025-09.

| Model | Mean WAPE | Median WAPE | 75th percentile WAPE |
|---|---:|---:|---:|
| Holt-Winters, damped trend + additive seasonality | **17.76%** | 18.23% | 21.72% |
| Holt-Winters on log revenue, damped + additive seasonality | 17.94% | 18.33% | 22.46% |
| Holt-Winters, damped trend + multiplicative seasonality | 17.97% | 17.86% | 22.42% |
| Gradient Boosting | 18.98% | 16.78% | 19.86% |
| Random Forest | 20.54% | 16.84% | 26.15% |
| Ridge | 22.08% | 20.14% | 28.69% |
| Seasonal Naive | 24.70% | 24.51% | 28.52% |

The three Holt-Winters specifications are effectively clustered around 17.8–18.0% mean WAPE. Additive seasonality is marginally best, but the difference from the current multiplicative specification is only about 0.2 percentage points.

## 2026 structural break test

Train only through 2025-12 and forecast 2026 Jan–Sep.

Actual 2026 Jan–Sep revenue: **NT$42.304bn**

| Model | Jan–Sep WAPE | Total forecast error |
|---|---:|---:|
| Damped Holt-Winters | **25.94%** | -25.15% |
| Ridge | 36.75% | -35.56% |
| Gradient Boosting | 39.27% | -38.80% |
| Random Forest | 40.52% | -40.30% |
| Seasonal Naive | 49.85% | -49.85% |

All models materially underforecast the 2026 acceleration. The time-series model reacts once the new observations enter the training sample, but it cannot anticipate an external regime change from pre-2026 history alone.

## Current forecast

Selected production specification: Damped Holt-Winters with damped additive trend and multiplicative seasonality.

| Year | Revenue | YoY |
|---|---:|---:|
| 2025A | NT$30.34bn | — |
| 2026E | **NT$61.71bn** | **+103.4%** |
| 2027F | **NT$104.65bn** | **+69.6%** |
| 2028F | **NT$144.41bn** | **+38.0%** |

2026E consists of NT$42.304bn actual Jan–Sep plus NT$19.402bn forecast Q4.

The Q4 forecast averages NT$6.47bn per month, about 5.6% above September 2026 actual revenue. It implies Q4 revenue growth of roughly 112.5% YoY.

## Forecast-specification sensitivity

Full-sample forecasts from alternative exponential-smoothing specifications:

| Specification | 2026 Q4 | 2027 | 2028 |
|---|---:|---:|---:|
| Damped trend + multiplicative seasonality | 19.40 | 104.65 | 144.41 |
| Damped trend + additive seasonality | 20.28 | 109.36 | 152.25 |
| Undamped trend + multiplicative seasonality | 19.44 | 106.00 | 149.48 |
| Log revenue + damped additive model | 19.43 | 109.10 | 177.20 |
| Seasonal-only model | 17.06 | 68.18 | 68.18 |

Units: NT$bn.

The trend-based exponential-smoothing variants are relatively close for 2027 (roughly NT$105–109bn) but spread widely by 2028 (roughly NT$144–177bn). Therefore the 2027 level is materially more robust than the 2028 level.

Cross-family model dispersion is even larger:

| Model | 2027 | 2028 |
|---|---:|---:|
| Seasonal Naive | 51.43 | 51.43 |
| Damped Holt-Winters | 104.65 | 144.41 |
| Ridge | 118.87 | 259.52 |
| Gradient Boosting | 60.53 | 57.37 |
| Random Forest | 56.45 | 55.37 |

Tree models flatten because they cannot extrapolate trend outside the range seen in training. Ridge extrapolates aggressively and becomes unstable at long horizons. This is why the model spread should not be interpreted as a conventional confidence interval.

## Interpretation

1. **2026E is the most anchored forecast.** Nine months are already actual. The model only needs to estimate Q4.
2. **The direction of 2027 growth is supported by the recent revenue regime, but +69.6% is an aggressive point estimate.** Trend-based ETS specifications cluster near NT$105–109bn, which supports the level internally, but the historical backtest error remains around 18–20%.
3. **2028 has low point-estimate confidence.** Recursive long-horizon extrapolation compounds model assumptions, and reasonable ETS specifications already span NT$144–177bn.
4. **Complex ML does not improve this dataset.** Gradient Boosting and Random Forest fit nonlinear history but do not extrapolate the 2026 regime well. Ridge can extrapolate but becomes too aggressive.
5. **For Pro Forma sensitivity analysis, use the forecast as a base case rather than a single deterministic truth.** Revenue-growth sensitivity is necessary, especially for 2027–2028.

