# Model Selection — 2026-10-07

## Selection protocol

- Data: monthly revenue, 2013-01 through 2026-09
- Forecast horizon per fold: 12 months
- Origins: quarter-end rolling origins from 2020-12
- Complete folds: 20 per model
- Primary metric: mean WAPE
- Tie-breakers: median WAPE, mean absolute annual-total error, model name
- Partial-year 2026 is excluded from production model selection and retained only as an annual-origin stress test.

## Selection result

| Model | Mean WAPE | Median WAPE | Mean abs annual-total error |
|---|---:|---:|---:|
| **HW_Damped_Add** | **17.76%** | 18.23% | **16.66%** |
| HW_Log_Damped_Add | 17.94% | 18.33% | 16.93% |
| HW_Damped_Mul | 17.97% | **17.86%** | 16.93% |
| GradientBoosting | 18.98% | 16.78% | 18.00% |
| RandomForest | 20.54% | 16.84% | 19.69% |
| Ridge | 22.08% | 20.14% | 20.84% |
| SeasonalNaive | 24.70% | 24.51% | 22.11% |

Production model: **Holt-Winters with damped additive trend and additive seasonality** (`HW_Damped_Add`).

## Forecast

| Year | Revenue (NT$ bn) | YoY |
|---|---:|---:|
| 2026E | 62.59 | 106.3% |
| 2027F | 109.36 | 74.7% |
| 2028F | 152.25 | 39.2% |

## Stress test

Annual-origin diagnostics are written to:

- `results/walk_forward_backtest.csv`
- `results/annual_backtest_summary.csv`

These diagnostics are not used to select the production model.
