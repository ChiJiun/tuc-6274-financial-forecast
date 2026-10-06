# 台燿科技（6274）Revenue Forecast — Completed Run

## Scope

- Company: 台燿科技（TUC, 6274）
- Frequency: monthly revenue
- Modeling window: 2013-01 to 2026-09
- Observations: 165
- Forecast horizon: 2026Q4–2028
- Validation: expanding walk-forward, no random train/test split

## Models compared

| Model | Mean MAPE | Mean WAPE | Mean absolute annual-total error |
|---|---:|---:|---:|
| Damped Holt-Winters | 19.1% | **20.0%** | 18.4% |
| Gradient Boosting | 20.8% | 21.7% | 20.9% |
| Random Forest | 21.7% | 22.7% | 22.0% |
| Ridge | 22.9% | 23.7% | 23.0% |
| Seasonal Naive | 26.0% | 26.3% | 24.8% |

Selected model: **Damped Holt-Winters** by lowest mean out-of-sample WAPE.

The main result is that the more complex ML models did **not** beat the classical time-series model on this dataset.

## Selected annual forecast

| Year | Revenue (NT$ bn) | YoY |
|---|---:|---:|
| 2025A | 30.34 | — |
| 2026E | 61.71 | 103.4% |
| 2027F | 104.65 | 69.6% |
| 2028F | 144.41 | 38.0% |

Public analyst-consensus references used for comparison in the completed run:

- 2026E: 59.98 bn
- 2027E: 105.02 bn
- 2028E: 174.00 bn

The model is close to consensus for 2026/2027 and materially more conservative for 2028.

## 2026 structural break

Using only data available through 2025-12, the selected model forecast 2026 Jan–Sep revenue at **31.66 bn**, versus actual **42.30 bn**, an annual-total error of roughly **-25.2%**.

This is evidence that a history-only revenue model does not capture exogenous information such as:

- AI/HPC and high-speed networking demand
- high-end CCL product mix
- ASP changes
- capacity expansion / Thailand ramp
- raw-material and FX effects

For the corporate-finance assignment, this forecast should therefore be treated as an **independent quantitative anchor** for the Sales Growth assumption, not as the sole forecast.

## Files

- `data/processed/TUC_6274_assignment_historical_2016_2025.csv` — human-readable historical financials and Table 3.2-style ratios
- `results/model_summary.csv` — model ranking
- `results/walk_forward_backtest.csv` — all walk-forward test-year results
- `results/monthly_forecast.csv` — 2026Q4–2028 monthly forecast comparison
- `results/annual_forecast.csv` — selected annual forecast and consensus comparison

## Data provenance note

The completed forecast run used the source-reconstructed comparable monthly revenue series assembled from StockGo/MoneyDJ and cross-checked against company/MOPS disclosures.

The repository additionally preserves the original MOPS historical monthly-revenue HTML files and TUC investor-relations documents under `data/raw/`, together with `data/raw/manifest.csv`, so the next rebuild can use the raw archive as the primary source.
