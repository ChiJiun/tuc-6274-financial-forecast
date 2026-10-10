# Issue #14 Phase 2 — Quarterly financials + peer revenue + FX

## Information cutoff

- TUC monthly revenue: through 2026-09.
- Peer monthly revenue (台光電 2383, 聯茂 6213): through 2026-09.
- CBC NTD/USD monthly average: through 2026-09.
- TUC official financial statements: through **2026Q2**.
- 2026Q3 financial statement is not included because it was not public as of 2026-10-11; the official TUC IR page listed only 2026Q1 and 2026Q2.

Financial feature groups include gross margin, operating margin, inventory/assets, PP&E/assets, receivables/assets, cash/assets, debt/assets, YTD CAPEX/assets, YTD CFO/assets, and YoY PP&E/inventory/receivables growth. PP&E growth and CAPEX are explicit expansion-capacity proxies.

## Evaluation design

- Quarterly rolling origins from 2020-12.
- Validation/model-selection origins through 2023-09.
- Later holdout origins from 2023-12 onward.
- Primary comparison: fixed 12-month path WAPE.
- All tabular feature groups use the same common point-in-time availability window.
- Future realized peer/FX/financial values are never supplied to multi-step forecasts; models use origin-known state plus forecast lead/calendar.

## 12-month validation ranking

| Model | Features | Folds | Mean WAPE | Median WAPE | Bias |
|---|---|---:|---:|---:|---:|
| HW_Damped_Add | revenue_only | 12 | 15.17% | 15.96% | +2.46% |
| GradientBoosting | all | 12 | 16.16% | 15.60% | +0.58% |
| GradientBoosting | quarterly_expanded | 12 | 16.30% | 15.29% | +0.79% |
| Ridge | quarterly_expanded | 12 | 16.68% | 15.69% | +1.05% |
| Ridge | all | 12 | 16.97% | 14.78% | +1.30% |
| Ridge | peer_fx | 12 | 17.25% | 17.18% | +3.78% |
| GradientBoosting | peer_fx | 12 | 17.37% | 16.71% | +1.16% |
| GradientBoosting | revenue_only | 12 | 18.26% | 16.44% | +1.96% |
| Ridge | revenue_only | 12 | 19.58% | 14.40% | +9.22% |

## 12-month later holdout ranking

| Model | Features | Folds | Mean WAPE | Median WAPE | Bias |
|---|---|---:|---:|---:|---:|
| HW_Damped_Add | revenue_only | 8 | 21.64% | 22.73% | -20.43% |
| Ridge | peer_fx | 8 | 27.24% | 26.42% | -26.44% |
| Ridge | all | 8 | 28.39% | 26.75% | -28.14% |
| GradientBoosting | peer_fx | 8 | 28.45% | 25.75% | -28.30% |
| GradientBoosting | all | 8 | 29.11% | 25.53% | -28.91% |
| GradientBoosting | quarterly_expanded | 8 | 29.49% | 25.40% | -29.31% |
| GradientBoosting | revenue_only | 8 | 29.58% | 27.80% | -29.48% |
| Ridge | revenue_only | 8 | 31.60% | 26.34% | -31.23% |
| Ridge | quarterly_expanded | 8 | 31.66% | 28.23% | -31.37% |

## Prespecified conclusion

- Validation winner: **HW_Damped_Add__revenue_only**, mean WAPE 15.17%.
- Its later-holdout WAPE: **21.64%**.
- Holt-Winters later-holdout WAPE: **21.64%**.
- Best non-HW validation candidate: **GradientBoosting__all** (16.16%), later-holdout **29.11%**.

The production model should only change if an externally enriched candidate beats Holt-Winters on the later holdout with the same target windows. Otherwise the external model remains an explanatory robustness check.

## Current forecast outputs

The phase-2 current forecast is emitted only through 2027-12 (15 months from the 2026-09 origin). This avoids presenting unvalidated 27-month exogenous extrapolation as a 2028 point forecast.

| Model | Features | Year | Forecast months total (NT$ bn) |
|---|---|---:|---:|
| GradientBoosting | all | 2026 | 11.50 |
| GradientBoosting | all | 2027 | 60.73 |
| HW_Damped_Add | revenue_only | 2026 | 20.28 |
| HW_Damped_Add | revenue_only | 2027 | 109.36 |

## Limitations

- Quarterly financial release dates use conservative proxy dates (Q1 Jun 1, Q2 Sep 1, Q3 Dec 1, annual next Apr 1), not original historical filing timestamps.
- Historical MOPS pages may reflect later presentation/revision conventions.
- Peer choice is limited to two major Taiwan CCL comparables; broader industry/capacity/raw-material series could still add information.
- Overlapping forecast paths are correlated and are not independent significance tests.
