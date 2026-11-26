# Issue #14 — Annual financial features (first ablation)

## Scope

- **Experiment only:** production forecasts and model selection remain unchanged.
- Source: committed 2016–2025 official-MOPS reconstructed annual financials plus monthly revenue.
- Financial inputs: latest presumed-published gross margin, operating margin, inventory/assets, PP&E/assets, and receivables/assets.
- Financial statement availability: **July 1 of the following year** (conservative proxy; **not verified actual announcement dates**).
- Revenue forecast origin: **16th of the following month** (calendar proxy; publication dates not verified).
- Direct multi-step forecast: all features at origin; only known target month and lead are used for future horizons. No future financial values.
- Same quarterly origins and observed targets for every candidate; selection through 2023-09, subsequent holdout from 2023-12.
- The holdout is newly reserved for this ablation, but **not independent of the previously selected HW baseline**.
- Overlapping rolling folds are correlated; no formal significance claim.

## 12-month results

| Stage | Model / features | Folds | Mean WAPE | Mean signed bias |
|---|---|---:|---:|---:|
| holdout | HW_Damped_Add / revenue_only | 8 | 21.64% | -20.43% |
| holdout | GradientBoosting / revenue_plus_financials | 8 | 29.60% | -29.46% |
| holdout | GradientBoosting / revenue_only | 8 | 29.93% | -29.82% |
| holdout | Ridge / revenue_only | 8 | 31.77% | -31.38% |
| holdout | Ridge / revenue_plus_financials | 8 | 32.98% | -32.73% |
| validation | HW_Damped_Add / revenue_only | 12 | 15.17% | +2.46% |
| validation | GradientBoosting / revenue_plus_financials | 12 | 17.62% | +1.01% |
| validation | GradientBoosting / revenue_only | 12 | 18.13% | +1.88% |
| validation | Ridge / revenue_plus_financials | 12 | 18.27% | +2.81% |
| validation | Ridge / revenue_only | 12 | 19.55% | +9.22% |

## Prespecified selection

- Best 12-month **validation** candidate: **HW_Damped_Add + revenue_only** (15.17% mean WAPE).
- Validation baseline: HW_Damped_Add (15.17% mean WAPE).
- Later **holdout**: chosen candidate 21.64% versus baseline 21.64%.

## Limitations and next steps

- **Do not treat the 17.76% production benchmark as directly comparable to the validation-only score.** Compare on the identical stage/origins.
- Only reconstructed annual financial statements were added; peer monthly revenue, FX and industry series are **not yet implemented**.
- The publication-date policy is a proxy. Replace with verifiable original announcements and handle revisions before claiming historical point-in-time accuracy.
- The chosen candidate is selected once from validation; do not tune after looking at the holdout.
- Annual ratios provide relatively few distinct states; this is exploratory and may not improve accuracy.
- Evaluate 1/3/6/12-month horizons in the score CSV; a 24-month study remains follow-up.
- Revisit the 2026 regime-change performance and multi-company/industry sources before promoting an alternate production model.
