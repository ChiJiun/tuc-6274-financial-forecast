# TUC-only Financial Feature Comparison

This report answers one narrow question only:

> Does adding TUC's own quarterly financial information improve revenue forecasting enough to beat the original revenue-only production model?

No peer-company revenue and no FX variables are used in this comparison.

Quarterly financial features:

- gross margin / operating margin
- inventory / assets
- PP&E / assets
- A/R / assets
- cash / assets
- debt / assets
- YTD CAPEX / assets
- YTD CFO / assets
- YoY PP&E / inventory / A/R growth

## Results

| Stage | Horizon | Model | Revenue-only WAPE | + TUC financials WAPE | Improvement | HW WAPE | Beats HW? |
|---|---:|---|---:|---:|---:|---:|---|
| validation | 1m | Ridge | 8.80% | 9.52% | -0.72 pp | 6.09% | no |
| validation | 1m | GradientBoosting | 7.67% | 7.16% | +0.52 pp | 6.09% | no |
| holdout | 1m | Ridge | 10.41% | 10.37% | +0.04 pp | 7.94% | no |
| holdout | 1m | GradientBoosting | 12.77% | 13.35% | -0.59 pp | 7.94% | no |
| validation | 3m | Ridge | 10.29% | 8.95% | +1.34 pp | 6.65% | no |
| validation | 3m | GradientBoosting | 9.72% | 9.93% | -0.21 pp | 6.65% | no |
| holdout | 3m | Ridge | 13.43% | 15.14% | -1.71 pp | 10.40% | no |
| holdout | 3m | GradientBoosting | 15.75% | 16.08% | -0.33 pp | 10.40% | no |
| validation | 6m | Ridge | 14.95% | 11.83% | +3.12 pp | 9.72% | no |
| validation | 6m | GradientBoosting | 14.55% | 13.32% | +1.23 pp | 9.72% | no |
| holdout | 6m | Ridge | 20.55% | 22.66% | -2.11 pp | 15.07% | no |
| holdout | 6m | GradientBoosting | 24.45% | 25.03% | -0.57 pp | 15.07% | no |
| validation | 12m | Ridge | 19.58% | 16.68% | +2.90 pp | 15.17% | no |
| validation | 12m | GradientBoosting | 18.26% | 16.30% | +1.96 pp | 15.17% | no |
| holdout | 12m | Ridge | 31.60% | 31.66% | -0.05 pp | 21.64% | no |
| holdout | 12m | GradientBoosting | 29.58% | 29.49% | +0.09 pp | 21.64% | no |
| validation | 24m | Ridge | 21.89% | 24.66% | -2.77 pp | 22.88% | no |
| validation | 24m | GradientBoosting | 22.11% | 21.41% | +0.70 pp | 22.88% | yes |
| holdout | 24m | Ridge | 41.60% | 37.93% | +3.67 pp | 32.13% | no |
| holdout | 24m | GradientBoosting | 43.18% | 41.44% | +1.74 pp | 32.13% | no |

## Conclusion

Adding TUC's own financial information improves some ML configurations during development, but it does not beat HW_Damped_Add on any later holdout horizon in the current experiment.

- 1m holdout: HW 7.94% vs best TUC-financial ML 10.37% (Ridge).
- 3m holdout: HW 10.40% vs best TUC-financial ML 15.14% (Ridge).
- 6m holdout: HW 15.07% vs best TUC-financial ML 22.66% (Ridge).
- 12m holdout: HW 21.64% vs best TUC-financial ML 29.49% (GradientBoosting).
- 24m holdout: HW 32.13% vs best TUC-financial ML 37.93% (Ridge).

Therefore the production revenue forecast remains HW_Damped_Add. The TUC financial-feature models remain useful as explanatory / robustness checks.
