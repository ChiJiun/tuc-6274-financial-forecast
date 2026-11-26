# Data Completeness through 2026Q3 — 2026-10-11

## Current repository coverage

| Dataset | Coverage | Status | Notes |
|---|---|---|---|
| TUC monthly revenue | 2013-01–2026-09 | Complete for model window | Official MOPS monthly revenue |
| TUC quarterly financial statements | 2016Q1–2026Q2 | Complete through latest published statement | 2026Q3 statement is not yet public |
| 台光電 2383 monthly revenue | 2013-01–2026-09 | Complete | Official MOPS listed-company monthly revenue |
| 聯茂 6213 monthly revenue | 2013-01–2026-09 | Complete | Official MOPS listed-company monthly revenue |
| NTD/USD monthly average | 2013-01–2026-09 | Complete | Central Bank of the Republic of China (Taiwan) |
| Quarterly accounting expansion proxies | 2016Q1–2026Q2 | Complete from statements | PP&E growth, CAPEX/assets, inventory, A/R, debt, CFO |
| Exact historical publication timestamps | Partial | Gap | Current backtest uses conservative availability proxies |
| Product mix / physical capacity / utilization | Incomplete | Gap | IR documents are not a consistent historical machine-readable series |
| Raw-material prices / industry production | Not yet included | Gap | Optional future extension from official/public sources |

## 2026Q3 interpretation

The model includes all information currently available through the end of 2026Q3 that is present in the selected public series:

- TUC September 2026 monthly revenue.
- Peer September 2026 monthly revenue.
- September 2026 NTD/USD monthly average.
- TUC 2026Q1 and 2026Q2 official financial statements.

It does **not** fabricate a 2026Q3 financial statement. As of 2026-10-11, TUC's official investor-relations page lists 2026Q1 and 2026Q2 presentations but no 2026Q3 presentation. Taiwan listed/OTC first- and third-quarter financial reports generally have a T+45-day filing deadline, so a 2026Q3 statement is not expected to be available yet.

## Remaining data gaps

The current public-data pipeline is sufficient for the assignment, but a research-grade forecasting extension would benefit most from company-specific forward-looking operating variables rather than another vendor copy of the same financial statements.

Highest-value remaining variables are:

1. product mix / high-end CCL share;
2. physical capacity and plant ramp timing;
3. utilization;
4. ASP / price changes;
5. order or demand indicators when publicly disclosed;
6. exact historical disclosure timestamps when available from official archives.

## Recommendation

For the current assignment, the existing public-data pipeline is sufficient to produce a defensible forecast and sensitivity analysis. Further work should prioritize company-specific capacity, product-mix, pricing and demand variables rather than duplicating the same accounting data from another vendor.
