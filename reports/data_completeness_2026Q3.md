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
| Raw-material prices / industry production | Not yet included | Gap | Potential TEJ / official macro source extension |

## 2026Q3 interpretation

The model includes all information currently available through the end of 2026Q3 that is present in the selected public series:

- TUC September 2026 monthly revenue.
- Peer September 2026 monthly revenue.
- September 2026 NTD/USD monthly average.
- TUC 2026Q1 and 2026Q2 official financial statements.

It does **not** fabricate a 2026Q3 financial statement. As of 2026-10-11, TUC's official investor-relations page lists 2026Q1 and 2026Q2 presentations but no 2026Q3 presentation. Taiwan listed/OTC first- and third-quarter financial reports generally have a T+45-day filing deadline, so a 2026Q3 statement is not expected to be available yet.

## TEJ Pro value

TEJ Pro can materially improve the research dataset even though it cannot provide an unpublished 2026Q3 financial statement.

High-value downloads:

1. **Monthly revenue, version/release-date fields**
   - TUC 6274, 台光電 2383, 聯茂 6213 and a broader CCL/PCB peer set.
   - Include the actual revenue announcement date and versioned revenue fields where available.

2. **Quarterly financial data**
   - Full consolidated financial statement fields.
   - Gross margin, operating margin, inventory, receivables, PP&E, depreciation, operating cash flow, CAPEX and debt.
   - Prefer both cumulative and single-quarter fields when available.

3. **Financial-report event dates**
   - Exact quarterly and annual report announcement dates.
   - This would replace the current conservative proxy release dates and make point-in-time backtests more rigorous.

4. **Product / sales mix**
   - Monthly or annual product-sales composition where available.
   - Useful for high-end CCL mix and AI/networking exposure.

5. **Capital formation / investment / borrowing**
   - Capital formation, borrowing details, long-term investment and property changes.
   - Useful expansion proxies beyond accounting PP&E alone.

6. **Macro / commodity series**
   - FX, copper and other relevant raw-material or electronics-industry indicators if included in the school's subscribed modules.

The National Central University TEJ Pro page confirms school-email access, but the exact modules available depend on NCU's subscription. Export availability should therefore be checked inside TEJ Pro / TEJ NEXT before assuming every TEJ database is licensed.

## Recommendation

For the current assignment, the public-data pipeline is sufficient to produce a defensible forecast and sensitivity analysis. For research-quality model comparison, TEJ data is worth downloading primarily because it provides richer historical fields and actual release/event dates—not because it will reveal a 2026Q3 financial statement before the company publishes it.
