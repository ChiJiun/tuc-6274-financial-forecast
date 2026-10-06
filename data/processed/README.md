# processed/

Human-readable and model-ready datasets derived from raw sources.

## Assignment-ready historical data

`TUC_6274_assignment_historical_2016_2025.csv` contains:

- Income Statement: Revenue, COGS, Gross Profit, Operating Expenses, Operating Income, PBT, Tax, Net Income, EPS
- Balance Sheet: Cash, A/R, Inventory, Current Assets, PP&E, Total Assets, borrowings, A/P, Current Liabilities, Total Liabilities, Equity
- Table 3.2-style forecast drivers:
  - Sales Growth
  - COGS / Sales
  - Gross Margin
  - Operating Expenses / Sales
  - Operating Margin
  - Days Sales in Cash
  - A/R Collection Days
  - Inventory Turnover
  - A/P Payment Days
  - Effective Tax Rate
  - Dividend Payout

Units are NT$ million except EPS, percentages, days, and turnover.

The 2021–2025 period is the recommended core historical window for the class presentation. 2016–2020 is retained for longer-run trend and robustness checks.

Raw source documents are preserved separately under `../raw/`.
