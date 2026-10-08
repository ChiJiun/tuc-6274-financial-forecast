# 台燿科技（6274）Financial Forecast

台燿科技（TUC, 6274）財務預測專案，包含：

- 原始歷史資料
- 清洗後財務資料
- 月營收特徵工程
- Walk-forward backtest
- 2026–2028 營收預測
- 作業用歷史財務資料與比率

## 專案結構

```text
.
├─ data/
│  ├─ raw/                    # 原始下載資料
│  │  ├─ mops_monthly/       # MOPS 歷史月營收
│  │  ├─ mops_financials/    # MOPS 合併財報
│  │  ├─ investor_presentations/
│  │  └─ manifest.csv
│  └─ processed/             # 清洗後資料
├─ src/
│  ├─ fetch_raw.py
│  ├─ build_assignment_financials.py
│  ├─ build_scenarios.py
│  └─ pipeline.py
├─ results/                  # Backtest 與 forecast 結果
├─ reports/                  # 結果說明
├─ tests/
├─ .github/workflows/ci.yml
├─ requirements.txt
└─ run.ps1
```

## 資料來源

主要來源：

- MOPS 歷史月營收（OTC）
- MOPS 官方合併財務報表
- 台燿官方 Investor Relations
- StockGo / MoneyDJ 交叉核對

`data/raw/` 保留原始下載檔；清洗後資料放在 `data/processed/`。

## 作業用歷史資料

`data/processed/TUC_6274_assignment_historical_2016_2025.csv`

由 `src/build_assignment_financials.py` 從 `data/raw/mops_financials/` 的官方 MOPS 財報重建；來源欄位與檢核結果分別在 `TUC_6274_assignment_provenance.csv`、`TUC_6274_assignment_reconciliation.csv`。

包含：

- Revenue、COGS、Gross Profit
- Operating Expenses、Operating Income
- PBT、Tax、Net Income、EPS
- Cash、A/R、Inventory、A/P、PP&E、Debt、Equity
- Sales Growth
- COGS / Sales
- Gross Margin
- Operating Expenses / Sales
- A/R Collection Days
- Inventory Turnover
- A/P Payment Days
- Effective Tax Rate
- Dividend Payout

建議作業主要使用 2021–2025，2016–2020 作為較長期歷史參考。

## 模型

比較模型：

- Seasonal Naive
- Holt-Winters / ETS variants
- Ridge
- Gradient Boosting
- Random Forest

模型選擇使用固定 12 個月 horizon 的 quarterly rolling-origin validation，以 mean WAPE 最低者為 production model；partial-year 2026 僅作 stress test。

目前選擇 **Holt-Winters（damped trend + additive seasonality）**。

| Year | Revenue Forecast (NT$ bn) | YoY |
|---|---:|---:|
| 2026E | 62.59 | 106.3% |
| 2027F | 109.36 | 74.7% |
| 2028F | 152.25 | 39.2% |

詳細結果：

- `results/model_summary.csv`
- `results/model_selection_12m_quarterly.csv`
- `results/walk_forward_backtest.csv`
- `results/partial_year_stress_test.csv`
- `results/monthly_forecast.csv`
- `results/annual_forecast.csv`
- `results/forecast_uncertainty.csv`
- `results/pro_forma_revenue_scenarios.csv`
- `results/pro_forma_growth_margin_sensitivity.csv`
- `reports/forecast_uncertainty_and_sensitivity.md`
- `reports/model_selection_2026-10-07.md`
- `reports/forecast_run_2026-10-06.md`（historical snapshot）
- `reports/model_validation_2026-10-06.md`（historical snapshot）

## 執行

Windows PowerShell：

```powershell
./run.ps1
./run.ps1 -AsOf 2026-09 -ForecastEnd 2028-12  # 重現指定資料截止日
```

或：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt

python src/fetch_raw.py
python src/build_assignment_financials.py
python src/pipeline.py --forecast-end 2028-12
python src/validate_models.py
python src/build_scenarios.py
pytest -q
```

只重跑模型：

```powershell
.\.venv\Scripts\python.exe src/pipeline.py
```

## License

程式碼採 MIT License。

`data/raw/` 中的第三方與公司文件仍依原始來源之授權與使用條款。
