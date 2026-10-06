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
│  │  ├─ investor_presentations/
│  │  └─ manifest.csv
│  └─ processed/             # 清洗後資料
├─ src/
│  ├─ fetch_raw.py
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
- 台燿官方 Investor Relations
- StockGo / MoneyDJ 交叉核對

`data/raw/` 保留原始下載檔；清洗後資料放在 `data/processed/`。

## 作業用歷史資料

`data/processed/TUC_6274_assignment_historical_2016_2025.csv`

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
- Damped Holt-Winters
- Ridge
- Gradient Boosting
- Random Forest

驗證方式：expanding walk-forward validation。

目前最低 mean WAPE 的模型為 **Damped Holt-Winters**。

| Year | Revenue Forecast (NT$ bn) | YoY |
|---|---:|---:|
| 2026E | 61.71 | 103.4% |
| 2027F | 104.65 | 69.6% |
| 2028F | 144.41 | 38.0% |

詳細結果：

- `results/model_summary.csv`
- `results/walk_forward_backtest.csv`
- `results/monthly_forecast.csv`
- `results/annual_forecast.csv`
- `reports/forecast_run_2026-10-06.md`

## 執行

Windows PowerShell：

```powershell
./run.ps1
```

或：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/fetch_raw.py
python src/pipeline.py
pytest -q
```

只重跑模型：

```powershell
.\.venv\Scripts\python.exe src/pipeline.py
```

## License

程式碼採 MIT License。

`data/raw/` 中的第三方與公司文件仍依原始來源之授權與使用條款。
