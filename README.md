# 台燿科技（6274）Financial Forecast / ML Research

以台燿科技（TUC, 6274）為研究對象，建立一個**可重現**的財務預測專案。專案同時保留原始歷史資料、清洗後資料、特徵工程、walk-forward backtest、模型比較，以及 2027/2028 revenue forecast。

> 目標不是用 ML 取代財務分析，而是用 data-driven forecast 作為課本 Pro Forma Financial Statements 中 `Sales Growth` assumption 的獨立量化依據。

## 專案結構

```text
.
├─ data/
│  ├─ raw/                    # 原始下載檔，不經人工改值
│  │  ├─ mops_monthly/       # MOPS 歷史月營收 HTML
│  │  ├─ investor_presentations/
│  │  └─ manifest.csv
│  └─ processed/             # 清洗後可建模資料
├─ src/
│  ├─ fetch_raw.py           # 抓官方/公開原始資料
│  └─ pipeline.py            # 清洗、特徵、訓練、backtest、forecast
├─ results/                  # backtest / forecast CSV
├─ reports/
│  ├─ figures/
│  └─ model_output.xlsx
├─ tests/
├─ .github/workflows/ci.yml
├─ requirements.txt
└─ run.ps1
```

## 資料來源

主要 raw source：

1. **MOPS 歷史月營收（OTC）**  
   `https://mopsov.twse.com.tw/nas/t21/otc/t21sc03_{ROC_YEAR}_{MONTH}_0.html`  
   台燿為上櫃公司，因此必須使用 `otc`，不是 `sii`。
2. **台燿官方 Investor Relations**  
   `https://www.tuc.com.tw/zh-tw/corporate/id/133`
3. 交叉核對用快照：StockGo / MoneyDJ。
4. 最新公開資料可另外由 TPEx/TWSE OpenAPI 取得。

`data/raw/` 只保存下載原文與 checksum；建模時一律讀取 raw 再產出 `data/processed/`。

## 為什麼主要模型從 2013-01 開始

2013 年 IFRSs 實施後，營收揭露口徑較一致。本 repo 可以保存更早的原始資料，但預設 model window 從 2013-01 開始，避免把不同會計制度直接當成同一條 stationary time series。

## 模型

目前比較：

- Seasonal Naive
- Damped Holt-Winters
- Ridge
- Gradient Boosting
- Random Forest

驗證方式使用 **expanding walk-forward validation**，不做 random train/test split。

例如：

```text
Train <= 2020-12 -> Forecast 2021
Train <= 2021-12 -> Forecast 2022
Train <= 2022-12 -> Forecast 2023
...
```

選模依據是 out-of-sample WAPE / MAPE，而不是 in-sample fit。

## Windows 本機執行

PowerShell：

```powershell
./run.ps1
```

手動流程：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/fetch_raw.py
python src/pipeline.py
pytest -q
```

只重跑模型、不重新下載 raw：

```powershell
.\.venv\Scripts\python.exe src/pipeline.py
```

## 已提交的資料與模型結果

### 作業用歷史資料

`data/processed/TUC_6274_assignment_historical_2016_2025.csv` 是給人直接閱讀、做 Table 3.1 / 3.2 的整理版，包含：

- 2016–2025 Income Statement / Balance Sheet 主要科目
- Sales Growth、COGS/Sales、Gross Margin、OpEx/Sales
- Days Sales in Cash、A/R Days、Inventory Turnover、A/P Days
- Effective Tax Rate、Dividend Payout

作業建議以 **2021–2025** 當核心 historical window，2016–2020 用於長期趨勢與穩健性檢查。

### 已完成 forecast run

目前 repo 保存本次已跑完的輸出：

- `results/walk_forward_backtest.csv`
- `results/model_summary.csv`
- `results/monthly_forecast.csv`
- `results/annual_forecast.csv`
- `reports/forecast_run_2026-10-06.md`

本次 walk-forward backtest 中，**Damped Holt-Winters** 的 mean WAPE 最低（約 20.0%），優於 Ridge、Gradient Boosting、Random Forest 與 Seasonal Naive。

Selected annual forecast：

| Year | Revenue (NT$ bn) | YoY |
|---|---:|---:|
| 2026E | 61.71 | 103.4% |
| 2027F | 104.65 | 69.6% |
| 2028F | 144.41 | 38.0% |

完整解讀與限制請看 `reports/forecast_run_2026-10-06.md`。

### Pipeline 重新執行時的輸出

`pipeline.py` 預計產生：

- `data/processed/monthly_revenue.csv`
- `data/processed/ml_features.csv`
- `results/walk_forward_backtest.csv`
- `results/model_summary.csv`
- `results/monthly_forecast.csv`
- `results/annual_forecast.csv`
- `reports/figures/history_and_forecast.png`
- `reports/figures/model_backtest_wape.png`
- `reports/model_output.xlsx`

## 與課堂作業的關係

老師要求的是 2027 / 2028 **年度 Pro Forma Financial Statements**。本專案的月模型只是幫助估計：

```text
monthly revenue forecast
        ↓
2027 / 2028 annual revenue
        ↓
Sales Growth assumption
        ↓
Table 3.2 assumptions
        ↓
2027 / 2028 Pro Forma
```

後續仍應加入季度/年度財報中的 COGS、Gross Margin、A/R、Inventory、A/P、PP&E、Debt、Tax 等 drivers。

## Data / License note

程式碼採 MIT License。  
`data/raw/` 中的第三方與公司原始文件**不因本 repo 的 MIT License 而重新授權**；其權利與使用條款依原始提供者規定。為避免誤用，所有 raw 檔都附來源與 hash manifest。
