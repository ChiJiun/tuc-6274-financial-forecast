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
│  │  ├─ exogenous/          # 同業月營收與 CBC FX snapshots
│  │  ├─ investor_presentations/
│  │  └─ manifest.csv
│  └─ processed/             # 清洗後資料
├─ src/
│  ├─ fetch_raw.py
│  ├─ build_assignment_financials.py
│  ├─ build_quarterly_financials.py
│  ├─ build_exogenous_features.py
│  ├─ build_exogenous_phase2.py
│  ├─ benchmark_exogenous_phase2.py
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
py -3.12 -m venv .venv
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

## 外生財務特徵實驗（Issue #14，第一階段）

此實驗**不修改正式營收預測模型**。新增 `src/build_exogenous_features.py` 與
`src/benchmark_exogenous.py`，檢驗「月營收」與「月營收＋已公布的年度財務資訊」
在相同 rolling-origin 測試期間的差異。

資料：現有 MOPS 月營收（2013/01–2026/09）與正式重建的 MOPS 年度合併財報
（2016–2025）。財務特徵包括毛利率、營業利益率、存貨／總資產、
PP&E／總資產、應收帳款／總資產。尚**未**加入同業月營收、匯率及產業資料。

為避免將期末資料當成當時已公布資訊，財報採用**下一年 7 月 1 日**
作為保守的**可用日期代理值**；月營收預測起點以該月結束後下一個月 16 日
作為代理值。這些日期**不是逐筆查證的歷史公告日期**；要宣稱嚴格歷史
point-in-time 評估仍需補上原始公告時間及修正版本的檢核。
未來月份的財務數值不會餵進模型。

使用相同季度起點、固定預測長度（1、3、6、12 個月）與已觀察的目標：
早期（至 2023/09）作為 model-selection validation，後期（自 2023/12）
為此新實驗保留的 holdout。但舊有 Holt-Winters 的選模早已用過全部 20 個
12-month folds，**不可把該 holdout 視為原 Holt-Winters 的全新獨立測試集**。

執行（需先產生既有 `data/processed/` 輸入）：

```powershell
python src/build_exogenous_features.py
python src/benchmark_exogenous.py
pytest -q
```

產出：
- `data/processed/exogenous_financial_releases.csv` — 年度財報來源及可用日期**代理值**
- `data/processed/exogenous_financial_features.csv` — 每月 point-in-time snapshots
- `results/exogenous_fold_predictions.csv` — 每個季度起點／目標月份的預測和實際值
- `results/exogenous_model_scores.csv` — 各模型、特徵組合、預測長度的表現
- `reports/exogenous_financial_ablation.md` — 比較結果與限制

要進一步加入同業、匯率等外部資料，應為每個來源紀錄其實際發布時間、
歷史可取得版本，並以相同規則再次驗證。無法在起點得知的未來外部變數，
不可使用事後實際值。

## Issue #14：台燿自身財報資訊是否改善營收預測

核心比較只使用 **台燿自己的資料**：

- Revenue-only
- Revenue + TUC quarterly financials

財報範圍為 2016Q1–2026Q2，包含 Gross Margin、Operating Margin、Inventory、A/R、
PP&E、Debt、CFO、CAPEX，以及 PP&E / Inventory / A/R YoY growth。

資料截止到 2026-09；2026Q3 月營收已涵蓋。2026Q3 財報尚未公開，因此最新財務狀態
是 2026Q2，不做 forward-fill 或自行估造 Q3 財報。

12-month comparable-fold 結果：

| Stage | Model | Revenue-only WAPE | + TUC financials WAPE |
|---|---|---:|---:|
| Validation | GradientBoosting | 18.26% | **16.30%** |
| Validation | Ridge | 19.58% | **16.68%** |
| Later holdout | GradientBoosting | 29.58% | 29.49% |
| Later holdout | Ridge | **31.60%** | 31.66% |
| Later holdout | HW_Damped_Add | **21.64%** | — |

財報資訊會改善部分 ML model 的 validation 表現，但在 later holdout 沒有穩定改善，
也沒有打敗 HW_Damped_Add，因此 **production model 維持 HW_Damped_Add**。

專門回答這個問題的產出：

- `results/tuc_financial_only_comparison.csv`
- `reports/tuc_financial_only_comparison.md`

Phase 2 另外保留台光電 2383、聯茂 6213 與 CBC NTD/USD 作為額外 robustness experiment；
這些外部資料不屬於上述核心比較。

主要 Phase 2 產出：

- `data/processed/TUC_6274_quarterly_financial_features_2016_2026Q2.csv`
- `data/processed/exogenous_phase2_features.csv`
- `results/exogenous_phase2_model_scores.csv`
- `reports/exogenous_phase2_report.md`
- `reports/data_completeness_2026Q3.md`

重建與產生比較表：

```powershell
python src/build_quarterly_financials.py
python src/build_exogenous_phase2.py
python src/report_tuc_financial_only.py
```

## TEJ 匯入

TEJ 原始匯出檔屬授權資料，預設放在 `data/private/tej/`，該目錄不進 Git。

支援 CSV / XLSX / XLS：

```powershell
python src/import_tej_tuc.py --input "data/private/tej/TUC_6274.xlsx"
```

若 TEJ 欄名無法自動辨識，可用 JSON 指定欄位對應：

```powershell
python src/import_tej_tuc.py `
  --input "data/private/tej/TUC_6274.xlsx" `
  --column-map "config/tej_tuc_column_map.json"
```

優先匯出欄位：股票代碼、財報期間、實際公告日、Revenue、Gross Margin、
Operating Margin、Inventory、A/R、PP&E、CFO、CAPEX、Cash、Assets、Liabilities。
其中實際公告日最重要，可取代目前 backtest 的 conservative availability proxy。

## License

程式碼採 MIT License。

`data/raw/` 中的第三方與公司文件仍依原始來源之授權與使用條款。
