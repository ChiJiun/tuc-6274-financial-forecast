from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MOPS_MONTHLY = RAW / "mops_monthly"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TICKER = "6274"
MODEL_START = pd.Timestamp("2013-01-01")
DEFAULT_FORECAST_END = pd.Timestamp("2028-12-01")
SELECTION_HORIZON_MONTHS = 12
SELECTION_ORIGIN_START = pd.Timestamp("2020-12-01")
SELECTION_MODEL_NAMES = [
    "SeasonalNaive",
    "HW_Damped_Mul",
    "HW_Damped_Add",
    "HW_Log_Damped_Add",
    "Ridge",
    "GradientBoosting",
    "RandomForest",
]


@dataclass
class BacktestResult:
    test_year: int
    model: str
    forecast_months: int
    mape_pct: float
    wape_pct: float
    total_error_pct: float
    actual_total_bn_twd: float
    forecast_total_bn_twd: float


def ensure_dirs():
    for d in [PROCESSED, RESULTS, REPORTS, FIGURES]:
        d.mkdir(parents=True, exist_ok=True)


def _decode_big5(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("big5", "cp950", "big5hkscs"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("big5", errors="replace")


def _normalize_col(col) -> str:
    if isinstance(col, tuple):
        parts = [str(x) for x in col if str(x) != "nan"]
        return " ".join(parts)
    return str(col)


def parse_mops_month(path: Path) -> dict | None:
    m = re.search(r"t21sc03_(\d+)_(\d+)_0\.html$", path.name)
    if not m:
        return None
    roc_year, month = int(m.group(1)), int(m.group(2))
    year = roc_year + 1911
    text = _decode_big5(path)

    try:
        tables = pd.read_html(StringIO(text))
    except Exception:
        return None

    for table in tables:
        if table.empty:
            continue
        table = table.copy()
        table.columns = [_normalize_col(c) for c in table.columns]
        stringified = table.astype(str)
        mask = stringified.apply(
            lambda row: row.str.strip().eq(TICKER).any(), axis=1
        )
        if not mask.any():
            continue

        row = table.loc[mask].iloc[0]
        cols = list(table.columns)

        name_col = next((c for c in cols if "公司名稱" in c), None)
        rev_col = next(
            (
                c
                for c in cols
                if ("當月營收" in c or "本月營收" in c)
                and "累計" not in c
                and "去年" not in c
                and "上月" not in c
            ),
            None,
        )

        def clean_num(x):
            if pd.isna(x):
                return np.nan
            s = re.sub(r"[^0-9.\-]", "", str(x))
            return float(s) if s not in {"", "-", "."} else np.nan

        if rev_col is not None:
            revenue = clean_num(row[rev_col])
            company_name = str(row[name_col]).strip() if name_col else "台燿"
        else:
            vals = [str(v).strip() for v in row.tolist()]
            try:
                idx = vals.index(TICKER)
            except ValueError:
                idx = next((i for i, v in enumerate(vals) if v == TICKER), -1)
            if idx < 0:
                continue
            company_name = vals[idx + 1] if idx + 1 < len(vals) else "台燿"
            revenue = np.nan
            for v in vals[idx + 2 :]:
                n = clean_num(v)
                if not np.isnan(n) and abs(n) >= 10000:
                    revenue = n
                    break

        if np.isnan(revenue):
            continue

        return {
            "date": pd.Timestamp(year=year, month=month, day=1),
            "revenue_k_twd": int(round(revenue)),
            "source_file": path.relative_to(ROOT).as_posix(),
            "company_name": company_name,
            "source_kind": "MOPS OTC historical monthly revenue HTML",
        }
    return None


def build_monthly_dataset() -> pd.DataFrame:
    records = []
    for p in sorted(MOPS_MONTHLY.glob("t21sc03_*_*_0.html")):
        rec = parse_mops_month(p)
        if rec:
            records.append(rec)

    if not records:
        raise RuntimeError("No MOPS monthly rows for 6274 could be parsed.")

    df = pd.DataFrame(records).sort_values("date")
    df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
    df["revenue_bn_twd"] = df["revenue_k_twd"] / 1_000_000.0
    df["mom"] = df["revenue_bn_twd"].pct_change()
    df["yoy"] = df["revenue_bn_twd"].pct_change(12)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["model_eligible"] = df["date"].ge(MODEL_START)

    df.to_csv(PROCESSED / "monthly_revenue.csv", index=False, encoding="utf-8-sig")
    return df


FEATURE_NAMES = [
    "log_lag1",
    "log_lag2",
    "log_lag3",
    "log_lag6",
    "log_lag12",
    "log_lag13",
    "log_lag24",
    "log_roll3",
    "log_roll6",
    "log_roll12",
    "yoy_log_lag1",
    "yoy_log_mean3",
    "yoy_log_mean6",
    "mom_log_lag1",
    "month_sin",
    "month_cos",
    "time_trend",
]


def level_features(vals, t: int, months: list[int]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    logs = np.log(arr)
    lag_list = [1, 2, 3, 6, 12, 13, 24]
    feats = [logs[t - lag] for lag in lag_list]
    feats += [
        logs[t - 3 : t].mean(),
        logs[t - 6 : t].mean(),
        logs[t - 12 : t].mean(),
    ]
    yoy_hist = [logs[j] - logs[j - 12] for j in range(max(12, t - 6), t)]
    feats += [
        yoy_hist[-1],
        float(np.mean(yoy_hist[-3:])),
        float(np.mean(yoy_hist[-6:])),
        logs[t - 1] - logs[t - 2],
    ]
    m = months[t]
    feats += [
        math.sin(2 * math.pi * m / 12),
        math.cos(2 * math.pi * m / 12),
        t / 100.0,
    ]
    return np.asarray(feats, dtype=float)


def make_feature_frame(model_df: pd.DataFrame) -> pd.DataFrame:
    y = model_df["revenue_bn_twd"].to_numpy(dtype=float)
    months = model_df["month"].astype(int).tolist()
    rows = []
    for t in range(24, len(y)):
        feat = level_features(y, t, months)
        row = {
            "date": model_df.iloc[t]["date"],
            "target_revenue_bn_twd": y[t],
            "target_log_revenue": math.log(y[t]),
        }
        row.update(dict(zip(FEATURE_NAMES, feat)))
        rows.append(row)
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(PROCESSED / "ml_features.csv", index=False, encoding="utf-8-sig")
    return feat_df


def train_level_model(vals, months: list[int], model_name: str):
    X, Y = [], []
    vals = np.asarray(vals, dtype=float)
    for t in range(24, len(vals)):
        X.append(level_features(vals, t, months))
        Y.append(math.log(vals[t]))
    X = np.asarray(X)
    Y = np.asarray(Y)

    if model_name == "Ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            random_state=42,
            n_estimators=150,
            max_depth=2,
            learning_rate=0.03,
            loss="huber",
        )
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            random_state=42,
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=3,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_name)

    model.fit(X, Y)
    return model


def recursive_ml_forecast(
    history: np.ndarray, first_history_month: int, horizon: int, model_name: str
):
    vals = list(map(float, history))
    months = [
        ((first_history_month - 1 + i) % 12) + 1
        for i in range(len(vals) + horizon)
    ]
    model = train_level_model(vals, months[: len(vals)], model_name)
    preds = []
    for _ in range(horizon):
        t = len(vals)
        x = level_features(vals, t, months)
        pred = float(np.exp(model.predict(x.reshape(1, -1))[0]))
        preds.append(pred)
        vals.append(pred)
    return np.asarray(preds)


def seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    vals = list(map(float, history))
    out = []
    for _ in range(horizon):
        pred = vals[-12]
        out.append(pred)
        vals.append(pred)
    return np.asarray(out)


def ets_forecast(history: np.ndarray, horizon: int, model_name: str) -> np.ndarray:
    history = np.asarray(history, dtype=float)

    if model_name == "HW_Damped_Mul":
        fit = ExponentialSmoothing(
            history,
            trend="add",
            damped_trend=True,
            seasonal="mul",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        return np.asarray(fit.forecast(horizon), dtype=float)

    if model_name == "HW_Damped_Add":
        fit = ExponentialSmoothing(
            history,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        return np.asarray(fit.forecast(horizon), dtype=float)

    if model_name == "HW_Log_Damped_Add":
        fit = ExponentialSmoothing(
            np.log(history),
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        return np.exp(np.asarray(fit.forecast(horizon), dtype=float))

    raise ValueError(model_name)


def hw_damped_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Backward-compatible alias for the previous production specification."""
    return ets_forecast(history, horizon, "HW_Damped_Mul")


def evaluate(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    mape = float(np.mean(np.abs((actual - pred) / actual)))
    wape = float(np.sum(np.abs(actual - pred)) / np.sum(actual))
    total_error = float((np.sum(pred) - np.sum(actual)) / np.sum(actual))
    return mape, wape, total_error


def backtest(model_df: pd.DataFrame, model_names: list[str] | None = None):
    """Annual-origin diagnostic/stress test; never used for production selection."""
    model_names = model_names or SELECTION_MODEL_NAMES
    y = model_df["revenue_bn_twd"].to_numpy(dtype=float)
    dates = model_df["date"].tolist()
    first_history_month = int(model_df.iloc[0]["month"])
    rows = []

    for test_year in range(2021, int(model_df["year"].max()) + 1):
        idx = [i for i, d in enumerate(dates) if d.year == test_year]
        if not idx:
            continue
        first = idx[0]
        horizon = len(idx)
        history = y[:first]
        actual = y[first : first + horizon]
        if len(history) < 36:
            continue

        preds = {
            name: forecast_model(name, history, first_history_month, horizon)
            for name in model_names
        }

        for name, pred in preds.items():
            mape, wape, total_error = evaluate(actual, pred)
            rows.append(
                BacktestResult(
                    test_year=test_year,
                    model=name,
                    forecast_months=horizon,
                    mape_pct=mape * 100,
                    wape_pct=wape * 100,
                    total_error_pct=total_error * 100,
                    actual_total_bn_twd=float(np.sum(actual)),
                    forecast_total_bn_twd=float(np.sum(pred)),
                ).__dict__
            )

    bt = pd.DataFrame(rows)
    bt["is_complete_12m"] = bt["forecast_months"].eq(12)
    bt["selection_eligible"] = False
    bt.to_csv(
        RESULTS / "walk_forward_backtest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    partial = bt.loc[bt["forecast_months"].lt(12)].copy()
    partial.to_csv(
        RESULTS / "partial_year_stress_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        bt.assign(abs_total_error=lambda x: x["total_error_pct"].abs())
        .groupby("model", as_index=False)
        .agg(
            mean_MAPE_pct=("mape_pct", "mean"),
            mean_WAPE_pct=("wape_pct", "mean"),
            mean_abs_annual_total_error_pct=("abs_total_error", "mean"),
        )
        .sort_values("mean_WAPE_pct")
        .reset_index(drop=True)
    )
    summary.to_csv(
        RESULTS / "annual_backtest_summary.csv", index=False, encoding="utf-8-sig"
    )
    return bt, summary


def rolling_selection_backtest(model_df: pd.DataFrame):
    """Equal-horizon quarterly rolling-origin validation used for production selection."""
    y = model_df["revenue_bn_twd"].to_numpy(dtype=float)
    dates = model_df["date"].reset_index(drop=True)
    first_history_month = int(model_df.iloc[0]["month"])
    rows = []

    for i, origin in enumerate(dates):
        if origin.month not in (3, 6, 9, 12):
            continue
        if origin < SELECTION_ORIGIN_START:
            continue

        end = i + 1
        if end + SELECTION_HORIZON_MONTHS > len(model_df):
            continue

        history = y[:end]
        actual = y[end : end + SELECTION_HORIZON_MONTHS]

        for name in SELECTION_MODEL_NAMES:
            pred = forecast_model(
                name,
                history,
                first_history_month,
                SELECTION_HORIZON_MONTHS,
            )
            mape, wape, total_error = evaluate(actual, pred)
            rows.append(
                {
                    "origin": origin,
                    "model": name,
                    "forecast_months": SELECTION_HORIZON_MONTHS,
                    "mape_pct": mape * 100,
                    "wape_pct": wape * 100,
                    "total_error_pct": total_error * 100,
                }
            )

    rolling = pd.DataFrame(rows)
    if rolling.empty:
        raise RuntimeError("No equal-horizon rolling-origin folds available.")

    rolling.to_csv(
        RESULTS / "model_selection_12m_quarterly.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        rolling.assign(abs_total_error=lambda x: x["total_error_pct"].abs())
        .groupby("model", as_index=False)
        .agg(
            mean_MAPE_pct=("mape_pct", "mean"),
            mean_WAPE_pct=("wape_pct", "mean"),
            median_WAPE_pct=("wape_pct", "median"),
            p75_WAPE_pct=("wape_pct", lambda x: x.quantile(0.75)),
            mean_abs_annual_total_error_pct=("abs_total_error", "mean"),
            folds=("origin", "count"),
        )
        .sort_values(
            [
                "mean_WAPE_pct",
                "median_WAPE_pct",
                "mean_abs_annual_total_error_pct",
                "model",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    summary.to_csv(
        RESULTS / "model_summary.csv", index=False, encoding="utf-8-sig"
    )
    return rolling, summary


def forecast_model(
    model_name: str,
    history: np.ndarray,
    first_history_month: int,
    horizon: int,
):
    if model_name == "SeasonalNaive":
        return seasonal_naive(history, horizon)
    if model_name in {"HW_Damped_Mul", "HW_Damped_Add", "HW_Log_Damped_Add"}:
        return ets_forecast(history, horizon, model_name)
    if model_name in {"Ridge", "GradientBoosting", "RandomForest"}:
        return recursive_ml_forecast(
            history, first_history_month, horizon, model_name
        )
    raise ValueError(model_name)


def build_forecast(
    model_df: pd.DataFrame,
    summary: pd.DataFrame,
    forecast_end: pd.Timestamp = DEFAULT_FORECAST_END,
):
    selected = str(summary.iloc[0]["model"])
    last_date = model_df["date"].max()
    if forecast_end <= last_date:
        raise ValueError("forecast_end must be after the latest observed month")
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), forecast_end, freq="MS"
    )
    horizon = len(future_dates)
    history = model_df["revenue_bn_twd"].to_numpy(dtype=float)
    first_history_month = int(model_df.iloc[0]["month"])

    model_names = SELECTION_MODEL_NAMES
    out = pd.DataFrame({"date": future_dates})
    for name in model_names:
        out[name] = forecast_model(
            name, history, first_history_month, horizon
        )
    out["selected_model"] = selected
    out["selected_revenue_bn_twd"] = out[selected]
    out.to_csv(
        RESULTS / "monthly_forecast.csv", index=False, encoding="utf-8-sig"
    )

    actual_annual = (
        model_df.groupby("year", as_index=False)["revenue_bn_twd"]
        .sum()
        .rename(columns={"revenue_bn_twd": "actual_revenue_bn_twd"})
    )
    fc_annual = (
        out.assign(year=out["date"].dt.year)
        .groupby("year", as_index=False)["selected_revenue_bn_twd"]
        .sum()
    )

    current_year = int(model_df["year"].max())
    actual_ytd = float(
        model_df.loc[
            model_df["year"].eq(current_year), "revenue_bn_twd"
        ].sum()
    )
    if (fc_annual["year"] == current_year).any():
        fc_annual.loc[
            fc_annual["year"].eq(current_year),
            "selected_revenue_bn_twd",
        ] += actual_ytd

    annual = fc_annual.copy()
    annual["yoy_growth_pct"] = np.nan
    prior_actual = actual_annual.loc[
        actual_annual["year"].eq(current_year - 1),
        "actual_revenue_bn_twd",
    ]
    prior = float(prior_actual.iloc[0]) if len(prior_actual) else np.nan
    for i in range(len(annual)):
        rev = float(annual.iloc[i]["selected_revenue_bn_twd"])
        if not np.isnan(prior):
            annual.loc[i, "yoy_growth_pct"] = (rev / prior - 1) * 100
        prior = rev

    annual["selected_model"] = selected
    annual.to_csv(
        RESULTS / "annual_forecast.csv", index=False, encoding="utf-8-sig"
    )
    return out, annual, selected


def plot_outputs(
    model_df: pd.DataFrame,
    monthly_fc: pd.DataFrame,
    summary: pd.DataFrame,
    selected: str,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(model_df["date"], model_df["revenue_bn_twd"], label="Actual")
    ax.plot(
        monthly_fc["date"],
        monthly_fc["selected_revenue_bn_twd"],
        label=f"{selected} forecast",
    )
    ax.set_title("TUC (6274) Monthly Revenue: History and Forecast")
    ax.set_ylabel("Revenue (NT$ bn)")
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "history_and_forecast.png", dpi=160)
    plt.close(fig)

    ordered = summary.sort_values("mean_WAPE_pct")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(ordered["model"], ordered["mean_WAPE_pct"])
    ax.set_title("12-Month Rolling-Origin Selection: Mean WAPE")
    ax.set_ylabel("Mean WAPE (%) - lower is better")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_backtest_wape.png", dpi=160)
    plt.close(fig)


def write_excel(
    monthly: pd.DataFrame,
    features: pd.DataFrame,
    backtest_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    summary: pd.DataFrame,
    monthly_fc: pd.DataFrame,
    annual_fc: pd.DataFrame,
):
    path = REPORTS / "model_output.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        monthly.to_excel(writer, sheet_name="Monthly_RawParsed", index=False)
        features.to_excel(writer, sheet_name="Features", index=False)
        backtest_df.to_excel(writer, sheet_name="Annual_Backtest", index=False)
        selection_df.to_excel(writer, sheet_name="Selection_12M_Rolling", index=False)
        summary.to_excel(writer, sheet_name="Model_Summary", index=False)
        monthly_fc.to_excel(writer, sheet_name="Forecast_Monthly", index=False)
        annual_fc.to_excel(writer, sheet_name="Forecast_Annual", index=False)


def write_methodology(
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    annual_fc: pd.DataFrame,
    selected: str,
):
    first = monthly.loc[monthly["model_eligible"], "date"].min()
    last = monthly.loc[monthly["model_eligible"], "date"].max()
    best = summary.iloc[0]
    lines = [
        "# 台燿科技（6274）Revenue Forecast Methodology",
        "",
        f"- Model window: {first:%Y-%m} to {last:%Y-%m}",
        f"- Selected model: **{selected}**",
        f"- Mean out-of-sample WAPE: **{best['mean_WAPE_pct']:.2f}%**",
        "",
        "## Validation",
        "",
        "Production model selection uses equal-horizon 12-month forecasts from quarterly",
        "rolling origins beginning 2020-12. Selection metric: mean out-of-sample WAPE.",
        "Partial-year 2026 performance is retained only in the annual-origin stress test.",
        "",
        "## Annual forecast",
        "",
        "| Year | Revenue (NT$ bn) | YoY |",
        "|---:|---:|---:|",
    ]
    for _, r in annual_fc.iterrows():
        lines.append(
            f"| {int(r['year'])} | {r['selected_revenue_bn_twd']:.2f} | "
            f"{r['yoy_growth_pct']:.1f}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This model is a quantitative anchor for the Sales Growth assumption.",
        "It does not replace capacity, product-mix, ASP, raw-material, FX, or industry analysis.",
        "",
        "## Raw-data rule",
        "",
        "All model inputs are rebuilt from data/raw. Raw files are not manually edited.",
    ]
    (REPORTS / "methodology.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_forecast_end(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m").date().replace(day=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--forecast-end must be YYYY-MM") from exc


def write_run_metadata(
    model_df: pd.DataFrame,
    annual_fc: pd.DataFrame,
    selection_summary: pd.DataFrame,
    selected: str,
    forecast_end: pd.Timestamp,
):
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": TICKER,
        "model_start": MODEL_START.strftime("%Y-%m"),
        "actual_data_through": model_df["date"].max().strftime("%Y-%m"),
        "observations": int(len(model_df)),
        "forecast_end": forecast_end.strftime("%Y-%m"),
        "selected_model": selected,
        "selection_protocol": "quarterly rolling-origin, fixed 12-month horizon",
        "selection_metric": "mean_WAPE_pct",
        "selection_tiebreakers": [
            "median_WAPE_pct",
            "mean_abs_annual_total_error_pct",
            "model",
        ],
        "selection_origin_start": SELECTION_ORIGIN_START.strftime("%Y-%m"),
        "selection_horizon_months": SELECTION_HORIZON_MONTHS,
        "selection_folds": int(selection_summary.iloc[0]["folds"]),
        "selection_score": float(selection_summary.iloc[0]["mean_WAPE_pct"]),
        "forecast_years": [int(y) for y in annual_fc["year"].tolist()],
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast-end",
        type=parse_forecast_end,
        default=DEFAULT_FORECAST_END,
        help="Last forecast month in YYYY-MM format (default: 2028-12).",
    )
    args = parser.parse_args()

    ensure_dirs()
    monthly_all = build_monthly_dataset()
    model_df = (
        monthly_all.loc[monthly_all["model_eligible"]]
        .copy()
        .reset_index(drop=True)
    )
    expected = pd.date_range(
        model_df["date"].min(), model_df["date"].max(), freq="MS"
    )
    missing = expected.difference(pd.DatetimeIndex(model_df["date"]))
    if len(missing):
        raise RuntimeError(
            "Missing monthly observations in model window: "
            + ", ".join(d.strftime("%Y-%m") for d in missing[:20])
        )

    features = make_feature_frame(model_df)
    selection_bt, summary = rolling_selection_backtest(model_df)
    selected_for_stress = str(summary.iloc[0]["model"])
    stress_models = [selected_for_stress]
    if selected_for_stress != "SeasonalNaive":
        stress_models.append("SeasonalNaive")
    annual_bt, _annual_summary = backtest(model_df, model_names=stress_models)
    monthly_fc, annual_fc, selected = build_forecast(
        model_df, summary, forecast_end=args.forecast_end
    )
    plot_outputs(model_df, monthly_fc, summary, selected)
    write_excel(
        monthly_all,
        features,
        annual_bt,
        selection_bt,
        summary,
        monthly_fc,
        annual_fc,
    )
    write_methodology(monthly_all, summary, annual_fc, selected)
    write_run_metadata(
        model_df,
        annual_fc,
        summary,
        selected,
        args.forecast_end,
    )

    print("\nSelected model:", selected)
    print(summary.to_string(index=False))
    print("\nAnnual forecast:")
    print(annual_fc.to_string(index=False))


if __name__ == "__main__":
    main()
