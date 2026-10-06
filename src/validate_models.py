from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402


def ets_forecast(history: np.ndarray, horizon: int, model_name: str) -> np.ndarray:
    history = np.asarray(history, dtype=float)

    if model_name == "HW_Damped_Mul":
        return pipeline.hw_damped_forecast(history, horizon)

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


def forecast(history: np.ndarray, first_month: int, horizon: int, model_name: str):
    if model_name == "SeasonalNaive":
        return pipeline.seasonal_naive(history, horizon)
    if model_name in {"HW_Damped_Mul", "HW_Damped_Add", "HW_Log_Damped_Add"}:
        return ets_forecast(history, horizon, model_name)
    return pipeline.recursive_ml_forecast(history, first_month, horizon, model_name)


def quarterly_rolling_validation(model_df: pd.DataFrame) -> pd.DataFrame:
    y = model_df["revenue_bn_twd"].to_numpy(dtype=float)
    dates = model_df["date"].reset_index(drop=True)
    first_month = int(model_df.iloc[0]["month"])

    model_names = [
        "SeasonalNaive",
        "HW_Damped_Mul",
        "HW_Damped_Add",
        "HW_Log_Damped_Add",
        "Ridge",
        "GradientBoosting",
        "RandomForest",
    ]

    rows = []
    for i, date in enumerate(dates):
        if (
            date.month not in (3, 6, 9, 12)
            or date < pd.Timestamp("2020-12-01")
            or i + 12 >= len(model_df)
        ):
            continue

        end = i + 1
        history = y[:end]
        actual = y[end : end + 12]

        for name in model_names:
            pred = forecast(history, first_month, 12, name)
            mape, wape, total_error = pipeline.evaluate(actual, pred)
            rows.append(
                {
                    "origin": date,
                    "model": name,
                    "MAPE_pct": mape * 100,
                    "WAPE_pct": wape * 100,
                    "annual_total_error_pct": total_error * 100,
                }
            )

    return pd.DataFrame(rows)


def ets_sensitivity(model_df: pd.DataFrame) -> pd.DataFrame:
    y = model_df["revenue_bn_twd"].to_numpy(dtype=float)
    horizon = 27
    specs = [
        "HW_Damped_Mul",
        "HW_Damped_Add",
        "HW_Log_Damped_Add",
    ]

    rows = []
    for name in specs:
        pred = ets_forecast(y, horizon, name)
        rows.append(
            {
                "model": name,
                "2026Q4_revenue_bn_twd": pred[:3].sum(),
                "2027_revenue_bn_twd": pred[3:15].sum(),
                "2028_revenue_bn_twd": pred[15:27].sum(),
            }
        )
    return pd.DataFrame(rows)


def main():
    pipeline.ensure_dirs()
    monthly = pd.read_csv(
        pipeline.PROCESSED / "monthly_revenue.csv",
        parse_dates=["date"],
    )
    model_df = monthly.loc[monthly["date"].ge(pipeline.MODEL_START)].copy()
    model_df = model_df.reset_index(drop=True)

    rolling = quarterly_rolling_validation(model_df)
    rolling.to_csv(
        pipeline.RESULTS / "robustness_12m_quarterly.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity = ets_sensitivity(model_df)
    sensitivity.to_csv(
        pipeline.RESULTS / "ets_forecast_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        rolling.assign(
            abs_total_error=lambda x: x["annual_total_error_pct"].abs()
        )
        .groupby("model", as_index=False)
        .agg(
            mean_WAPE_pct=("WAPE_pct", "mean"),
            median_WAPE_pct=("WAPE_pct", "median"),
            p75_WAPE_pct=("WAPE_pct", lambda x: x.quantile(0.75)),
            mean_abs_annual_total_error_pct=("abs_total_error", "mean"),
        )
        .sort_values("mean_WAPE_pct")
    )

    print(summary.to_string(index=False))
    print()
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
