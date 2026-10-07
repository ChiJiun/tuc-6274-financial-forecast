from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402


def ets_sensitivity(model_df: pd.DataFrame) -> pd.DataFrame:
    history = model_df["revenue_bn_twd"].to_numpy(dtype=float)
    horizon = 27
    specs = [
        "HW_Damped_Mul",
        "HW_Damped_Add",
        "HW_Log_Damped_Add",
    ]

    rows = []
    for name in specs:
        pred = pipeline.ets_forecast(history, horizon, name)
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

    selection_path = pipeline.RESULTS / "model_selection_12m_quarterly.csv"
    summary_path = pipeline.RESULTS / "model_summary.csv"
    if selection_path.exists() and summary_path.exists():
        rolling = pd.read_csv(selection_path, parse_dates=["origin"])
        summary = pd.read_csv(summary_path)
    else:
        rolling, summary = pipeline.rolling_selection_backtest(model_df)

    # Compatibility output retained for the detailed validation report.
    compatibility = rolling.rename(
        columns={
            "mape_pct": "MAPE_pct",
            "wape_pct": "WAPE_pct",
            "total_error_pct": "annual_total_error_pct",
        }
    )
    compatibility = compatibility[
        ["origin", "model", "MAPE_pct", "WAPE_pct", "annual_total_error_pct"]
    ]
    compatibility.to_csv(
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

    print(summary.to_string(index=False))
    print()
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
