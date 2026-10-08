"""Point-in-time annual financial features for Issue #14.

This is deliberately a conservative publication-date *proxy*, NOT verified
historical announcement metadata. Do not treat fiscal period-end as release date.
The project currently only has annual financial statement extracts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "processed" / "TUC_6274_assignment_historical_2016_2025.csv"
MONTHLY = ROOT / "data" / "processed" / "monthly_revenue.csv"
OUTPUT = ROOT / "data" / "processed" / "exogenous_financial_features.csv"
RELEASES_OUTPUT = ROOT / "data" / "processed" / "exogenous_financial_releases.csv"

# Annual statements are assumed available only on July 1 of the next year.
# This is a documented, deliberately delayed availability PROXY, not an actual
# MOPS publication timestamp. Replace with source-specific announcement dates
# before any claim of strictly point-in-time historical availability.
FINANCIAL_FEATURES = (
    "fin_gross_margin_pct",
    "fin_operating_margin_pct",
    "fin_inventory_asset_ratio",
    "fin_ppe_asset_ratio",
    "fin_ar_asset_ratio",
)
REVENUE_FEATURES = (
    "log_last_revenue",
    "log_mean3",
    "log_mean12",
    "log_lag12_revenue",
    "log_yoy_growth",
    "time_trend",
)


def financial_releases(history: pd.DataFrame) -> pd.DataFrame:
    """Build a versioned annual *proxy* calendar from reconstructed MOPS figures."""
    required = {
        "year", "gross_margin_pct", "operating_margin_pct",
        "inventory_ntd_m", "ppe_ntd_m", "accounts_receivable_ntd_m",
        "total_assets_ntd_m",
    }
    if not required.issubset(history.columns):
        raise ValueError(f"Missing annual financial fields: {sorted(required - set(history.columns))}")
    raw = history.copy().sort_values("year")
    if raw["year"].duplicated().any():
        raise ValueError("Duplicate annual financial years")
    if (raw["total_assets_ntd_m"] <= 0).any():
        raise ValueError("Assets must be positive to compute ratios")

    years = raw["year"].astype(int)
    result = pd.DataFrame({
        "financial_year": years,
        "financial_period_end": pd.to_datetime(years.astype(str) + "-12-31"),
        "financial_available_at": pd.to_datetime((years + 1).astype(str) + "-07-01"),
        "financial_availability_policy": "next_year_july_1_proxy_not_actual_release",
        "financial_source": "data/processed/TUC_6274_assignment_historical_2016_2025.csv",
        "financial_raw_file": years.map(lambda year: f"data/raw/mops_financials/6274_{year}_Q4_C.html"),
        "financial_source_url": years.map(lambda year: (
            "https://mopsov.twse.com.tw/server-java/t164sb01"
            f"?step=1&CO_ID=6274&SYEAR={year}&SSEASON=4&REPORT_ID=C"
        )),
        "fin_gross_margin_pct": raw["gross_margin_pct"].to_numpy(),
        "fin_operating_margin_pct": raw["operating_margin_pct"].to_numpy(),
        "fin_inventory_asset_ratio": raw["inventory_ntd_m"].to_numpy() / raw["total_assets_ntd_m"].to_numpy(),
        "fin_ppe_asset_ratio": raw["ppe_ntd_m"].to_numpy() / raw["total_assets_ntd_m"].to_numpy(),
        "fin_ar_asset_ratio": raw["accounts_receivable_ntd_m"].to_numpy() / raw["total_assets_ntd_m"].to_numpy(),
    })
    if not np.isfinite(result[list(FINANCIAL_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite annual financial features")
    return result.sort_values("financial_available_at").reset_index(drop=True)


def build_features(monthly: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Use only lagged revenue and annual figures presumed published by origin."""
    required = {"date", "revenue_bn_twd"}
    if not required.issubset(monthly.columns):
        raise ValueError("Missing monthly date/revenue")
    data = monthly.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    if data["date"].duplicated().any():
        raise ValueError("Duplicate monthly dates")
    expected = pd.date_range(data["date"].min(), data["date"].max(), freq="MS")
    if not pd.DatetimeIndex(data["date"]).equals(expected):
        raise ValueError("Monthly data must be contiguous")
    if (data["revenue_bn_twd"] <= 0).any():
        raise ValueError("Revenue must be positive")

    # Forecast origin = month whose revenue has become available. We use the
    # 16th of the following month as a conservative calendar proxy, NOT a
    # verified month-specific MOPS publication timestamp.
    data["origin"] = data["date"]
    data["as_of"] = data["origin"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=15)
    logs = np.log(data["revenue_bn_twd"].to_numpy(dtype=float))
    data["log_last_revenue"] = logs
    data["log_mean3"] = pd.Series(logs).rolling(3).mean()
    data["log_mean12"] = pd.Series(logs).rolling(12).mean()
    data["log_lag12_revenue"] = pd.Series(logs).shift(12)
    data["log_yoy_growth"] = data["log_last_revenue"] - data["log_lag12_revenue"]
    data["time_trend"] = np.arange(len(data)) / 100.0

    releases = financial_releases(history)
    result = pd.merge_asof(
        data.sort_values("as_of"), releases,
        left_on="as_of", right_on="financial_available_at",
        direction="backward", allow_exact_matches=True,
    )
    result = result.sort_values("origin").reset_index(drop=True)
    known = result["financial_available_at"].notna()
    if (result.loc[known, "financial_available_at"] > result.loc[known, "as_of"]).any():
        raise ValueError("Point-in-time leak: statement not yet available")
    if (result.loc[known, "financial_period_end"] > result.loc[known, "as_of"]).any():
        raise ValueError("Point-in-time leak: fiscal period after as-of date")
    return result[[
        "origin", "as_of", "revenue_bn_twd",
        *REVENUE_FEATURES,
        "financial_year", "financial_period_end", "financial_available_at",
        "financial_availability_policy", "financial_source",
        "financial_raw_file", "financial_source_url",
        *FINANCIAL_FEATURES,
    ]]


def load_features() -> pd.DataFrame:
    monthly = pd.read_csv(MONTHLY, parse_dates=["date"])
    # The raw archive predates the model window and may contain older gaps.
    # Require continuity only within the already-validated 2013+ model window.
    monthly = monthly.loc[monthly["date"].ge("2013-01-01")].copy()
    history = pd.read_csv(HISTORY)
    return build_features(monthly, history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    features = load_features()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, encoding="utf-8-sig")
    releases = financial_releases(pd.read_csv(HISTORY))
    RELEASES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    releases.to_csv(RELEASES_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved {len(features)} monthly feature snapshots to {args.output}")
    print(f"Saved {len(releases)} annual release records to {RELEASES_OUTPUT}")
    print("WARNING: availability dates are conservative proxies, not verified announcement dates.")


if __name__ == "__main__":
    main()
