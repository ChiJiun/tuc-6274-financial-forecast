from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_exogenous_phase2 as phase2
import pipeline

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

FEATURES_PATH = PROCESSED / "exogenous_phase2_features.csv"
PREDICTIONS_PATH = RESULTS / "exogenous_phase2_fold_predictions.csv"
SCORES_PATH = RESULTS / "exogenous_phase2_model_scores.csv"
CURRENT_FORECAST_PATH = RESULTS / "exogenous_phase2_current_forecast.csv"
REPORT_PATH = REPORTS / "exogenous_phase2_report.md"

FIRST_ORIGIN = pd.Timestamp("2020-12-01")
VALIDATION_LAST_ORIGIN = pd.Timestamp("2023-09-01")
LATEST_ACTUAL_ORIGIN = pd.Timestamp("2026-09-01")
HORIZONS = (1, 3, 6, 12, 24)
PRIMARY_HORIZON = 12
CURRENT_FORECAST_HORIZON = 15

REVENUE_FEATURES = (
    "log_last_revenue",
    "log_mean3",
    "log_mean12",
    "log_lag12_revenue",
    "log_yoy_growth",
    "time_trend",
)

FEATURE_GROUPS = {
    "revenue_only": REVENUE_FEATURES,
    "quarterly_expanded": (
        REVENUE_FEATURES
        + phase2.FINANCIAL_FEATURES_CORE
        + phase2.FINANCIAL_FEATURES_EXPANSION
    ),
    "peer_fx": REVENUE_FEATURES + phase2.PEER_FX_FEATURES,
    "all": (
        REVENUE_FEATURES
        + phase2.FINANCIAL_FEATURES_CORE
        + phase2.FINANCIAL_FEATURES_EXPANSION
        + phase2.PEER_FX_FEATURES
    ),
}

METHODS = ("Ridge", "GradientBoosting")


def load_features() -> pd.DataFrame:
    data = pd.read_csv(
        FEATURES_PATH,
        parse_dates=[
            "origin",
            "as_of",
            "peer_available_at",
            "fx_available_at",
            "financial_period_end",
            "financial_available_at",
        ],
    ).sort_values("origin").reset_index(drop=True)

    revenue = data["revenue_bn_twd"].astype(float)
    logs = np.log(revenue)
    data["log_last_revenue"] = logs
    data["log_mean3"] = logs.rolling(3).mean()
    data["log_mean12"] = logs.rolling(12).mean()
    data["log_lag12_revenue"] = logs.shift(12)
    data["log_yoy_growth"] = data["log_last_revenue"] - data["log_lag12_revenue"]
    data["time_trend"] = np.arange(len(data), dtype=float) / 100.0

    common_fields = FEATURE_GROUPS["all"]
    common_array = data[list(common_fields)].to_numpy(dtype=float)
    data["common_eligible"] = np.isfinite(common_array).all(axis=1)

    latest = data.iloc[-1]
    if latest["origin"] != LATEST_ACTUAL_ORIGIN:
        raise RuntimeError(
            f"Expected latest actual {LATEST_ACTUAL_ORIGIN:%Y-%m}, "
            f"got {latest['origin']:%Y-%m}"
        )
    if latest["financial_period"] != "2026Q2":
        raise RuntimeError(
            "Latest phase-2 financial state must be 2026Q2 because 2026Q3 "
            "financials were not public as of 2026-10-10"
        )
    return data


def make_model(method: str):
    if method == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if method == "GradientBoosting":
        return GradientBoostingRegressor(
            random_state=42,
            n_estimators=150,
            max_depth=2,
            learning_rate=0.03,
            loss="huber",
        )
    raise ValueError(method)


def vector(row: pd.Series, lead: int, group: str) -> np.ndarray:
    origin = pd.Timestamp(row["origin"])
    target_month = ((origin.month - 1 + lead) % 12) + 1
    values = [float(row[field]) for field in FEATURE_GROUPS[group]]
    values.extend(
        [
            float(lead) / 12.0,
            math.sin(2 * math.pi * target_month / 12),
            math.cos(2 * math.pi * target_month / 12),
        ]
    )
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(
            f"Non-finite phase-2 vector at origin={origin:%Y-%m}, "
            f"lead={lead}, group={group}"
        )
    return result


def training_set(
    snapshots: pd.DataFrame,
    revenues: np.ndarray,
    eval_idx: int,
    group: str,
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    x: list[np.ndarray] = []
    y: list[float] = []

    # Use the exact same common availability window for every feature group.
    for j in range(12, eval_idx):
        if not bool(snapshots.iloc[j]["common_eligible"]):
            continue
        row = snapshots.iloc[j]
        for lead in range(1, min(max_horizon, eval_idx - j) + 1):
            label_idx = j + lead
            if label_idx > eval_idx:
                raise AssertionError("Future target leaked into phase-2 training")
            x.append(vector(row, lead, group))
            y.append(math.log(float(revenues[label_idx])))

    if len(x) < 40:
        raise ValueError(
            f"Insufficient training examples for {group}/h={max_horizon}: {len(x)}"
        )
    return np.asarray(x), np.asarray(y)


def predict_direct(
    snapshots: pd.DataFrame,
    revenues: np.ndarray,
    eval_idx: int,
    horizon: int,
    group: str,
    method: str,
) -> np.ndarray:
    x_train, y_train = training_set(
        snapshots,
        revenues,
        eval_idx,
        group,
        horizon,
    )
    model = make_model(method)
    model.fit(x_train, y_train)
    row = snapshots.iloc[eval_idx]
    x_future = np.vstack(
        [vector(row, lead, group) for lead in range(1, horizon + 1)]
    )
    return np.exp(model.predict(x_future))


def eligible_origins(
    snapshots: pd.DataFrame,
    horizon: int,
) -> list[int]:
    dates = snapshots["origin"].reset_index(drop=True)
    out = []
    for i, origin in enumerate(dates):
        if origin < FIRST_ORIGIN or origin.month not in (3, 6, 9, 12):
            continue
        if i + horizon >= len(snapshots):
            continue
        if not bool(snapshots.iloc[i]["common_eligible"]):
            continue
        out.append(i)
    return out


def benchmark_horizon(
    snapshots: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    revenues = snapshots["revenue_bn_twd"].to_numpy(dtype=float)
    dates = snapshots["origin"].reset_index(drop=True)
    rows = []

    for i in eligible_origins(snapshots, horizon):
        origin = dates.iloc[i]
        stage = (
            "validation"
            if origin <= VALIDATION_LAST_ORIGIN
            else "holdout"
        )
        actual = revenues[i + 1 : i + 1 + horizon]

        candidates = {
            "HW_Damped_Add__revenue_only": pipeline.ets_forecast(
                revenues[: i + 1],
                horizon,
                "HW_Damped_Add",
            )
        }
        for group in FEATURE_GROUPS:
            for method in METHODS:
                key = f"{method}__{group}"
                candidates[key] = predict_direct(
                    snapshots,
                    revenues,
                    i,
                    horizon,
                    group,
                    method,
                )

        for key, predictions in candidates.items():
            model_name, feature_group = key.split("__", 1)
            for lead, (y_true, y_pred) in enumerate(
                zip(actual, predictions),
                start=1,
            ):
                if not np.isfinite(y_pred) or y_pred <= 0:
                    raise RuntimeError(
                        f"Invalid prediction for {key} {origin:%Y-%m} lead={lead}"
                    )
                rows.append(
                    {
                        "origin": origin.strftime("%Y-%m"),
                        "stage": stage,
                        "horizon_months": horizon,
                        "model": model_name,
                        "features": feature_group,
                        "lead_month": lead,
                        "target_month": dates.iloc[i + lead].strftime("%Y-%m"),
                        "actual_bn_twd": float(y_true),
                        "predicted_bn_twd": float(y_pred),
                    }
                )
        print(
            f"h={horizon:>2} {origin:%Y-%m} "
            f"{stage} candidates={len(candidates)}"
        )
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    fold_keys = [
        "stage",
        "horizon_months",
        "model",
        "features",
        "origin",
    ]

    def fold_metrics(group: pd.DataFrame) -> pd.Series:
        actual = group["actual_bn_twd"].to_numpy(dtype=float)
        pred = group["predicted_bn_twd"].to_numpy(dtype=float)
        denom = actual.sum()
        return pd.Series(
            {
                "wape_pct": np.abs(pred - actual).sum() / denom * 100,
                "mae_bn_twd": np.abs(pred - actual).mean(),
                "bias_pct": (pred.sum() / denom - 1.0) * 100,
                "abs_total_error_pct": abs(pred.sum() / denom - 1.0) * 100,
            }
        )

    folds = (
        predictions.groupby(fold_keys, sort=True)
        .apply(fold_metrics, include_groups=False)
        .reset_index()
    )
    scores = (
        folds.groupby(
            ["stage", "horizon_months", "model", "features"],
            as_index=False,
        )
        .agg(
            folds=("origin", "count"),
            mean_WAPE_pct=("wape_pct", "mean"),
            median_WAPE_pct=("wape_pct", "median"),
            mean_MAE_bn_twd=("mae_bn_twd", "mean"),
            mean_bias_pct=("bias_pct", "mean"),
            mean_abs_total_error_pct=("abs_total_error_pct", "mean"),
        )
        .sort_values(
            [
                "horizon_months",
                "stage",
                "mean_WAPE_pct",
                "model",
                "features",
            ]
        )
        .reset_index(drop=True)
    )
    return scores


def best_validation_candidate(scores: pd.DataFrame) -> pd.Series:
    candidates = scores.loc[
        scores["stage"].eq("validation")
        & scores["horizon_months"].eq(PRIMARY_HORIZON)
    ].sort_values(
        ["mean_WAPE_pct", "median_WAPE_pct", "model", "features"]
    )
    if candidates.empty:
        raise RuntimeError("No 12-month validation candidates")
    return candidates.iloc[0]


def current_forecasts(
    snapshots: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    revenues = snapshots["revenue_bn_twd"].to_numpy(dtype=float)
    eval_idx = len(snapshots) - 1
    origin = snapshots.iloc[eval_idx]["origin"]

    best = best_validation_candidate(scores)
    best_key = f"{best['model']}__{best['features']}"

    candidate_keys = [best_key]
    # Also output the best external/tabular candidate even if Holt-Winters wins.
    tabular = (
        scores.loc[
            scores["stage"].eq("validation")
            & scores["horizon_months"].eq(PRIMARY_HORIZON)
            & ~scores["model"].eq("HW_Damped_Add")
        ]
        .sort_values(["mean_WAPE_pct", "median_WAPE_pct"])
        .iloc[0]
    )
    tabular_key = f"{tabular['model']}__{tabular['features']}"
    if tabular_key not in candidate_keys:
        candidate_keys.append(tabular_key)

    forecasts = {
        "HW_Damped_Add__revenue_only": pipeline.ets_forecast(
            revenues,
            CURRENT_FORECAST_HORIZON,
            "HW_Damped_Add",
        )
    }

    for key in candidate_keys:
        if key in forecasts:
            continue
        method, group = key.split("__", 1)
        forecasts[key] = predict_direct(
            snapshots,
            revenues,
            eval_idx,
            CURRENT_FORECAST_HORIZON,
            group,
            method,
        )

    rows = []
    future_dates = pd.date_range(
        origin + pd.offsets.MonthBegin(1),
        periods=CURRENT_FORECAST_HORIZON,
        freq="MS",
    )
    for key, values in forecasts.items():
        model, group = key.split("__", 1)
        for date, value in zip(future_dates, values):
            rows.append(
                {
                    "origin": origin,
                    "forecast_month": date,
                    "model": model,
                    "features": group,
                    "forecast_bn_twd": float(value),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    scores: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    primary = scores.loc[
        scores["horizon_months"].eq(PRIMARY_HORIZON)
    ].copy()
    validation = primary.loc[primary["stage"].eq("validation")].sort_values(
        "mean_WAPE_pct"
    )
    holdout = primary.loc[primary["stage"].eq("holdout")].sort_values(
        "mean_WAPE_pct"
    )
    best = validation.iloc[0]
    best_key = f"{best['model']}__{best['features']}"
    holdout_best = holdout.loc[
        holdout["model"].eq(best["model"])
        & holdout["features"].eq(best["features"])
    ].iloc[0]
    baseline_holdout = holdout.loc[
        holdout["model"].eq("HW_Damped_Add")
    ].iloc[0]

    external_validation = validation.loc[
        ~validation["model"].eq("HW_Damped_Add")
    ].iloc[0]
    external_key = (
        f"{external_validation['model']}__{external_validation['features']}"
    )
    external_holdout = holdout.loc[
        holdout["model"].eq(external_validation["model"])
        & holdout["features"].eq(external_validation["features"])
    ].iloc[0]

    def table_lines(df: pd.DataFrame, limit: int = 12) -> list[str]:
        lines = [
            "| Model | Features | Folds | Mean WAPE | Median WAPE | Bias |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, row in df.head(limit).iterrows():
            lines.append(
                f"| {row['model']} | {row['features']} | {int(row['folds'])} "
                f"| {row['mean_WAPE_pct']:.2f}% "
                f"| {row['median_WAPE_pct']:.2f}% "
                f"| {row['mean_bias_pct']:+.2f}% |"
            )
        return lines

    lines = [
        "# Issue #14 Phase 2 — Quarterly financials + peer revenue + FX",
        "",
        "## Information cutoff",
        "",
        "- TUC monthly revenue: through 2026-09.",
        "- Peer monthly revenue (台光電 2383, 聯茂 6213): through 2026-09.",
        "- CBC NTD/USD monthly average: through 2026-09.",
        "- TUC official financial statements: through **2026Q2**.",
        "- 2026Q3 financial statement is not included because it was not public as of 2026-10-10; the official TUC IR page listed only 2026Q1 and 2026Q2.",
        "",
        "Financial feature groups include gross margin, operating margin, inventory/assets, PP&E/assets, receivables/assets, cash/assets, debt/assets, YTD CAPEX/assets, YTD CFO/assets, and YoY PP&E/inventory/receivables growth. PP&E growth and CAPEX are explicit expansion-capacity proxies.",
        "",
        "## Evaluation design",
        "",
        "- Quarterly rolling origins from 2020-12.",
        "- Validation/model-selection origins through 2023-09.",
        "- Later holdout origins from 2023-12 onward.",
        "- Primary comparison: fixed 12-month path WAPE.",
        "- All tabular feature groups use the same common point-in-time availability window.",
        "- Future realized peer/FX/financial values are never supplied to multi-step forecasts; models use origin-known state plus forecast lead/calendar.",
        "",
        "## 12-month validation ranking",
        "",
        *table_lines(validation, limit=16),
        "",
        "## 12-month later holdout ranking",
        "",
        *table_lines(holdout, limit=16),
        "",
        "## Prespecified conclusion",
        "",
        f"- Validation winner: **{best_key}**, mean WAPE {best['mean_WAPE_pct']:.2f}%.",
        f"- Its later-holdout WAPE: **{holdout_best['mean_WAPE_pct']:.2f}%**.",
        f"- Holt-Winters later-holdout WAPE: **{baseline_holdout['mean_WAPE_pct']:.2f}%**.",
        f"- Best non-HW validation candidate: **{external_key}** ({external_validation['mean_WAPE_pct']:.2f}%), later-holdout **{external_holdout['mean_WAPE_pct']:.2f}%**.",
        "",
        "The production model should only change if an externally enriched candidate beats Holt-Winters on the later holdout with the same target windows. Otherwise the external model remains an explanatory robustness check.",
        "",
        "## Current forecast outputs",
        "",
        "The phase-2 current forecast is emitted only through 2027-12 (15 months from the 2026-09 origin). This avoids presenting unvalidated 27-month exogenous extrapolation as a 2028 point forecast.",
        "",
    ]

    pivot = (
        current.assign(year=current["forecast_month"].dt.year)
        .groupby(["model", "features", "year"], as_index=False)["forecast_bn_twd"]
        .sum()
    )
    lines += [
        "| Model | Features | Year | Forecast months total (NT$ bn) |",
        "|---|---|---:|---:|",
    ]
    for _, row in pivot.iterrows():
        lines.append(
            f"| {row['model']} | {row['features']} | {int(row['year'])} "
            f"| {row['forecast_bn_twd']:.2f} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "- Quarterly financial release dates use conservative proxy dates (Q1 Jun 1, Q2 Sep 1, Q3 Dec 1, annual next Apr 1), not original historical filing timestamps.",
        "- Historical MOPS pages may reflect later presentation/revision conventions.",
        "- Peer choice is limited to two major Taiwan CCL comparables; broader industry/capacity/raw-material series could still add information.",
        "- Overlapping forecast paths are correlated and are not independent significance tests.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(HORIZONS),
        choices=list(HORIZONS),
    )
    args = parser.parse_args()

    snapshots = load_features()
    all_predictions = []
    for horizon in args.horizons:
        all_predictions.append(benchmark_horizon(snapshots, horizon))
    predictions = pd.concat(all_predictions, ignore_index=True)
    scores = summarize(predictions)
    current = current_forecasts(snapshots, scores)

    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    scores.to_csv(SCORES_PATH, index=False, encoding="utf-8-sig")
    current.to_csv(CURRENT_FORECAST_PATH, index=False, encoding="utf-8-sig")
    write_report(scores, current)

    primary = scores.loc[
        scores["horizon_months"].eq(PRIMARY_HORIZON)
    ]
    print("\n12-month validation:")
    print(
        primary.loc[primary["stage"].eq("validation")]
        .sort_values("mean_WAPE_pct")
        .to_string(index=False)
    )
    print("\n12-month holdout:")
    print(
        primary.loc[primary["stage"].eq("holdout")]
        .sort_values("mean_WAPE_pct")
        .to_string(index=False)
    )
    print("\nCurrent 15-month forecast saved:", CURRENT_FORECAST_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
