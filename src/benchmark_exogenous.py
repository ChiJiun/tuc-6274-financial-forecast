"""Leakage-aware, annual-financial feature ablation for Issue #14.

This research command is intentionally separate from the production pipeline.
It does not rewrite the selected production model or annual forecasts.
"""
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

import build_exogenous_features as exogenous
import pipeline

ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = ROOT / "results" / "exogenous_fold_predictions.csv"
SCORES_PATH = ROOT / "results" / "exogenous_model_scores.csv"
REPORT_PATH = ROOT / "reports" / "exogenous_financial_ablation.md"
FIRST_ORIGIN = pd.Timestamp("2020-12-01")
VALIDATION_LAST_ORIGIN = pd.Timestamp("2023-09-01")
HORIZONS = (1, 3, 6, 12)
METHODS = ("Ridge", "GradientBoosting")
GROUPS = ("revenue_only", "revenue_plus_financials")
FIELDS_BY_GROUP = {
    "revenue_only": exogenous.REVENUE_FEATURES,
    "revenue_plus_financials": exogenous.REVENUE_FEATURES + exogenous.FINANCIAL_FEATURES,
}


def vector(row: pd.Series, lead: int, group: str) -> np.ndarray:
    """Forecast-date calendar is known in advance; financial state is origin-only."""
    origin = pd.Timestamp(row["origin"])
    target_month = ((origin.month - 1 + lead) % 12) + 1
    vals = [float(row[k]) for k in FIELDS_BY_GROUP[group]]
    vals.extend([
        float(lead) / 12.0,
        math.sin(2 * math.pi * target_month / 12),
        math.cos(2 * math.pi * target_month / 12),
    ])
    x = np.asarray(vals, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite input at origin={origin}, lead={lead}")
    return x


def training_set(
    snapshots: pd.DataFrame, revenue: np.ndarray, eval_idx: int,
    group: str, max_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Only labels at or before eval_idx are observable at a forecast origin."""
    x, y = [], []
    for j in range(24, eval_idx):
        row = snapshots.iloc[j]
        if pd.isna(row["financial_year"]):
            continue
        for lead in range(1, min(max_horizon, eval_idx - j) + 1):
            label_idx = j + lead
            if label_idx > eval_idx:
                raise AssertionError("Target leakage in training set")
            x.append(vector(row, lead, group))
            y.append(math.log(float(revenue[label_idx])))
    # A 1-month direct horizon has one example per origin; longer horizons
    # naturally produce more rows. Do not require 100 rows for every horizon.
    if len(x) < 30:
        raise ValueError(f"Insufficient point-in-time training samples: {len(x)}")
    return np.asarray(x), np.asarray(y)


def predict_direct(
    snapshots: pd.DataFrame,
    revenue: np.ndarray,
    eval_idx: int,
    horizon: int,
    group: str,
    method: str,
) -> np.ndarray:
    X, y = training_set(snapshots, revenue, eval_idx, group, horizon)
    if method == "Ridge":
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif method == "GradientBoosting":
        estimator = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.04,
            loss="huber", random_state=42,
        )
    else:
        raise ValueError(method)
    estimator.fit(X, y)
    row = snapshots.iloc[eval_idx]
    ahead = np.vstack([vector(row, h, group) for h in range(1, horizon + 1)])
    return np.exp(estimator.predict(ahead))


def benchmark(features: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Produce per-month predictions; fold stage never affects training."""
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}")
    ordered = features.loc[features["origin"].ge(pipeline.MODEL_START)].reset_index(drop=True)
    actual = ordered["revenue_bn_twd"].to_numpy(dtype=float)
    dates = pd.to_datetime(ordered["origin"])
    first_history_month = dates.iloc[0].month
    rows = []

    for i, origin in enumerate(dates):
        if origin < FIRST_ORIGIN or origin.month not in (3, 6, 9, 12):
            continue
        if i + horizon >= len(ordered):
            continue
        stage = "validation" if origin <= VALIDATION_LAST_ORIGIN else "holdout"
        if pd.isna(ordered.iloc[i]["financial_year"]):
            raise ValueError(f"No published financial data at origin {origin}")
        observed = actual[i + 1:i + 1 + horizon]
        candidates = {
            "HW_Damped_Add__revenue_only": pipeline.ets_forecast(
                actual[:i + 1], horizon, "HW_Damped_Add"
            )
        }
        for group in GROUPS:
            for method in METHODS:
                key = f"{method}__{group}"
                candidates[key] = predict_direct(ordered, actual, i, horizon, group, method)

        for name, preds in candidates.items():
            model, group = name.split("__", 1)
            for lead, (y_true, y_pred) in enumerate(zip(observed, preds), 1):
                if not np.isfinite(y_pred) or y_pred <= 0:
                    raise ValueError(f"Invalid prediction at {origin} for {name}")
                rows.append({
                    "origin": origin.strftime("%Y-%m"),
                    "stage": stage,
                    "horizon_months": horizon,
                    "model": model,
                    "features": group,
                    "lead_month": lead,
                    "target_month": dates.iloc[i + lead].strftime("%Y-%m"),
                    "actual_bn_twd": y_true,
                    "predicted_bn_twd": y_pred,
                })
    if not rows:
        raise ValueError("No eligible folds")
    return pd.DataFrame(rows)


def summarize(per_month: pd.DataFrame) -> pd.DataFrame:
    """Mean fold WAPE, not WAPE across all overlapping observations."""
    fold_keys = ["stage", "horizon_months", "model", "features", "origin"]
    def fold_metrics(data: pd.DataFrame) -> pd.Series:
        actual = data["actual_bn_twd"].to_numpy()
        pred = data["predicted_bn_twd"].to_numpy()
        denom = float(np.sum(actual))
        return pd.Series({
            "wape_pct": 100 * float(np.sum(np.abs(pred - actual))) / denom,
            "mae_bn_twd": float(np.mean(np.abs(pred - actual))),
            "bias_pct": 100 * float(np.sum(pred - actual)) / denom,
            "abs_total_error_pct": 100 * abs(float(np.sum(pred - actual))) / denom,
        })
    folds = per_month.groupby(fold_keys, sort=True).apply(
        fold_metrics, include_groups=False
    ).reset_index()
    result = (folds.groupby(["stage", "horizon_months", "model", "features"])
              .agg(folds=("origin", "count"),
                   mean_WAPE_pct=("wape_pct", "mean"),
                   median_WAPE_pct=("wape_pct", "median"),
                   mean_MAE_bn_twd=("mae_bn_twd", "mean"),
                   mean_bias_pct=("bias_pct", "mean"),
                   mean_abs_total_error_pct=("abs_total_error_pct", "mean"))
              .reset_index().sort_values(
                   ["horizon_months", "stage", "mean_WAPE_pct", "model", "features"]))
    return result


def write_report(scores: pd.DataFrame) -> None:
    val = scores.loc[(scores["stage"] == "validation") & (scores["horizon_months"] == 12)]
    test = scores.loc[(scores["stage"] == "holdout") & (scores["horizon_months"] == 12)]
    if val.empty or test.empty:
        raise ValueError("No complete 12-month validation and holdout")
    chosen = val.sort_values(["mean_WAPE_pct", "model", "features"]).iloc[0]
    label = f'{chosen["model"]} + {chosen["features"]}'
    base = val.loc[val["model"].eq("HW_Damped_Add")].iloc[0]
    chosen_test = test.loc[
        test["model"].eq(chosen["model"]) & test["features"].eq(chosen["features"])
    ].iloc[0]
    base_test = test.loc[test["model"].eq("HW_Damped_Add")].iloc[0]
    lines = [
        "# Issue #14 — Annual financial features (first ablation)",
        "",
        "## Scope",
        "",
        "- **Experiment only:** production forecasts and model selection remain unchanged.",
        "- Source: committed 2016–2025 official-MOPS reconstructed annual financials plus monthly revenue.",
        "- Financial inputs: latest presumed-published gross margin, operating margin, inventory/assets, PP&E/assets, and receivables/assets.",
        "- Financial statement availability: **July 1 of the following year** (conservative proxy; **not verified actual announcement dates**).",
        "- Revenue forecast origin: **16th of the following month** (calendar proxy; publication dates not verified).",
        "- Direct multi-step forecast: all features at origin; only known target month and lead are used for future horizons. No future financial values.",
        "- Same quarterly origins and observed targets for every candidate; selection through 2023-09, subsequent holdout from 2023-12.",
        "- The holdout is newly reserved for this ablation, but **not independent of the previously selected HW baseline**.",
        "- Overlapping rolling folds are correlated; no formal significance claim.",
        "",
        "## 12-month results",
        "",
        "| Stage | Model / features | Folds | Mean WAPE | Mean signed bias |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in scores.loc[scores["horizon_months"].eq(12)].iterrows():
        lines.append(
            f'| {r["stage"]} | {r["model"]} / {r["features"]} | {int(r["folds"])} '
            f'| {r["mean_WAPE_pct"]:.2f}% | {r["mean_bias_pct"]:+.2f}% |'
        )
    lines += [
        "",
        "## Prespecified selection",
        "",
        f'- Best 12-month **validation** candidate: **{label}** ({chosen["mean_WAPE_pct"]:.2f}% mean WAPE).',
        f'- Validation baseline: HW_Damped_Add ({base["mean_WAPE_pct"]:.2f}% mean WAPE).',
        f'- Later **holdout**: chosen candidate {chosen_test["mean_WAPE_pct"]:.2f}% versus baseline {base_test["mean_WAPE_pct"]:.2f}%.',
        "",
        "## Limitations and next steps",
        "",
        "- **Do not treat the 17.76% production benchmark as directly comparable to the validation-only score.** Compare on the identical stage/origins.",
        "- Only reconstructed annual financial statements were added; peer monthly revenue, FX and industry series are **not yet implemented**.",
        "- The publication-date policy is a proxy. Replace with verifiable original announcements and handle revisions before claiming historical point-in-time accuracy.",
        "- The chosen candidate is selected once from validation; do not tune after looking at the holdout.",
        "- Annual ratios provide relatively few distinct states; this is exploratory and may not improve accuracy.",
        "- Evaluate 1/3/6/12-month horizons in the score CSV; a 24-month study remains follow-up.",
        "- Revisit the 2026 regime-change performance and multi-company/industry sources before promoting an alternate production model.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    args = parser.parse_args()
    features = exogenous.load_features()
    combined = pd.concat([benchmark(features, horizon=h) for h in args.horizons])
    scores = summarize(combined)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FOLDS_PATH, index=False)
    scores.to_csv(SCORES_PATH, index=False)
    if 12 in args.horizons:
        write_report(scores)
    print(scores.to_string(index=False))
    print(f"Saved {FOLDS_PATH} and {SCORES_PATH}")


if __name__ == "__main__":
    main()
