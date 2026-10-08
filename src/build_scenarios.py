from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 6274
INTERVAL_LEVEL = 0.80
GROWTH_SHOCK_PP = 20.0
GROSS_MARGIN_SHOCK_PP = 3.0


def selected_model_name(summary: pd.DataFrame) -> str:
    mask = summary["selected"].astype(str).str.lower().eq("true")
    selected = summary.loc[mask]
    if len(selected) != 1:
        raise RuntimeError("model_summary.csv must contain exactly one selected model")
    return str(selected.iloc[0]["model"])


def empirical_bootstrap_uncertainty(
    annual_forecast: pd.DataFrame,
    selection_folds: pd.DataFrame,
    model_summary: pd.DataFrame,
    ets_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    model = selected_model_name(model_summary)
    errors = (
        selection_folds.loc[
            selection_folds["model"].eq(model),
            "total_error_pct",
        ]
        .dropna()
        .to_numpy(dtype=float)
        / 100.0
    )
    if len(errors) < 10:
        raise RuntimeError("Need at least 10 out-of-sample folds for uncertainty calibration")
    if np.any(errors <= -0.95):
        raise RuntimeError("Backtest errors too close to -100% for multiplicative calibration")

    # Backtest total_error = (forecast - actual) / actual.
    # Therefore actual = forecast / (1 + total_error).
    correction_factors = 1.0 / (1.0 + errors)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(correction_factors, size=BOOTSTRAP_DRAWS, replace=True)

    alpha = (1.0 - INTERVAL_LEVEL) / 2.0
    q_low, q_med, q_high = np.quantile(sampled, [alpha, 0.5, 1.0 - alpha])

    spec_by_year: dict[int, tuple[float, float]] = {}
    for year in (2027, 2028):
        col = f"{year}_revenue_bn_twd"
        if col in ets_sensitivity.columns:
            values = ets_sensitivity[col].dropna().astype(float)
            if len(values):
                spec_by_year[year] = (float(values.min()), float(values.max()))

    rows = []
    for _, record in annual_forecast.iterrows():
        year = int(record["year"])
        if year < 2027:
            continue
        point = float(record["selected_revenue_bn_twd"])
        spec_low, spec_high = spec_by_year.get(year, (np.nan, np.nan))
        rows.append(
            {
                "year": year,
                "point_forecast_bn_twd": point,
                "empirical_80_low_bn_twd": point * q_low,
                "empirical_80_median_bn_twd": point * q_med,
                "empirical_80_high_bn_twd": point * q_high,
                "ets_spec_low_bn_twd": spec_low,
                "ets_spec_high_bn_twd": spec_high,
                "selected_model": model,
                "calibration_folds": len(errors),
                "interval_level": INTERVAL_LEVEL,
                "method": (
                    "bootstrap of multiplicative correction factors from "
                    "12-month rolling-origin total forecast errors"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_revenue_scenarios(annual_forecast: pd.DataFrame) -> pd.DataFrame:
    annual = annual_forecast.set_index("year").sort_index()
    if not {2026, 2027, 2028}.issubset(set(annual.index)):
        raise RuntimeError("Annual forecast must contain 2026, 2027 and 2028")

    base_growth = {
        year: float(annual.loc[year, "yoy_growth_pct"])
        for year in (2027, 2028)
    }
    starting_revenue = float(annual.loc[2026, "selected_revenue_bn_twd"])

    shocks = {
        "Downside": -GROWTH_SHOCK_PP,
        "Base": 0.0,
        "Upside": GROWTH_SHOCK_PP,
    }

    rows = []
    for scenario, shock in shocks.items():
        prior = starting_revenue
        for year in (2027, 2028):
            growth = max(-99.0, base_growth[year] + shock)
            revenue = prior * (1.0 + growth / 100.0)
            rows.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "growth_shock_pp": shock,
                    "base_model_growth_pct": base_growth[year],
                    "scenario_growth_pct": growth,
                    "revenue_bn_twd": revenue,
                    "rule": (
                        f"base model annual growth {shock:+.0f} percentage points; "
                        "scenario path compounds year to year"
                    ),
                }
            )
            prior = revenue
    return pd.DataFrame(rows)


def build_growth_margin_sensitivity(
    annual_forecast: pd.DataFrame,
    assignment_history: pd.DataFrame,
) -> pd.DataFrame:
    annual = annual_forecast.set_index("year").sort_index()
    latest = assignment_history.sort_values("year").iloc[-1]
    base_margin = float(latest["gross_margin_pct"])

    growth_shocks = {
        "Downside": -GROWTH_SHOCK_PP,
        "Base": 0.0,
        "Upside": GROWTH_SHOCK_PP,
    }
    margin_shocks = {
        "Low": -GROSS_MARGIN_SHOCK_PP,
        "Base": 0.0,
        "High": GROSS_MARGIN_SHOCK_PP,
    }

    rows = []
    for year in (2027, 2028):
        prior_base = float(annual.loc[year - 1, "selected_revenue_bn_twd"])
        base_growth = float(annual.loc[year, "yoy_growth_pct"])
        for growth_case, growth_shock in growth_shocks.items():
            growth = max(-99.0, base_growth + growth_shock)
            revenue = prior_base * (1.0 + growth / 100.0)
            for margin_case, margin_shock in margin_shocks.items():
                margin = min(100.0, max(0.0, base_margin + margin_shock))
                rows.append(
                    {
                        "year": year,
                        "growth_case": growth_case,
                        "margin_case": margin_case,
                        "growth_shock_pp": growth_shock,
                        "gross_margin_shock_pp": margin_shock,
                        "revenue_growth_pct": growth,
                        "gross_margin_pct": margin,
                        "prior_revenue_bn_twd": prior_base,
                        "revenue_bn_twd": revenue,
                        "gross_profit_bn_twd": revenue * margin / 100.0,
                        "gross_margin_anchor_year": int(latest["year"]),
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    uncertainty: pd.DataFrame,
    scenarios: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    lines = [
        "# Forecast Uncertainty and Pro Forma Sensitivity",
        "",
        "## Statistical uncertainty",
        "",
        (
            "80% empirical calibration bands use the selected model's 12-month "
            "rolling-origin total forecast errors. They are not business scenarios "
            "and do not include future regime-change risk."
        ),
        "",
    ]
    lines += markdown_table(
        uncertainty,
        [
            "year",
            "point_forecast_bn_twd",
            "empirical_80_low_bn_twd",
            "empirical_80_high_bn_twd",
            "ets_spec_low_bn_twd",
            "ets_spec_high_bn_twd",
        ],
        [
            "Year",
            "Point",
            "Empirical 80% low",
            "Empirical 80% high",
            "ETS spec low",
            "ETS spec high",
        ],
    )

    lines += [
        "",
        "## Business scenarios",
        "",
        (
            "Downside/Base/Upside are deterministic assumptions, not probability "
            "intervals. They apply -20/0/+20 percentage-point shocks to the model's "
            "annual revenue-growth rate and compound the path."
        ),
        "",
    ]
    lines += markdown_table(
        scenarios,
        ["scenario", "year", "scenario_growth_pct", "revenue_bn_twd"],
        ["Scenario", "Year", "Revenue growth", "Revenue (NT$ bn)"],
    )

    lines += [
        "",
        "## Revenue Growth × Gross Margin sensitivity",
        "",
        (
            f"Gross-margin sensitivity is anchored to the latest official full-year "
            f"gross margin and shifted by ±{GROSS_MARGIN_SHOCK_PP:.0f} percentage points."
        ),
        "",
    ]

    for year in (2027, 2028):
        subset = sensitivity.loc[sensitivity["year"].eq(year)].copy()
        pivot = subset.pivot(
            index="growth_case",
            columns="margin_case",
            values="gross_profit_bn_twd",
        ).reindex(index=["Downside", "Base", "Upside"], columns=["Low", "Base", "High"])
        lines.append(f"### {year} Gross Profit (NT$ bn)")
        lines.append("")
        lines.append("| Revenue growth case | Low GM | Base GM | High GM |")
        lines.append("|---|---:|---:|---:|")
        for case, values in pivot.iterrows():
            lines.append(
                f"| {case} | {values['Low']:.2f} | {values['Base']:.2f} | {values['High']:.2f} |"
            )
        lines.append("")

    (REPORTS / "forecast_uncertainty_and_sensitivity.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    annual = pd.read_csv(RESULTS / "annual_forecast.csv")
    selection = pd.read_csv(RESULTS / "model_selection_12m_quarterly.csv")
    summary = pd.read_csv(RESULTS / "model_summary.csv")
    ets = pd.read_csv(RESULTS / "ets_forecast_sensitivity.csv")
    history = pd.read_csv(
        PROCESSED / "TUC_6274_assignment_historical_2016_2025.csv"
    )

    uncertainty = empirical_bootstrap_uncertainty(
        annual,
        selection,
        summary,
        ets,
    )
    scenarios = build_revenue_scenarios(annual)
    sensitivity = build_growth_margin_sensitivity(annual, history)

    uncertainty.to_csv(
        RESULTS / "forecast_uncertainty.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenarios.to_csv(
        RESULTS / "pro_forma_revenue_scenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity.to_csv(
        RESULTS / "pro_forma_growth_margin_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(uncertainty, scenarios, sensitivity)

    print("Statistical uncertainty:")
    print(uncertainty.to_string(index=False))
    print("\nBusiness scenarios:")
    print(scenarios.to_string(index=False))


if __name__ == "__main__":
    main()
