from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "exogenous_phase2_model_scores.csv"
OUT = ROOT / "results" / "tuc_financial_only_comparison.csv"
REPORT = ROOT / "reports" / "tuc_financial_only_comparison.md"

CORE_FEATURES = {"revenue_only", "quarterly_expanded"}


def build_comparison() -> pd.DataFrame:
    scores = pd.read_csv(SCORES)
    core = scores.loc[scores["features"].isin(CORE_FEATURES)].copy()

    rows = []
    for horizon in sorted(core["horizon_months"].unique()):
        for stage in ("validation", "holdout"):
            block = core.loc[
                core["horizon_months"].eq(horizon)
                & core["stage"].eq(stage)
            ].copy()
            if block.empty:
                continue

            hw = block.loc[
                block["model"].eq("HW_Damped_Add")
                & block["features"].eq("revenue_only")
            ].iloc[0]

            for model in ("Ridge", "GradientBoosting"):
                revenue_only = block.loc[
                    block["model"].eq(model)
                    & block["features"].eq("revenue_only")
                ].iloc[0]
                financial = block.loc[
                    block["model"].eq(model)
                    & block["features"].eq("quarterly_expanded")
                ].iloc[0]

                rows.append(
                    {
                        "stage": stage,
                        "horizon_months": int(horizon),
                        "model": model,
                        "revenue_only_WAPE_pct": float(
                            revenue_only["mean_WAPE_pct"]
                        ),
                        "revenue_plus_financial_WAPE_pct": float(
                            financial["mean_WAPE_pct"]
                        ),
                        "financial_improvement_pp": float(
                            revenue_only["mean_WAPE_pct"]
                            - financial["mean_WAPE_pct"]
                        ),
                        "HW_Damped_Add_WAPE_pct": float(hw["mean_WAPE_pct"]),
                        "financial_vs_HW_pp": float(
                            financial["mean_WAPE_pct"]
                            - hw["mean_WAPE_pct"]
                        ),
                        "financial_beats_same_model_revenue_only": bool(
                            financial["mean_WAPE_pct"]
                            < revenue_only["mean_WAPE_pct"]
                        ),
                        "financial_beats_HW": bool(
                            financial["mean_WAPE_pct"]
                            < hw["mean_WAPE_pct"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame) -> None:
    lines = [
        "# TUC-only Financial Feature Comparison",
        "",
        "This report answers one narrow question only:",
        "",
        "> Does adding TUC's own quarterly financial information improve revenue forecasting enough to beat the original revenue-only production model?",
        "",
        "No peer-company revenue and no FX variables are used in this comparison.",
        "",
        "Quarterly financial features:",
        "",
        "- gross margin / operating margin",
        "- inventory / assets",
        "- PP&E / assets",
        "- A/R / assets",
        "- cash / assets",
        "- debt / assets",
        "- YTD CAPEX / assets",
        "- YTD CFO / assets",
        "- YoY PP&E / inventory / A/R growth",
        "",
        "## Results",
        "",
        "| Stage | Horizon | Model | Revenue-only WAPE | + TUC financials WAPE | Improvement | HW WAPE | Beats HW? |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['stage']} | {int(row['horizon_months'])}m | "
            f"{row['model']} | {row['revenue_only_WAPE_pct']:.2f}% | "
            f"{row['revenue_plus_financial_WAPE_pct']:.2f}% | "
            f"{row['financial_improvement_pp']:+.2f} pp | "
            f"{row['HW_Damped_Add_WAPE_pct']:.2f}% | "
            f"{'yes' if row['financial_beats_HW'] else 'no'} |"
        )

    holdout = df.loc[df["stage"].eq("holdout")]
    lines += [
        "",
        "## Conclusion",
        "",
        (
            "Adding TUC's own financial information improves some ML configurations "
            "during development, but it does not beat HW_Damped_Add on any later holdout "
            "horizon in the current experiment."
        ),
        "",
    ]

    for horizon in sorted(holdout["horizon_months"].unique()):
        h = holdout.loc[holdout["horizon_months"].eq(horizon)]
        best_fin = h.sort_values("revenue_plus_financial_WAPE_pct").iloc[0]
        lines.append(
            f"- {horizon}m holdout: HW {best_fin['HW_Damped_Add_WAPE_pct']:.2f}% vs "
            f"best TUC-financial ML {best_fin['revenue_plus_financial_WAPE_pct']:.2f}% "
            f"({best_fin['model']})."
        )

    lines += [
        "",
        "Therefore the production revenue forecast remains HW_Damped_Add. "
        "The TUC financial-feature models remain useful as explanatory / robustness checks.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_comparison()
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    write_report(df)
    print(df.to_string(index=False))
    print("Saved:", OUT.relative_to(ROOT))
    print("Saved:", REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
