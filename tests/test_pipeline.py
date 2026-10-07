from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_assignment_financials
import fetch_raw
import pipeline


def test_raw_directory_exists():
    assert (ROOT / "data" / "raw").exists()


def test_parse_latest_mops_month_exact_value():
    p = ROOT / "data" / "raw" / "mops_monthly" / "t21sc03_115_9_0.html"
    assert p.exists(), "expected committed MOPS fixture for 2026-09"
    rec = pipeline.parse_mops_month(p)
    assert rec is not None
    assert rec["date"] == pd.Timestamp("2026-09-01")
    assert rec["revenue_k_twd"] == 6_125_570


def test_monthly_dataset_integrity():
    path = ROOT / "data" / "processed" / "monthly_revenue.csv"
    assert path.exists(), "run src/pipeline.py before tests"
    df = pd.read_csv(path, parse_dates=["date"])

    model_df = df.loc[df["date"].ge(pipeline.MODEL_START)].copy()
    expected = pd.date_range(model_df["date"].min(), model_df["date"].max(), freq="MS")

    assert not model_df["date"].duplicated().any()
    assert list(model_df["date"]) == list(expected)
    assert (model_df["revenue_k_twd"] > 0).all()
    assert np.isfinite(model_df["revenue_bn_twd"]).all()


def test_feature_output_schema_and_values():
    path = ROOT / "data" / "processed" / "ml_features.csv"
    assert path.exists(), "run src/pipeline.py before tests"
    df = pd.read_csv(path)

    expected = {
        "date",
        "target_revenue_bn_twd",
        "target_log_revenue",
        *pipeline.FEATURE_NAMES,
    }
    assert expected.issubset(df.columns)

    numeric = df[["target_revenue_bn_twd", "target_log_revenue", *pipeline.FEATURE_NAMES]]
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_model_outputs_are_within_reference_tolerance():
    summary = pd.read_csv(ROOT / "results" / "model_summary.csv")
    annual = pd.read_csv(ROOT / "results" / "annual_forecast.csv")

    selected = summary.loc[summary["selected"].astype(str).str.lower().eq("true")]
    assert len(selected) == 1
    assert selected.iloc[0]["model"] == "HW_Damped_Add"
    assert abs(float(selected.iloc[0]["mean_WAPE_pct"]) - 17.76) < 0.5
    assert int(selected.iloc[0]["folds"]) >= 15

    expected = {
        2026: 62.59,
        2027: 109.36,
        2028: 152.25,
    }
    actual = annual.set_index("year")["selected_revenue_bn_twd"].to_dict()
    for year, reference in expected.items():
        assert year in actual
        assert abs(float(actual[year]) - reference) < 1.0


def test_dynamic_cutoff_helpers():
    assert fetch_raw.previous_month(2026, 1) == (2025, 12)
    assert fetch_raw.previous_month(2026, 10) == (2026, 9)
    assert fetch_raw.parse_as_of("2026-09") == (2026, 9)
    assert pipeline.parse_forecast_end("2028-12") == pd.Timestamp("2028-12-01")


def test_assignment_history_comes_from_official_mops_and_reconciles():
    history = pd.read_csv(
        ROOT / "data" / "processed" / "TUC_6274_assignment_historical_2016_2025.csv"
    )
    provenance = pd.read_csv(
        ROOT / "data" / "processed" / "TUC_6274_assignment_provenance.csv"
    )
    reconciliation = pd.read_csv(
        ROOT / "data" / "processed" / "TUC_6274_assignment_reconciliation.csv"
    )

    assert history["year"].tolist() == list(range(2016, 2026))
    assert abs(history.loc[history["year"].eq(2024), "revenue_ntd_m"].iloc[0] - 23070.425) < 1e-6
    assert abs(history.loc[history["year"].eq(2025), "revenue_ntd_m"].iloc[0] - 30340.235) < 1e-6
    assert abs(history.loc[history["year"].eq(2025), "cash_dividend_ntd_m"].iloc[0] - 2168.0) < 1e-6
    assert reconciliation["passed"].all()

    reported = provenance.loc[
        provenance["statement"].isin(["income", "balance", "statement_of_changes_in_equity"])
    ]
    assert not reported.empty
    assert reported["source_url"].str.startswith(
        "https://mopsov.twse.com.tw/server-java/t164sb01"
    ).all()
    assert reported["source_file"].str.startswith("data/raw/mops_financials/").all()


def test_assignment_financial_parser_rebuilds_known_year():
    path = build_assignment_financials.raw_path(2024, 4)
    tables = build_assignment_financials.parse_tables(path)
    row, provenance = build_assignment_financials.build_year(2024, tables, path)
    assert abs(row["revenue_ntd_m"] - 23070.425) < 1e-6
    assert abs(row["total_assets_ntd_m"] - 25877.883) < 1e-6
    assert any(p["field"] == "revenue_ntd_m" for p in provenance)


def test_feature_vector_is_finite():
    vals = [1.0 + i * 0.01 for i in range(40)]
    months = [((i % 12) + 1) for i in range(40)]
    x = pipeline.level_features(vals, 30, months)
    assert len(x) == len(pipeline.FEATURE_NAMES)
    assert pd.notna(x).all()
