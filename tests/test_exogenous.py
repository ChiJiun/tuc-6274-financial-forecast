from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import benchmark_exogenous as benchmark
import build_exogenous_features as exogenous
import build_exogenous_phase2 as phase2
import build_quarterly_financials as quarterly


def sample_history():
    return pd.DataFrame({
        "year": [2016, 2017],
        "gross_margin_pct": [20.0, 99.0],
        "operating_margin_pct": [10.0, 20.0],
        "inventory_ntd_m": [10.0, 15.0],
        "ppe_ntd_m": [20.0, 25.0],
        "accounts_receivable_ntd_m": [30.0, 35.0],
        "total_assets_ntd_m": [100.0, 100.0],
    })


def sample_monthly():
    dates = pd.date_range("2014-01-01", "2019-12-01", freq="MS")
    return pd.DataFrame({"date": dates, "revenue_bn_twd": 1 + np.arange(len(dates)) * 0.01})


def test_financial_asof_rejects_unpublished_fiscal_year():
    data = exogenous.build_features(sample_monthly(), sample_history())
    by_month = data.set_index("origin")
    assert pd.isna(by_month.loc["2017-05-01", "financial_year"])
    assert by_month.loc["2017-06-01", "financial_year"] == 2016
    assert by_month.loc["2018-05-01", "financial_year"] == 2016
    assert by_month.loc["2018-06-01", "financial_year"] == 2017
    # The 2017 statement carries a deliberately distinctive 99% margin:
    # it cannot appear in feature snapshots before the proxy release date.
    assert by_month.loc["2018-05-01", "fin_gross_margin_pct"] == 20.0
    assert by_month.loc["2018-06-01", "fin_gross_margin_pct"] == 99.0
    observed = data.dropna(subset=["financial_available_at"])
    assert (observed["financial_available_at"] <= observed["as_of"]).all()
    assert (observed["financial_period_end"] <= observed["as_of"]).all()


def test_reject_duplicate_months_and_years():
    monthly = sample_monthly()
    history = sample_history()
    with pytest.raises(ValueError, match="Duplicate monthly"):
        exogenous.build_features(pd.concat([monthly, monthly.iloc[[0]]]), history)
    with pytest.raises(ValueError, match="Duplicate annual"):
        exogenous.financial_releases(pd.concat([history, history.iloc[[0]]]))


def test_future_targets_not_in_training_examples():
    feat = exogenous.build_features(sample_monthly(), sample_history())
    revenues = feat["revenue_bn_twd"].to_numpy()
    i = 60  # Known information includes months 0..60, not months 61 onward.
    before_x, before_y = benchmark.training_set(feat, revenues, i, "revenue_plus_financials", 12)
    altered = revenues.copy()
    altered[i + 1:] = 999.0
    after_x, after_y = benchmark.training_set(feat, altered, i, "revenue_plus_financials", 12)
    np.testing.assert_array_equal(before_x, after_x)
    np.testing.assert_array_equal(before_y, after_y)


def test_direct_features_use_origin_financials_only():
    features = exogenous.build_features(sample_monthly(), sample_history())
    row = features.loc[features["origin"].eq(pd.Timestamp("2017-06-01"))].iloc[0]
    vec1 = benchmark.vector(row, 1, "revenue_plus_financials")
    vec12 = benchmark.vector(row, 12, "revenue_plus_financials")
    assert len(vec1) == len(exogenous.REVENUE_FEATURES) + len(exogenous.FINANCIAL_FEATURES) + 3
    np.testing.assert_array_equal(vec1[:-3], vec12[:-3])


def test_real_financial_source_and_proxy_policy():
    features = exogenous.load_features()
    model_data = features.loc[features["origin"].ge("2013-01-01")]
    assert len(model_data) >= 165
    assert model_data["origin"].is_unique
    assert model_data["origin"].max() == pd.Timestamp("2026-09-01")
    assert (model_data["financial_availability_policy"].dropna()
            == "next_year_july_1_proxy_not_actual_release").all()
    dec2025 = features.loc[features["origin"].eq("2025-12-01")].iloc[0]
    jun2026 = features.loc[features["origin"].eq("2026-06-01")].iloc[0]
    assert dec2025["financial_year"] == 2024
    assert jun2026["financial_year"] == 2025


def test_phase2_quarterly_financials_cover_latest_public_statement():
    data = pd.read_csv(
        ROOT / "data" / "processed"
        / "TUC_6274_quarterly_financial_features_2016_2026Q2.csv"
    )
    assert len(data) == 42
    assert data["financial_period"].iloc[0] == "2016Q1"
    assert data["financial_period"].iloc[-1] == "2026Q2"

    q2 = data.iloc[-1]
    assert abs(q2["revenue_ntd_m"] - 24_355.008) < 1e-6
    assert abs(q2["fin_gross_margin_pct"] - 27.830535) < 1e-5
    assert q2["ppe_yoy_growth"] > 0.40
    assert q2["fin_capex_ytd_asset_ratio"] > 0


def test_phase2_latest_origin_has_q3_public_data_without_fake_q3_financials():
    data = pd.read_csv(
        ROOT / "data" / "processed" / "exogenous_phase2_features.csv",
        parse_dates=[
            "origin",
            "as_of",
            "peer_available_at",
            "fx_available_at",
            "financial_available_at",
        ],
    )
    latest = data.iloc[-1]
    assert latest["origin"] == pd.Timestamp("2026-09-01")
    assert latest["financial_period"] == "2026Q2"
    assert abs(latest["revenue_bn_twd"] - 6.125570) < 1e-6
    assert latest["emc_2383_revenue_bn_twd"] > 0
    assert latest["iteq_6213_revenue_bn_twd"] > 0
    assert latest["ntd_usd"] > 0

    for date_column in (
        "peer_available_at",
        "fx_available_at",
        "financial_available_at",
    ):
        known = data[date_column].notna()
        assert (data.loc[known, date_column] <= data.loc[known, "as_of"]).all()


def test_phase2_saved_scorecards_keep_hw_as_holdout_winner():
    summary = pd.read_csv(
        ROOT / "results" / "exogenous_phase2_comparison_summary.csv"
    )
    for horizon in (1, 3, 6, 12, 24):
        row = summary.loc[
            summary["horizon_months"].eq(horizon)
            & summary["stage"].eq("holdout")
        ].iloc[0]
        assert row["best_model"] == "HW_Damped_Add"
        assert row["best_features"] == "revenue_only"
        assert row["best_enriched_WAPE_pct"] > row["hw_WAPE_pct"]

    validation_24 = summary.loc[
        summary["horizon_months"].eq(24)
        & summary["stage"].eq("validation")
    ].iloc[0]
    assert validation_24["best_model"] == "GradientBoosting"
    assert validation_24["best_features"] == "all"


def test_saved_exogenous_outputs_are_consistent():
    scores = pd.read_csv(ROOT / "results" / "exogenous_model_scores.csv")
    predictions = pd.read_csv(ROOT / "results" / "exogenous_fold_predictions.csv")
    releases = pd.read_csv(ROOT / "data" / "processed" / "exogenous_financial_releases.csv")
    assert set(scores["horizon_months"]) == {1, 3, 6, 12}
    assert set(scores["stage"]) == {"validation", "holdout"}
    assert len(releases) == 10
    assert releases["financial_source_url"].str.startswith("https://mopsov.twse.com.tw/").all()
    assert (pd.to_datetime(releases["financial_period_end"]) <
            pd.to_datetime(releases["financial_available_at"])).all()

    # Reference production score is the weighted average of identical 12-month
    # HW folds; older and later stages are reported separately by design.
    hw = scores.loc[(scores["model"].eq("HW_Damped_Add")) &
                    (scores["horizon_months"].eq(12))]
    assert len(hw) == 2
    pooled = np.average(hw["mean_WAPE_pct"], weights=hw["folds"])
    assert abs(pooled - 17.757179643690584) < 0.01
    assert set(predictions["features"]) == {"revenue_only", "revenue_plus_financials"}
    assert (predictions["predicted_bn_twd"] > 0).all()
