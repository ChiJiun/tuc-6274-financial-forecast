from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fetch_raw
import pipeline


def test_raw_directory_exists():
    assert (ROOT / "data" / "raw").exists()


def test_parse_latest_mops_month_if_present():
    p = ROOT / "data" / "raw" / "mops_monthly" / "t21sc03_115_9_0.html"
    if not p.exists():
        return
    rec = pipeline.parse_mops_month(p)
    assert rec is not None
    assert rec["date"] == pd.Timestamp("2026-09-01")
    assert rec["revenue_k_twd"] > 1_000_000


def test_dynamic_cutoff_helpers():
    assert fetch_raw.previous_month(2026, 1) == (2025, 12)
    assert fetch_raw.previous_month(2026, 10) == (2026, 9)
    assert fetch_raw.parse_as_of("2026-09") == (2026, 9)
    assert pipeline.parse_forecast_end("2028-12") == pd.Timestamp("2028-12-01")


def test_feature_vector_is_finite():
    vals = [1.0 + i * 0.01 for i in range(40)]
    months = [((i % 12) + 1) for i in range(40)]
    x = pipeline.level_features(vals, 30, months)
    assert len(x) == len(pipeline.FEATURE_NAMES)
    assert pd.notna(x).all()
