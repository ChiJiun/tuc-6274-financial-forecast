from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import import_tej_tuc


def test_normalize_tej_filters_tuc_and_maps_quarter_label():
    frame = pd.DataFrame(
        {
            "股票代碼": ["6274", "2383"],
            "季別": ["2026Q2", "2026Q2"],
            "財報發布日": ["2026-08-13", "2026-08-12"],
            "營業收入": ["24,355.008", "11,111"],
            "毛利率": ["27.83%", "25.0%"],
            "不動產、廠房及設備": ["6,586.0", "7,000"],
            "資本支出": ["1,234.0", "500"],
        }
    )
    out = import_tej_tuc.normalize_tej(frame, ticker="6274")
    assert len(out) == 1
    assert out.iloc[0]["period"] == pd.Timestamp("2026-06-30")
    assert out.iloc[0]["release_date"] == pd.Timestamp("2026-08-13")
    assert abs(out.iloc[0]["revenue_ntd_m"] - 24355.008) < 1e-9
    assert abs(out.iloc[0]["gross_margin_pct"] - 27.83) < 1e-9


def test_normalize_tej_supports_explicit_column_map():
    frame = pd.DataFrame(
        {
            "security": ["6274"],
            "fiscal_period": ["2025-12-31"],
            "announced": ["2026-03-20"],
            "sales": [30340.235],
        }
    )
    out = import_tej_tuc.normalize_tej(
        frame,
        column_map={
            "ticker": "security",
            "period": "fiscal_period",
            "release_date": "announced",
            "revenue_ntd_m": "sales",
        },
    )
    assert out.iloc[0]["period"] == pd.Timestamp("2025-12-31")
    assert out.iloc[0]["revenue_ntd_m"] == 30340.235
