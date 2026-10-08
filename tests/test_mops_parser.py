from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline


def test_parse_latest_mops_month_exact_value():
    path = ROOT / "data" / "raw" / "mops_monthly" / "t21sc03_115_9_0.html"
    rec = pipeline.parse_mops_month(path)
    assert rec["date"] == pd.Timestamp("2026-09-01")
    assert rec["revenue_k_twd"] == 6_125_570


def test_parse_older_mops_layout_exact_value():
    path = ROOT / "data" / "raw" / "mops_monthly" / "t21sc03_102_1_0.html"
    rec = pipeline.parse_mops_month(path)
    assert rec["date"] == pd.Timestamp("2013-01-01")
    assert rec["company_name"] == "台燿"
    assert rec["revenue_k_twd"] == 1_048_274


def test_mops_parser_uses_pandas_fallback_for_alternate_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    alternate = tmp_path / "t21sc03_115_9_0.html"
    alternate.write_text(
        """
        <html><table>
          <thead><tr><th>公司代號</th><th>公司名稱</th><th>本月營收</th></tr></thead>
          <tbody><tr><th>6274</th><td>台燿</td><td>1,234,567</td></tr></tbody>
        </table></html>
        """,
        encoding="big5",
    )
    rec = pipeline.parse_mops_month(alternate)
    assert rec["revenue_k_twd"] == 1_234_567
    assert rec["company_name"] == "台燿"


def test_mops_parser_distinguishes_missing_ticker_from_bad_schema(tmp_path):
    missing = tmp_path / "t21sc03_115_9_0.html"
    missing.write_text(
        "<html><table><tr><td>9999</td><td>Other</td><td>123,456</td></tr></table></html>",
        encoding="big5",
    )
    with pytest.raises(pipeline.MopsTickerNotFound):
        pipeline.parse_mops_month(missing)

    malformed = tmp_path / "t21sc03_115_9_0.html"
    malformed.write_text(
        "<html><table><tr><td>6274</td><td>台燿</td><td>not-a-number</td></tr></table></html>",
        encoding="big5",
    )
    with pytest.raises(pipeline.MopsSchemaError, match="direct parser failed"):
        pipeline.parse_mops_month(malformed)
