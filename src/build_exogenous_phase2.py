from __future__ import annotations

import re
from html import unescape
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "exogenous"
PEER_DIR = RAW / "mops_sii_peer_rows"
CBC_HISTORICAL = RAW / "cbc_ntd_usd_monthly_historical.html"
CBC_CURRENT = RAW / "cbc_ntd_usd_monthly_current.html"
PROCESSED = ROOT / "data" / "processed"
QUARTERLY_FINANCIALS = (
    PROCESSED / "TUC_6274_quarterly_financial_features_2016_2026Q2.csv"
)
MONTHLY_REVENUE = PROCESSED / "monthly_revenue.csv"
OUTPUT = PROCESSED / "exogenous_phase2_features.csv"
COVERAGE_OUTPUT = PROCESSED / "exogenous_phase2_coverage.csv"

PEERS = {
    "2383": "emc_2383",
    "6213": "iteq_6213",
}

FINANCIAL_FEATURES_CORE = (
    "fin_gross_margin_pct",
    "fin_operating_margin_pct",
    "fin_inventory_asset_ratio",
    "fin_ppe_asset_ratio",
    "fin_ar_asset_ratio",
)

FINANCIAL_FEATURES_EXPANSION = (
    "fin_cash_asset_ratio",
    "fin_debt_asset_ratio",
    "fin_capex_ytd_asset_ratio",
    "fin_cfo_ytd_asset_ratio",
    "ppe_yoy_growth",
    "inventory_yoy_growth",
    "accounts_receivable_yoy_growth",
)

PEER_FX_FEATURES = (
    "emc_2383_log",
    "emc_2383_yoy",
    "emc_2383_mom",
    "iteq_6213_log",
    "iteq_6213_yoy",
    "iteq_6213_mom",
    "peer_mean_yoy",
    "peer_sum_log",
    "ntd_usd",
    "ntd_usd_mom",
    "ntd_usd_yoy",
    "ntd_usd_roll3",
)


def _clean_number(value: str) -> float:
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in {"", "-", ".", "--"}:
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _extract_peer_revenue(fragment: str, ticker: str) -> float:
    ticker_match = re.search(
        rf"<td\b[^>]*>\s*{re.escape(ticker)}\s*</td>",
        fragment,
        flags=re.IGNORECASE,
    )
    if ticker_match is None:
        raise RuntimeError(f"Ticker {ticker} missing from peer snapshot")

    row_start = fragment.rfind("<tr", 0, ticker_match.start())
    row_end = fragment.find("</tr>", ticker_match.end())
    if row_start < 0 or row_end < 0:
        raise RuntimeError(f"Incomplete row for peer ticker {ticker}")

    row = fragment[row_start : row_end + len("</tr>")]
    raw_cells = re.findall(
        r"<td\b[^>]*>(.*?)</td>",
        row,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cells = []
    for cell in raw_cells:
        plain = re.sub(r"<[^>]+>", " ", cell)
        cells.append(re.sub(r"\s+", " ", unescape(plain)).strip())

    index = cells.index(ticker)
    if len(cells) <= index + 2:
        raise RuntimeError(f"Peer row schema too short for {ticker}: {cells!r}")
    revenue = _clean_number(cells[index + 2])
    if not np.isfinite(revenue) or revenue <= 0:
        raise RuntimeError(f"Invalid peer revenue for {ticker}: {cells[index + 2]!r}")
    return revenue


def build_peer_monthly() -> pd.DataFrame:
    rows = []
    for path in sorted(PEER_DIR.glob("peers_????_??.html")):
        match = re.search(r"peers_(\d{4})_(\d{2})\.html$", path.name)
        if not match:
            continue
        year, month = map(int, match.groups())
        period = pd.Timestamp(year=year, month=month, day=1)
        # Monthly revenue must be disclosed by the 10th; use the 10th as a
        # conservative availability proxy and forecast only after the 16th.
        availability = (
            period + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)
        )
        fragment = path.read_text(encoding="utf-8")
        row = {
            "date": period,
            "peer_available_at": availability,
        }
        for ticker, prefix in PEERS.items():
            row[f"{prefix}_revenue_k_twd"] = _extract_peer_revenue(
                fragment,
                ticker,
            )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No peer revenue snapshots were parsed")

    for prefix in PEERS.values():
        revenue = df[f"{prefix}_revenue_k_twd"] / 1_000_000.0
        df[f"{prefix}_revenue_bn_twd"] = revenue
        df[f"{prefix}_log"] = np.log(revenue)
        df[f"{prefix}_yoy"] = revenue.pct_change(12)
        df[f"{prefix}_mom"] = revenue.pct_change()

    df["peer_mean_yoy"] = df[
        [f"{prefix}_yoy" for prefix in PEERS.values()]
    ].mean(axis=1)
    df["peer_revenue_sum_bn_twd"] = sum(
        df[f"{prefix}_revenue_bn_twd"] for prefix in PEERS.values()
    )
    df["peer_sum_log"] = np.log(df["peer_revenue_sum_bn_twd"])
    return df


def _parse_cbc_table(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    tables = pd.read_html(StringIO(text))
    if not tables:
        raise RuntimeError(f"No CBC table found in {path}")

    table = tables[0].iloc[:, :2].copy()
    current_year: int | None = None
    rows = []
    for period_raw, value_raw in table.itertuples(index=False, name=None):
        period = str(period_raw).strip()
        if period.lower().startswith("period"):
            continue

        match = re.match(r"^(\d{4})/\s*(\d{1,2})$", period)
        if match:
            current_year = int(match.group(1))
            month = int(match.group(2))
        else:
            month_match = re.match(r"^(\d{1,2})$", period)
            if not month_match or current_year is None:
                continue
            month = int(month_match.group(1))

        try:
            value = float(value_raw)
        except (TypeError, ValueError):
            continue
        if not 1 <= month <= 12 or not np.isfinite(value):
            continue
        rows.append(
            {
                "date": pd.Timestamp(
                    year=current_year,
                    month=month,
                    day=1,
                ),
                "ntd_usd": value,
            }
        )
    return pd.DataFrame(rows)


def build_fx_monthly() -> pd.DataFrame:
    frames = [
        _parse_cbc_table(CBC_HISTORICAL),
        _parse_cbc_table(CBC_CURRENT),
    ]
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    df = df.loc[df["date"].between("2013-01-01", "2026-09-01")].copy()
    # CBC states the monthly average is published on the first business day of
    # the next month; use the 5th calendar day as a conservative proxy.
    df["fx_available_at"] = (
        df["date"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=4)
    )
    df["ntd_usd_mom"] = df["ntd_usd"].pct_change()
    df["ntd_usd_yoy"] = df["ntd_usd"].pct_change(12)
    df["ntd_usd_roll3"] = df["ntd_usd"].rolling(3).mean()
    return df


def build_phase2_features() -> pd.DataFrame:
    monthly = pd.read_csv(MONTHLY_REVENUE, parse_dates=["date"])
    monthly = monthly.loc[monthly["date"].ge("2013-01-01")].copy()
    monthly = monthly.sort_values("date").reset_index(drop=True)
    monthly["origin"] = monthly["date"]
    monthly["as_of"] = (
        monthly["origin"]
        + pd.offsets.MonthBegin(1)
        + pd.Timedelta(days=15)
    )

    peer = build_peer_monthly()
    fx = build_fx_monthly()
    quarterly = pd.read_csv(
        QUARTERLY_FINANCIALS,
        parse_dates=[
            "financial_period_end",
            "financial_available_at",
        ],
    ).sort_values("financial_available_at")

    result = monthly.merge(peer, on="date", how="left", validate="one_to_one")
    result = result.merge(fx, on="date", how="left", validate="one_to_one")
    result = pd.merge_asof(
        result.sort_values("as_of"),
        quarterly,
        left_on="as_of",
        right_on="financial_available_at",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("origin").reset_index(drop=True)

    if result["origin"].duplicated().any():
        raise RuntimeError("Duplicate monthly origins")
    expected = pd.date_range(result["origin"].min(), result["origin"].max(), freq="MS")
    if not pd.DatetimeIndex(result["origin"]).equals(expected):
        raise RuntimeError("Monthly exogenous feature origins are not contiguous")

    checks = (
        ("peer_available_at", "peer"),
        ("fx_available_at", "fx"),
        ("financial_available_at", "financial"),
    )
    for column, label in checks:
        known = result[column].notna()
        if (result.loc[known, column] > result.loc[known, "as_of"]).any():
            raise RuntimeError(f"Point-in-time leakage in {label} availability")

    # Latest-origin assertions are intentional: this experiment is explicitly
    # required to cover all public information available through 2026Q3 as of
    # 2026-10-11. 2026Q3 monthly revenue/peer/FX are available, while the
    # company's 2026Q3 financial statement is not yet public; 2026Q2 is the
    # latest official financial statement.
    latest = result.iloc[-1]
    if latest["origin"] != pd.Timestamp("2026-09-01"):
        raise RuntimeError(f"Unexpected latest origin: {latest['origin']}")
    if str(latest["financial_period"]) != "2026Q2":
        raise RuntimeError(
            "Expected 2026Q2 to be the latest official financial statement "
            "available at the 2026-09 revenue origin"
        )

    required_latest = [
        "revenue_bn_twd",
        *PEER_FX_FEATURES,
        *FINANCIAL_FEATURES_CORE,
        *FINANCIAL_FEATURES_EXPANSION,
    ]
    if not np.isfinite(
        latest[required_latest].to_numpy(dtype=float)
    ).all():
        missing = [
            col
            for col in required_latest
            if not np.isfinite(float(latest[col]))
        ]
        raise RuntimeError(f"Latest origin has missing features: {missing}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    coverage = []
    for column in [
        "revenue_bn_twd",
        "emc_2383_revenue_bn_twd",
        "iteq_6213_revenue_bn_twd",
        "ntd_usd",
        "fin_gross_margin_pct",
        "fin_ppe_asset_ratio",
        "fin_capex_ytd_asset_ratio",
        "ppe_yoy_growth",
    ]:
        valid = result.loc[result[column].notna(), ["origin", column]]
        coverage.append(
            {
                "series": column,
                "first_period": (
                    valid["origin"].min().strftime("%Y-%m")
                    if len(valid)
                    else ""
                ),
                "last_period": (
                    valid["origin"].max().strftime("%Y-%m")
                    if len(valid)
                    else ""
                ),
                "observations": int(len(valid)),
                "missing": int(result[column].isna().sum()),
            }
        )
    pd.DataFrame(coverage).to_csv(
        COVERAGE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )
    return result


def main() -> None:
    data = build_phase2_features()
    cols = [
        "origin",
        "revenue_bn_twd",
        "emc_2383_revenue_bn_twd",
        "iteq_6213_revenue_bn_twd",
        "ntd_usd",
        "financial_period",
        "fin_gross_margin_pct",
        "fin_operating_margin_pct",
        "fin_ppe_asset_ratio",
        "fin_capex_ytd_asset_ratio",
        "ppe_yoy_growth",
    ]
    print(data.tail(12)[cols].to_string(index=False))
    print(f"Saved {len(data)} rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
