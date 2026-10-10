from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import build_assignment_financials as annual

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "mops_financials"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT = PROCESSED_DIR / "TUC_6274_quarterly_financial_features_2016_2026Q2.csv"

TICKER = "6274"
START_YEAR = 2016
LATEST_YEAR = 2026
LATEST_QUARTER = 2

INCOME_FIELDS = {
    "revenue_ntd_m": {
        "codes": ["4000"],
        "titles": ["營業收入合計"],
    },
    "cogs_ntd_m": {
        "codes": ["5000"],
        "titles": ["營業成本合計"],
    },
    "gross_profit_ntd_m": {
        "codes": ["5900", "5950"],
        "titles": ["營業毛利（毛損）", "營業毛利（毛損）淨額"],
    },
    "opex_ntd_m": {
        "codes": ["6000"],
        "titles": ["營業費用合計"],
    },
    "operating_income_ntd_m": {
        "codes": ["6900"],
        "titles": ["營業利益（損失）"],
    },
    "pbt_ntd_m": {
        "codes": ["7900"],
        "titles": ["繼續營業單位稅前淨利（淨損）", "稅前淨利（淨損）"],
    },
    "tax_ntd_m": {
        "codes": ["7950"],
        "titles": ["所得稅費用（利益）合計", "所得稅費用（利益）"],
    },
    "net_income_ntd_m": {
        "codes": ["8200", "8000"],
        "titles": ["本期淨利（淨損）", "繼續營業單位本期淨利（淨損）"],
    },
}

BALANCE_FIELDS = {
    "cash_ntd_m": {
        "codes": ["1100"],
        "titles": ["現金及約當現金合計", "現金及約當現金總額", "現金及約當現金"],
    },
    "accounts_receivable_ntd_m": {
        "codes": ["1170"],
        "titles": ["應收帳款淨額", "應收帳款"],
    },
    "inventory_ntd_m": {
        "codes": ["130X"],
        "titles": ["存貨合計", "存貨"],
    },
    "ppe_ntd_m": {
        "codes": ["1600"],
        "titles": ["不動產、廠房及設備合計", "不動產、廠房及設備"],
    },
    "total_assets_ntd_m": {
        "codes": ["1XXX"],
        "titles": ["資產總計"],
    },
    "short_term_borrowings_ntd_m": {
        "codes": ["2100"],
        "titles": ["短期借款合計", "短期借款"],
        "optional_zero": True,
    },
    "accounts_payable_ntd_m": {
        "codes": ["2170"],
        "titles": ["應付帳款合計", "應付帳款"],
    },
    "bonds_payable_ntd_m": {
        "codes": ["2530"],
        "titles": ["應付公司債合計", "應付公司債"],
        "optional_zero": True,
    },
    "long_term_borrowings_ntd_m": {
        "codes": ["2540"],
        "titles": ["長期借款合計", "長期借款"],
        "optional_zero": True,
    },
    "total_liabilities_ntd_m": {
        "codes": ["2XXX"],
        "titles": ["負債總計"],
    },
    "equity_ntd_m": {
        "codes": ["3XXX"],
        "titles": ["權益總額", "權益總計"],
    },
}

CASHFLOW_FIELDS = {
    "cfo_ytd_ntd_m": {
        "codes": ["AAAA"],
        "titles": ["營業活動之淨現金流入（流出）"],
    },
    "capex_ytd_ntd_m": {
        "codes": ["B02700"],
        "titles": ["取得不動產、廠房及設備"],
    },
}


def _normalize(value) -> str:
    return re.sub(r"\s+", "", str(value)).replace("　", "")


def _find_cashflow_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        header = " ".join(map(str, table.columns))
        if "現金流量表" in header or "Statements of Cash Flows" in header:
            return table
    raise RuntimeError("Could not find cash flow table")


def _title_column(table: pd.DataFrame) -> str:
    for col in table.columns:
        text = str(col)
        if "會計項目" in text or "Accounting Title" in text:
            return col
    return table.columns[0]


def _code_column(table: pd.DataFrame) -> str | None:
    for col in table.columns:
        text = str(col)
        if "代號" in text or "Code" in text:
            return col
    return None


def _period_column(
    table: pd.DataFrame,
    year: int,
    quarter: int,
    statement: str,
) -> str:
    candidates = [
        col
        for col in table.columns
        if str(year) in str(col)
        and "會計項目" not in str(col)
        and "Accounting Title" not in str(col)
        and "代號" not in str(col)
        and "Code" not in str(col)
    ]
    if not candidates:
        raise RuntimeError(
            f"No {year} period column in {statement}: {list(table.columns)!r}"
        )

    if statement == "balance":
        # MOPS puts the current quarter-end balance first.
        return candidates[0]

    end_month = quarter * 3
    end_day = 31 if end_month in {3, 12} else 30

    def is_ytd(col: str) -> bool:
        text = _normalize(col)
        # Old layout: 2018年01月01日至2018年09月30日
        old_pattern = (
            rf"{year}年0?1月0?1日至(?:{year}年)?0?{end_month}月{end_day}日"
        )
        # New layout also embeds: 2026/1/1To6/30
        slash_pattern = rf"{year}/0?1/0?1To0?{end_month}/{end_day}"
        return bool(re.search(old_pattern, text)) or bool(
            re.search(slash_pattern, text, flags=re.IGNORECASE)
        )

    for candidate in candidates:
        if is_ytd(str(candidate)):
            return candidate

    # Q1/Q4 usually have only one current-year income/cash-flow column.
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Could not identify YTD {year}Q{quarter} column in {statement}: "
        f"{candidates!r}"
    )


def _extract(
    table: pd.DataFrame,
    value_col: str,
    codes: list[str],
    titles: list[str],
    *,
    optional_zero: bool = False,
) -> float:
    title_col = _title_column(table)
    code_col = _code_column(table)

    if code_col is not None:
        codes_norm = table[code_col].map(annual.normalize_code)
        for code in codes:
            matched = table.loc[codes_norm.eq(code.upper())]
            for _, row in matched.iterrows():
                value = annual.parse_number(row[value_col])
                if value is not None:
                    return float(value) / 1000.0

    title_norm = table[title_col].map(annual.clean_text)
    for title in titles:
        target = annual.clean_text(title)
        matched = table.loc[title_norm.map(lambda x: x == target or x.startswith(target))]
        for _, row in matched.iterrows():
            value = annual.parse_number(row[value_col])
            if value is not None:
                return float(value) / 1000.0

    if optional_zero:
        return 0.0
    raise RuntimeError(
        f"Missing account codes={codes!r} titles={titles!r} from column={value_col!r}"
    )


def conservative_available_at(year: int, quarter: int) -> pd.Timestamp:
    # Deliberately later than Taiwan filing deadlines. These are safe proxy dates,
    # not exact historical filing timestamps.
    if quarter == 1:
        return pd.Timestamp(year=year, month=6, day=1)
    if quarter == 2:
        return pd.Timestamp(year=year, month=9, day=1)
    if quarter == 3:
        return pd.Timestamp(year=year, month=12, day=1)
    return pd.Timestamp(year=year + 1, month=4, day=1)


def period_end(year: int, quarter: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=quarter * 3, day=1) + pd.offsets.MonthEnd(0)


def source_url(year: int, quarter: int) -> str:
    return annual.MOPS_URL.format(
        ticker=TICKER,
        year=year,
        season=quarter,
    )


def build_quarter(year: int, quarter: int) -> dict:
    path = annual.raw_path(year, quarter)
    if not path.exists():
        raise FileNotFoundError(path)

    tables = annual.parse_tables(path)
    income = annual.find_statement_table(tables, "income")
    balance = annual.find_statement_table(tables, "balance")
    cashflow = _find_cashflow_table(tables)

    income_col = _period_column(income, year, quarter, "income")
    balance_col = _period_column(balance, year, quarter, "balance")
    cashflow_col = _period_column(cashflow, year, quarter, "cashflow")

    row: dict[str, object] = {
        "financial_year": year,
        "financial_quarter": quarter,
        "financial_period": f"{year}Q{quarter}",
        "financial_period_end": period_end(year, quarter),
        "financial_available_at": conservative_available_at(year, quarter),
        "financial_availability_policy": (
            "conservative_proxy_Q1_Jun1_Q2_Sep1_Q3_Dec1_Q4_next_Apr1"
        ),
        "financial_source_file": path.relative_to(ROOT).as_posix(),
        "financial_source_url": source_url(year, quarter),
    }

    for name, spec in INCOME_FIELDS.items():
        row[name] = _extract(
            income,
            income_col,
            spec["codes"],
            spec["titles"],
            optional_zero=spec.get("optional_zero", False),
        )
    for name, spec in BALANCE_FIELDS.items():
        row[name] = _extract(
            balance,
            balance_col,
            spec["codes"],
            spec["titles"],
            optional_zero=spec.get("optional_zero", False),
        )
    for name, spec in CASHFLOW_FIELDS.items():
        row[name] = _extract(
            cashflow,
            cashflow_col,
            spec["codes"],
            spec["titles"],
            optional_zero=spec.get("optional_zero", False),
        )

    revenue = float(row["revenue_ntd_m"])
    assets = float(row["total_assets_ntd_m"])
    if revenue <= 0 or assets <= 0:
        raise RuntimeError(f"Invalid revenue/assets for {year}Q{quarter}")

    row["fin_gross_margin_pct"] = (
        float(row["gross_profit_ntd_m"]) / revenue * 100.0
    )
    row["fin_operating_margin_pct"] = (
        float(row["operating_income_ntd_m"]) / revenue * 100.0
    )
    row["fin_inventory_asset_ratio"] = float(row["inventory_ntd_m"]) / assets
    row["fin_ppe_asset_ratio"] = float(row["ppe_ntd_m"]) / assets
    row["fin_ar_asset_ratio"] = float(row["accounts_receivable_ntd_m"]) / assets
    row["fin_cash_asset_ratio"] = float(row["cash_ntd_m"]) / assets
    debt = (
        float(row["short_term_borrowings_ntd_m"])
        + float(row["bonds_payable_ntd_m"])
        + float(row["long_term_borrowings_ntd_m"])
    )
    row["fin_debt_asset_ratio"] = debt / assets
    row["fin_capex_ytd_asset_ratio"] = abs(float(row["capex_ytd_ntd_m"])) / assets
    row["fin_cfo_ytd_asset_ratio"] = float(row["cfo_ytd_ntd_m"]) / assets
    return row


def build_all() -> pd.DataFrame:
    rows = []
    for year in range(START_YEAR, LATEST_YEAR + 1):
        max_quarter = 4 if year < LATEST_YEAR else LATEST_QUARTER
        for quarter in range(1, max_quarter + 1):
            rows.append(build_quarter(year, quarter))

    data = pd.DataFrame(rows).sort_values(
        ["financial_year", "financial_quarter"]
    ).reset_index(drop=True)

    for column in ("ppe_ntd_m", "inventory_ntd_m", "accounts_receivable_ntd_m"):
        data[f"{column.removesuffix('_ntd_m')}_yoy_growth"] = (
            data[column] / data[column].shift(4) - 1.0
        )

    # Verify exact quarter continuity through the latest available financial report.
    expected = [
        (year, quarter)
        for year in range(START_YEAR, LATEST_YEAR + 1)
        for quarter in range(
            1,
            (4 if year < LATEST_YEAR else LATEST_QUARTER) + 1,
        )
    ]
    actual_pairs = list(
        zip(
            data["financial_year"].astype(int),
            data["financial_quarter"].astype(int),
        )
    )
    if actual_pairs != expected:
        raise RuntimeError("Quarterly financial series is not contiguous")

    feature_cols = [
        "fin_gross_margin_pct",
        "fin_operating_margin_pct",
        "fin_inventory_asset_ratio",
        "fin_ppe_asset_ratio",
        "fin_ar_asset_ratio",
        "fin_cash_asset_ratio",
        "fin_debt_asset_ratio",
        "fin_capex_ytd_asset_ratio",
        "fin_cfo_ytd_asset_ratio",
    ]
    if not np.isfinite(data[feature_cols].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite quarterly financial features")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    return data


def main() -> None:
    data = build_all()
    cols = [
        "financial_period",
        "financial_available_at",
        "revenue_ntd_m",
        "fin_gross_margin_pct",
        "fin_operating_margin_pct",
        "fin_inventory_asset_ratio",
        "fin_ppe_asset_ratio",
        "fin_capex_ytd_asset_ratio",
        "ppe_yoy_growth",
    ]
    print(data.tail(12)[cols].to_string(index=False))
    print(f"Saved {len(data)} quarterly releases to {OUTPUT.relative_to(ROOT)}")
    print(
        "Latest official financial statement included:",
        data.iloc[-1]["financial_period"],
    )
    print(
        "2026Q3 financial statement is intentionally absent: as of 2026-10-11 "
        "the official TUC IR page lists only 2026Q1 and 2026Q2."
    )


if __name__ == "__main__":
    main()
