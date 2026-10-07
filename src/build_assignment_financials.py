from __future__ import annotations

import argparse
import math
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "mops_financials"
PROCESSED_DIR = ROOT / "data" / "processed"

TICKER = "6274"
START_YEAR = 2016
END_YEAR = 2025
MOPS_URL = (
    "https://mopsov.twse.com.tw/server-java/t164sb01"
    "?step=1&CO_ID={ticker}&SYEAR={year}&SSEASON={season}&REPORT_ID=C"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}

FIELD_SPECS = {
    "revenue_ntd_m": {
        "statement": "income",
        "codes": ["4000"],
        "titles": ["營業收入合計"],
    },
    "cogs_ntd_m": {
        "statement": "income",
        "codes": ["5000"],
        "titles": ["營業成本合計"],
    },
    "gross_profit_ntd_m": {
        "statement": "income",
        "codes": ["5900", "5950"],
        "titles": ["營業毛利（毛損）", "營業毛利（毛損）淨額"],
    },
    "opex_ntd_m": {
        "statement": "income",
        "codes": ["6000"],
        "titles": ["營業費用合計"],
    },
    "operating_income_ntd_m": {
        "statement": "income",
        "codes": ["6900"],
        "titles": ["營業利益（損失）"],
    },
    "pbt_ntd_m": {
        "statement": "income",
        "codes": ["7900"],
        "titles": ["繼續營業單位稅前淨利（淨損）", "稅前淨利（淨損）"],
    },
    "tax_ntd_m": {
        "statement": "income",
        "codes": ["7950"],
        "titles": ["所得稅費用（利益）合計", "所得稅費用（利益）"],
    },
    "net_income_ntd_m": {
        "statement": "income",
        "codes": ["8200", "8000"],
        "titles": ["本期淨利（淨損）", "繼續營業單位本期淨利（淨損）"],
    },
    "eps_ntd": {
        "statement": "income",
        "codes": ["9750", "9710"],
        "titles": ["基本每股盈餘合計", "繼續營業單位淨利（淨損）"],
        "scale": 1.0,
    },
    "cash_ntd_m": {
        "statement": "balance",
        "codes": ["1100"],
        "titles": ["現金及約當現金合計", "現金及約當現金總額", "現金及約當現金"],
    },
    "accounts_receivable_ntd_m": {
        "statement": "balance",
        "codes": ["1170"],
        "titles": ["應收帳款淨額", "應收帳款"],
    },
    "inventory_ntd_m": {
        "statement": "balance",
        "codes": ["130X"],
        "titles": ["存貨合計", "存貨"],
    },
    "current_assets_ntd_m": {
        "statement": "balance",
        "codes": ["11XX"],
        "titles": ["流動資產合計"],
    },
    "ppe_ntd_m": {
        "statement": "balance",
        "codes": ["1600"],
        "titles": ["不動產、廠房及設備合計", "不動產、廠房及設備"],
    },
    "total_assets_ntd_m": {
        "statement": "balance",
        "codes": ["1XXX"],
        "titles": ["資產總計"],
    },
    "short_term_borrowings_ntd_m": {
        "statement": "balance",
        "codes": ["2100"],
        "titles": ["短期借款合計", "短期借款"],
        "optional_zero": True,
    },
    "accounts_payable_ntd_m": {
        "statement": "balance",
        "codes": ["2170"],
        "titles": ["應付帳款合計", "應付帳款"],
    },
    "current_liabilities_ntd_m": {
        "statement": "balance",
        "codes": ["21XX"],
        "titles": ["流動負債合計"],
    },
    "bonds_payable_ntd_m": {
        "statement": "balance",
        "codes": ["2530"],
        "titles": ["應付公司債合計", "應付公司債"],
        "optional_zero": True,
    },
    "long_term_borrowings_ntd_m": {
        "statement": "balance",
        "codes": ["2540"],
        "titles": ["長期借款合計", "長期借款"],
        "optional_zero": True,
    },
    "total_liabilities_ntd_m": {
        "statement": "balance",
        "codes": ["2XXX"],
        "titles": ["負債總計"],
    },
    "equity_ntd_m": {
        "statement": "balance",
        "codes": ["3XXX"],
        "titles": ["權益總額", "權益總計"],
    },
}


def source_url(year: int, season: int = 4) -> str:
    return MOPS_URL.format(ticker=TICKER, year=year, season=season)


def raw_path(year: int, season: int = 4) -> Path:
    return RAW_DIR / f"{TICKER}_{year}_Q{season}_C.html"


def download_statement(year: int, season: int = 4, refresh: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = raw_path(year, season)
    if path.exists() and not refresh:
        return path

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                source_url(year, season),
                headers=HEADERS,
                timeout=90,
            )
            response.raise_for_status()
            if len(response.content) < 10_000:
                raise RuntimeError(
                    f"Unexpectedly small MOPS response for {year} Q{season}: "
                    f"{len(response.content)} bytes"
                )
            path.write_bytes(response.content)
            return path
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(
        f"Failed to download MOPS statement for {year} Q{season} after 3 attempts"
    ) from last_error


def decode_mops(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("big5", "cp950", "big5hkscs", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("big5", errors="replace")


def flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [
            " | ".join(str(part) for part in col if str(part) != "nan")
            for col in table.columns
        ]
    else:
        table.columns = [str(col) for col in table.columns]
    return table


def parse_tables(path: Path) -> list[pd.DataFrame]:
    text = decode_mops(path)
    return [flatten_columns(t) for t in pd.read_html(StringIO(text))]


def find_statement_table(
    tables: list[pd.DataFrame],
    statement: str,
) -> pd.DataFrame:
    needles = {
        "balance": ("資產負債表", "Balance Sheet"),
        "income": ("綜合損益表", "Statement of Comprehensive Income"),
    }[statement]

    for table in tables:
        header = " ".join(map(str, table.columns))
        if any(needle in header for needle in needles):
            return table
    raise RuntimeError(f"Could not find {statement} table")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).strip()


def parse_number(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if math.isnan(float(value)):
            return None
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text or text.lower() == "nan" or text in {"-", "--"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    try:
        result = float(text)
    except ValueError:
        return None
    return -result if negative else result


def normalize_code(value) -> str:
    text = clean_text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.upper()


def current_year_column(table: pd.DataFrame, year: int) -> str:
    candidates = [
        col
        for col in table.columns
        if str(year) in str(col)
        and "會計項目" not in str(col)
        and "代號" not in str(col)
    ]
    if not candidates:
        raise RuntimeError(f"No current-year column found for {year}")
    return candidates[0]


def title_column(table: pd.DataFrame) -> str:
    for col in table.columns:
        if "會計項目" in str(col) or "Accounting Title" in str(col):
            return col
    return table.columns[0]


def code_column(table: pd.DataFrame) -> str | None:
    for col in table.columns:
        if "代號" in str(col) or "Code" in str(col):
            return col
    return None


def extract_account(
    table: pd.DataFrame,
    year: int,
    codes: list[str],
    titles: list[str],
) -> tuple[float | None, str | None, str | None]:
    value_col = current_year_column(table, year)
    title_col = title_column(table)
    code_col = code_column(table)

    if code_col is not None:
        normalized = table[code_col].map(normalize_code)
        for code in codes:
            matched = table.loc[normalized.eq(code.upper())]
            for _, row in matched.iterrows():
                value = parse_number(row[value_col])
                if value is not None:
                    return value, str(row[title_col]), normalize_code(row[code_col])

    normalized_titles = table[title_col].map(clean_text)
    for title in titles:
        target = clean_text(title)
        matched = table.loc[normalized_titles.eq(target)]
        for _, row in matched.iterrows():
            value = parse_number(row[value_col])
            if value is not None:
                code = normalize_code(row[code_col]) if code_col else None
                return value, str(row[title_col]), code

    return None, None, None


def extract_cash_dividend(
    tables: list[pd.DataFrame],
) -> tuple[float | None, str | None]:
    """Return ordinary-share cash dividend in native statement unit (NT$ thousand)."""
    for table in tables:
        row_mask = table.astype(str).apply(
            lambda col: col.str.contains(
                "普通股現金股利",
                regex=False,
                na=False,
            )
        ).any(axis=1)
        if not row_mask.any():
            continue

        row = table.loc[row_mask].iloc[0]
        values = [parse_number(v) for v in row.tolist()]
        numeric = [abs(v) for v in values if v is not None and abs(v) > 0]
        if numeric:
            return max(numeric), "普通股現金股利 Cash dividends of ordinary share"
    return None, None


def build_year(
    year: int,
    tables: list[pd.DataFrame],
    source_file: Path,
) -> tuple[dict, list[dict]]:
    statement_tables = {
        "balance": find_statement_table(tables, "balance"),
        "income": find_statement_table(tables, "income"),
    }
    row = {"year": year}
    provenance = []

    for field, spec in FIELD_SPECS.items():
        value, title, code = extract_account(
            statement_tables[spec["statement"]],
            year,
            spec.get("codes", []),
            spec.get("titles", []),
        )

        status = "reported"
        if value is None and spec.get("optional_zero"):
            value = 0.0
            status = "not_present_assumed_zero"
        if value is None:
            raise RuntimeError(
                f"Missing required field {field} for {year} "
                f"from {source_file.name}"
            )

        scale = float(spec.get("scale", 1 / 1000))
        row[field] = value * scale
        provenance.append(
            {
                "year": year,
                "field": field,
                "value": row[field],
                "unit": "NTD" if field == "eps_ntd" else "NTD_million",
                "statement": spec["statement"],
                "accounting_code": code or "",
                "accounting_title": title or "",
                "source_period": f"{year}-Q4",
                "source_file": str(source_file.relative_to(ROOT)).replace("\\", "/"),
                "source_url": source_url(year, 4),
                "derivation": status,
            }
        )
    return row, provenance


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("year").reset_index(drop=True).copy()

    df["sales_growth_pct"] = df["revenue_ntd_m"].pct_change() * 100
    df["cogs_to_sales_pct"] = df["cogs_ntd_m"] / df["revenue_ntd_m"] * 100
    df["gross_margin_pct"] = (
        df["gross_profit_ntd_m"] / df["revenue_ntd_m"] * 100
    )
    df["opex_to_sales_pct"] = df["opex_ntd_m"] / df["revenue_ntd_m"] * 100
    df["operating_margin_pct"] = (
        df["operating_income_ntd_m"] / df["revenue_ntd_m"] * 100
    )
    df["days_sales_in_cash"] = df["cash_ntd_m"] / df["revenue_ntd_m"] * 365

    avg_ar = (df["accounts_receivable_ntd_m"] + df["accounts_receivable_ntd_m"].shift(1)) / 2
    avg_inventory = (df["inventory_ntd_m"] + df["inventory_ntd_m"].shift(1)) / 2
    avg_ap = (df["accounts_payable_ntd_m"] + df["accounts_payable_ntd_m"].shift(1)) / 2

    df["ar_collection_days"] = avg_ar / df["revenue_ntd_m"] * 365
    df["inventory_turnover_x"] = df["cogs_ntd_m"] / avg_inventory
    df["ap_payment_days"] = avg_ap / df["cogs_ntd_m"] * 365
    df["effective_tax_rate_pct"] = df["tax_ntd_m"] / df["pbt_ntd_m"] * 100
    df["dividend_payout_pct"] = (
        df["cash_dividend_ntd_m"] / df["net_income_ntd_m"] * 100
    )
    return df


def reconciliation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = {
        "gross_profit_equals_revenue_minus_cogs": (
            df["gross_profit_ntd_m"] - (df["revenue_ntd_m"] - df["cogs_ntd_m"])
        ),
        "operating_income_equals_gross_profit_minus_opex": (
            df["operating_income_ntd_m"]
            - (df["gross_profit_ntd_m"] - df["opex_ntd_m"])
        ),
        "balance_sheet_balances": (
            df["total_assets_ntd_m"]
            - (df["total_liabilities_ntd_m"] + df["equity_ntd_m"])
        ),
    }
    tolerance_ntd_m = 0.01
    for check, differences in checks.items():
        for year, difference in zip(df["year"], differences):
            difference = float(difference)
            rows.append(
                {
                    "year": int(year),
                    "check": check,
                    "difference_ntd_m": difference,
                    "tolerance_ntd_m": tolerance_ntd_m,
                    "passed": abs(difference) <= tolerance_ntd_m,
                }
            )
    return pd.DataFrame(rows)


def add_derived_provenance(df: pd.DataFrame, provenance: list[dict]) -> None:
    formulas = {
        "sales_growth_pct": "revenue_t / revenue_t-1 - 1",
        "cogs_to_sales_pct": "cogs / revenue",
        "gross_margin_pct": "gross_profit / revenue",
        "opex_to_sales_pct": "opex / revenue",
        "operating_margin_pct": "operating_income / revenue",
        "days_sales_in_cash": "ending_cash / revenue * 365",
        "ar_collection_days": "average_AR / revenue * 365",
        "inventory_turnover_x": "cogs / average_inventory",
        "ap_payment_days": "average_AP / cogs * 365",
        "effective_tax_rate_pct": "tax / PBT",
        "dividend_payout_pct": "next-period ordinary cash dividend / net_income",
    }
    for _, row in df.iterrows():
        year = int(row["year"])
        for field, formula in formulas.items():
            provenance.append(
                {
                    "year": year,
                    "field": field,
                    "value": row[field],
                    "unit": (
                        "times"
                        if field == "inventory_turnover_x"
                        else "days"
                        if field in {
                            "days_sales_in_cash",
                            "ar_collection_days",
                            "ap_payment_days",
                        }
                        else "percent"
                    ),
                    "statement": "derived",
                    "accounting_code": "",
                    "accounting_title": "",
                    "source_period": f"{year}",
                    "source_file": "",
                    "source_url": "",
                    "derivation": formula,
                }
            )


def build_assignment_history(
    fetch_missing: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    required = [(year, 4) for year in range(START_YEAR, END_YEAR + 1)]
    required.append((END_YEAR + 1, 1))

    for year, season in required:
        path = raw_path(year, season)
        if refresh or (fetch_missing and not path.exists()):
            download_statement(year, season, refresh=refresh)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run with --fetch-missing once to cache official MOPS data."
            )

    table_cache = {
        (year, season): parse_tables(raw_path(year, season))
        for year, season in required
    }

    records = []
    provenance: list[dict] = []
    for year in range(START_YEAR, END_YEAR + 1):
        source_file = raw_path(year, 4)
        row, year_provenance = build_year(
            year,
            table_cache[(year, 4)],
            source_file,
        )

        dividend_source_year = year + 1
        dividend_season = 4 if dividend_source_year <= END_YEAR else 1
        dividend, dividend_title = extract_cash_dividend(
            table_cache[(dividend_source_year, dividend_season)]
        )
        if dividend is None:
            raise RuntimeError(
                f"Could not extract dividend paid for earnings year {year} "
                f"from {dividend_source_year} Q{dividend_season}"
            )
        row["cash_dividend_ntd_m"] = dividend / 1000
        dividend_file = raw_path(dividend_source_year, dividend_season)
        provenance.append(
            {
                "year": year,
                "field": "cash_dividend_ntd_m",
                "value": row["cash_dividend_ntd_m"],
                "unit": "NTD_million",
                "statement": "statement_of_changes_in_equity",
                "accounting_code": "B5",
                "accounting_title": dividend_title or "",
                "source_period": f"{dividend_source_year}-Q{dividend_season}",
                "source_file": str(dividend_file.relative_to(ROOT)).replace("\\", "/"),
                "source_url": source_url(dividend_source_year, dividend_season),
                "derivation": "reported cash dividend paid in following period",
            }
        )
        records.append(row)
        provenance.extend(year_provenance)

    df = add_ratios(pd.DataFrame(records))
    reconciliation = reconciliation_table(df)
    if not reconciliation["passed"].all():
        failures = reconciliation.loc[~reconciliation["passed"]]
        raise RuntimeError(
            "Official financial statement reconciliation failed:\n"
            + failures.to_string(index=False)
        )

    add_derived_provenance(df, provenance)
    provenance_df = pd.DataFrame(provenance)

    output = PROCESSED_DIR / "TUC_6274_assignment_historical_2016_2025.csv"
    provenance_output = PROCESSED_DIR / "TUC_6274_assignment_provenance.csv"
    reconciliation_output = PROCESSED_DIR / "TUC_6274_assignment_reconciliation.csv"

    df.to_csv(output, index=False, encoding="utf-8-sig")
    provenance_df.to_csv(provenance_output, index=False, encoding="utf-8-sig")
    reconciliation.to_csv(
        reconciliation_output,
        index=False,
        encoding="utf-8-sig",
    )

    return df, provenance_df, reconciliation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Download missing official MOPS financial-statement HTML files.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload all official MOPS financial-statement HTML files.",
    )
    args = parser.parse_args()

    df, provenance, reconciliation = build_assignment_history(
        fetch_missing=args.fetch_missing or args.refresh,
        refresh=args.refresh,
    )
    print(
        f"Generated {len(df)} annual rows, "
        f"{len(provenance)} provenance rows, "
        f"{len(reconciliation)} reconciliation checks."
    )
    print(
        df[
            [
                "year",
                "revenue_ntd_m",
                "net_income_ntd_m",
                "total_assets_ntd_m",
                "equity_ntd_m",
                "dividend_payout_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
