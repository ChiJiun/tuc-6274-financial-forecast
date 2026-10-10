from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "tej_tuc_6274_normalized.csv"

ALIASES = {
    "ticker": [
        "ticker", "code", "coid", "公司代碼", "股票代碼", "證券代碼",
    ],
    "period": [
        "period", "date", "年月", "資料年月", "財報年月", "季別", "季度",
    ],
    "release_date": [
        "release_date", "announcement_date", "公布日", "發布日",
        "財報發布日", "公告日", "財報公告日",
    ],
    "revenue_ntd_m": [
        "revenue_ntd_m", "營業收入", "營收", "營業收入淨額",
    ],
    "gross_margin_pct": [
        "gross_margin_pct", "毛利率", "營業毛利率",
    ],
    "operating_margin_pct": [
        "operating_margin_pct", "營業利益率", "營益率",
    ],
    "inventory_ntd_m": [
        "inventory_ntd_m", "存貨",
    ],
    "accounts_receivable_ntd_m": [
        "accounts_receivable_ntd_m", "應收帳款", "應收帳款淨額",
    ],
    "ppe_ntd_m": [
        "ppe_ntd_m", "不動產廠房及設備", "不動產、廠房及設備",
    ],
    "cash_ntd_m": [
        "cash_ntd_m", "現金及約當現金",
    ],
    "total_assets_ntd_m": [
        "total_assets_ntd_m", "資產總額", "資產總計",
    ],
    "total_liabilities_ntd_m": [
        "total_liabilities_ntd_m", "負債總額", "負債總計",
    ],
    "cfo_ntd_m": [
        "cfo_ntd_m", "營業活動現金流量", "營業活動之淨現金流入",
    ],
    "capex_ntd_m": [
        "capex_ntd_m", "資本支出", "購置不動產廠房設備",
        "取得不動產廠房及設備",
    ],
}


def normalize_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("（", "")
        .replace("）", "")
        .replace("、", "")
    )


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Could not decode CSV: {path}")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("TEJ import supports CSV/XLSX/XLS files")


def resolve_columns(
    columns: list[str],
    explicit_map: dict[str, str] | None = None,
) -> dict[str, str]:
    explicit_map = explicit_map or {}
    normalized = {normalize_name(col): col for col in columns}
    resolved: dict[str, str] = {}

    for canonical, source in explicit_map.items():
        if source not in columns:
            raise KeyError(
                f"Mapped TEJ column {source!r} for {canonical!r} does not exist"
            )
        resolved[canonical] = source

    for canonical, aliases in ALIASES.items():
        if canonical in resolved:
            continue
        for alias in [canonical, *aliases]:
            key = normalize_name(alias)
            if key in normalized:
                resolved[canonical] = normalized[key]
                break
    return resolved


def normalize_tej(
    frame: pd.DataFrame,
    *,
    ticker: str = "6274",
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    resolved = resolve_columns(list(frame.columns), column_map)

    required = {"period"}
    missing = sorted(required - set(resolved))
    if missing:
        raise RuntimeError(
            f"Missing required TEJ field(s): {missing}. "
            f"Detected columns: {list(frame.columns)!r}. "
            "Use --column-map for an explicit mapping."
        )

    data = frame.copy()
    if "ticker" in resolved:
        ticker_series = data[resolved["ticker"]].astype(str).str.extract(
            r"(\d+)", expand=False
        )
        data = data.loc[ticker_series.eq(str(ticker))].copy()
        if data.empty:
            raise RuntimeError(f"No TEJ rows found for ticker {ticker}")

    out = pd.DataFrame()
    for canonical, source in resolved.items():
        out[canonical] = data[source]

    period = out["period"].astype(str).str.strip()
    quarter_label = period.str.match(r"^\d{4}Q[1-4]$", na=False)
    parsed_period = pd.to_datetime(
        period.mask(quarter_label),
        errors="coerce",
    )

    # Force quarter labels such as 2026Q2 to quarter-end. pandas otherwise
    # interprets some quarter strings as the quarter start.
    qmask = quarter_label
    if qmask.any():
        years = period.loc[qmask].str.slice(0, 4).astype(int)
        quarters = period.loc[qmask].str.slice(-1).astype(int)
        months = quarters * 3
        parsed_period.loc[qmask] = pd.to_datetime(
            {
                "year": years,
                "month": months,
                "day": 1,
            }
        ) + pd.offsets.MonthEnd(0)

    if parsed_period.isna().any():
        bad = period.loc[parsed_period.isna()].head(10).tolist()
        raise RuntimeError(f"Could not parse TEJ period values: {bad}")
    out["period"] = parsed_period

    if "release_date" in out:
        out["release_date"] = pd.to_datetime(
            out["release_date"], errors="coerce"
        )

    numeric_columns = [
        col
        for col in out.columns
        if col not in {"ticker", "period", "release_date"}
    ]
    for col in numeric_columns:
        out[col] = (
            out[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values("period").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a TEJ Pro / TEJ NEXT export for TUC (6274). "
            "Raw TEJ exports should stay under data/private/tej/ and are not committed."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ticker", default="6274")
    parser.add_argument(
        "--column-map",
        type=Path,
        help="Optional JSON mapping: canonical_name -> TEJ export column",
    )
    args = parser.parse_args()

    mapping = None
    if args.column_map:
        mapping = json.loads(args.column_map.read_text(encoding="utf-8"))

    frame = read_input(args.input)
    normalized = normalize_tej(
        frame,
        ticker=args.ticker,
        column_map=mapping,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Rows:", len(normalized))
    print("Columns:", list(normalized.columns))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
