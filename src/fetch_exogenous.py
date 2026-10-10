from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "exogenous"
PEER_DIR = RAW / "mops_sii_peer_rows"
PEER_MANIFEST = RAW / "peer_revenue_manifest.csv"
CBC_HISTORICAL_HTML = RAW / "cbc_ntd_usd_monthly_historical.html"
CBC_CURRENT_HTML = RAW / "cbc_ntd_usd_monthly_current.html"
SOURCE_INVENTORY = RAW / "source_inventory.csv"

PEERS = {
    "2383": "台光電",
    "6213": "聯茂",
}
START_YEAR = 2013
START_MONTH = 1
END_YEAR = 2026
END_MONTH = 9

CBC_HISTORICAL_URL = "https://www.cbc.gov.tw/en/cp-480-58820-D92DC-2.html"
CBC_CURRENT_URL = "https://www.cbc.gov.tw/en/cp-480-58819-C8475-2.html"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/154 Safari/537.36"
    )
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def decode_big5(raw: bytes) -> str:
    for encoding in ("big5", "cp950", "big5hkscs"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("big5", errors="replace")


def mops_sii_url(year: int, month: int) -> str:
    roc_year = year - 1911
    return (
        "https://mopsov.twse.com.tw/nas/t21/sii/"
        f"t21sc03_{roc_year}_{month}_0.html"
    )


def extract_peer_rows(text: str, source_url: str) -> str:
    rows = []
    for ticker, name in PEERS.items():
        ticker_match = re.search(
            rf"<td\b[^>]*>\s*{ticker}\s*</td>",
            text,
            flags=re.IGNORECASE,
        )
        if ticker_match is None:
            raise RuntimeError(f"{source_url}: peer ticker {ticker} ({name}) not found")
        start = text.rfind("<tr", 0, ticker_match.start())
        end = text.find("</tr>", ticker_match.end())
        if start < 0 or end < 0:
            raise RuntimeError(f"{source_url}: incomplete row for peer {ticker}")
        rows.append(text[start : end + len("</tr>")])
    return "\n".join(rows) + "\n"


def fetch_peer_rows(
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
    refresh: bool = False,
) -> None:
    PEER_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for year, month in iter_months(start_year, start_month, end_year, end_month):
        destination = PEER_DIR / f"peers_{year}_{month:02d}.html"
        url = mops_sii_url(year, month)

        if not destination.exists() or refresh:
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    response = requests.get(
                        url,
                        headers=HEADERS,
                        timeout=60,
                    )
                    response.raise_for_status()
                    text = decode_big5(response.content)
                    fragment = extract_peer_rows(text, url)
                    destination.write_text(fragment, encoding="utf-8", newline="\n")
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 3:
                        raise RuntimeError(
                            f"Failed to fetch peer revenue for {year}-{month:02d}"
                        ) from last_error
                    time.sleep(attempt * 2)

        rows.append(
            {
                "period": f"{year}-{month:02d}",
                "source_url": url,
                "source_file": destination.relative_to(ROOT).as_posix(),
                "captured_at_utc": datetime.fromtimestamp(
                    destination.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
                "availability_rule": (
                    "conservative proxy: 10th calendar day of following month"
                ),
                "extraction": "exact peer <tr> fragments from official MOPS SII page",
            }
        )
        print("peer", year, month, destination.name)

    with PEER_MANIFEST.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fetch_cbc_fx(
    refresh: bool = False,
    *,
    allow_insecure_tls: bool = False,
) -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    for url, destination in (
        (CBC_HISTORICAL_URL, CBC_HISTORICAL_HTML),
        (CBC_CURRENT_URL, CBC_CURRENT_HTML),
    ):
        if destination.exists() and not refresh:
            continue
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
            verify=not allow_insecure_tls,
        )
        response.raise_for_status()
        if len(response.content) < 20_000:
            raise RuntimeError(
                f"Unexpectedly small CBC response from {url}: {len(response.content)} bytes"
            )
        destination.write_bytes(response.content)


def write_source_inventory() -> None:
    rows = [
        {
            "series": "peer_monthly_revenue_2383",
            "entity": "台光電 2383",
            "source": "MOPS SII historical monthly revenue",
            "source_url": "https://mopsov.twse.com.tw/nas/t21/sii/",
            "unit": "NTD thousand",
            "frequency": "monthly",
            "coverage": "2013-01 through 2026-09",
            "availability_rule": "10th calendar day of following month (conservative proxy)",
            "revision_policy": "raw row snapshot pinned by SHA-256; no backfill before availability",
        },
        {
            "series": "peer_monthly_revenue_6213",
            "entity": "聯茂 6213",
            "source": "MOPS SII historical monthly revenue",
            "source_url": "https://mopsov.twse.com.tw/nas/t21/sii/",
            "unit": "NTD thousand",
            "frequency": "monthly",
            "coverage": "2013-01 through 2026-09",
            "availability_rule": "10th calendar day of following month (conservative proxy)",
            "revision_policy": "raw row snapshot pinned by SHA-256; no backfill before availability",
        },
        {
            "series": "ntd_usd_monthly_average",
            "entity": "NTD/USD",
            "source": "Central Bank of the Republic of China (Taiwan)",
            "source_url": (
                f"{CBC_HISTORICAL_URL} ; {CBC_CURRENT_URL}"
            ),
            "unit": "NTD per USD",
            "frequency": "monthly",
            "coverage": "official historical page; experiment uses 2013-01 onward",
            "availability_rule": "5th calendar day of following month (conservative proxy)",
            "revision_policy": "committed raw CBC page; values parsed as published",
        },
        {
            "series": "tuc_quarterly_financial_drivers",
            "entity": "台燿 6274",
            "source": "MOPS consolidated quarterly financial statements",
            "source_url": "https://mopsov.twse.com.tw/server-java/t164sb01",
            "unit": "ratios / NTD million",
            "frequency": "quarterly",
            "coverage": "2016Q1 through 2026Q2",
            "availability_rule": (
                "Q1 Jun 1 / Q2 Sep 1 / Q3 Dec 1 / annual next Apr 1 "
                "(conservative proxies, not actual filing dates)"
            ),
            "revision_policy": (
                "uses committed official statement snapshots; exact historical "
                "announcement/version timestamps remain a documented gap"
            ),
        },
    ]

    with SOURCE_INVENTORY.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-peers", action="store_true")
    parser.add_argument("--skip-fx", action="store_true")
    parser.add_argument(
        "--allow-insecure-cbc-tls",
        action="store_true",
        help=(
            "Disable TLS verification only when the local certificate stack "
            "cannot validate the CBC site. Do not use by default."
        ),
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if not args.skip_peers:
        fetch_peer_rows(refresh=args.refresh)
    if not args.skip_fx:
        fetch_cbc_fx(
            refresh=args.refresh,
            allow_insecure_tls=args.allow_insecure_cbc_tls,
        )
    write_source_inventory()

    for path in (CBC_HISTORICAL_HTML, CBC_CURRENT_HTML):
        print(
            "CBC:",
            path.relative_to(ROOT),
            sha256(path) if path.exists() else "missing",
        )
    print("Inventory:", SOURCE_INVENTORY.relative_to(ROOT))


if __name__ == "__main__":
    main()
