from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MOPS_MONTHLY = RAW / "mops_monthly"
IR_DIR = RAW / "investor_presentations"
MANIFEST = RAW / "manifest.csv"

CORE_SNAPSHOTS = {
    "stockgo_6274_revenue.html": "https://stockgo.tw/stock/6274/revenue/",
    "moneydj_6274_revenue.html": "https://5850web.moneydj.com/z/zc/zch/zch_6274.djhtm",
    "tuc_investor_relations.html": "https://www.tuc.com.tw/zh-tw/corporate/id/133",
    "mops_monthly_revenue_entry.html": "https://mops.twse.com.tw/mops/web/t120sb02_q10",
}

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/154 Safari/537.36"
        )
    }
)

# Files actually downloaded/replaced during the current process.
# Cached files are intentionally absent so their original manifest timestamp can be preserved.
DOWNLOAD_EVENTS: dict[str, dict[str, str]] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, refresh: bool = False, timeout: int = 45) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    tmp.write_bytes(r.content)
    tmp.replace(dest)

    rel = dest.relative_to(ROOT).as_posix()
    DOWNLOAD_EVENTS[rel] = {
        "source_url": url,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return True


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            month = 1
            year += 1


def previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def default_as_of(today: date | None = None) -> tuple[int, int]:
    """Return the latest normally complete monthly-revenue period."""
    today = today or date.today()
    return previous_month(today.year, today.month)


def parse_as_of(value: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--as-of must be YYYY-MM") from exc
    return parsed.year, parsed.month


def fetch_mops_monthly(
    start_year: int = 2003,
    start_month: int = 12,
    end_year: int | None = None,
    end_month: int | None = None,
    refresh: bool = False,
):
    """Download MOPS OTC historical monthly revenue HTML."""
    if end_year is None or end_month is None:
        end_year, end_month = default_as_of()
    for year, month in iter_months(start_year, start_month, end_year, end_month):
        roc = year - 1911
        url = (
            "https://mopsov.twse.com.tw/nas/t21/otc/"
            f"t21sc03_{roc}_{month}_0.html"
        )
        dest = MOPS_MONTHLY / f"t21sc03_{roc}_{month}_0.html"
        try:
            changed = download(url, dest, refresh=refresh)
            print(("downloaded" if changed else "cached"), dest.relative_to(ROOT))
        except Exception as exc:
            print("WARN MOPS", year, month, exc)


def safe_filename_from_tuc_link(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    raw_name = query.get("name", [""])[0]
    raw_name = unquote(raw_name).strip()
    if not raw_name:
        raw_name = query.get("file", ["download"])[0]
    raw_name = re.sub(r'[<>:"/\\|?*]+', "_", raw_name)
    raw_name = re.sub(r"\s+", " ", raw_name).strip()
    return raw_name or "download"


def fetch_tuc_ir_attachments(refresh: bool = False):
    ir_html = RAW / "tuc_investor_relations.html"
    # Refresh the listing so newly published IR documents can be discovered.
    download(CORE_SNAPSHOTS["tuc_investor_relations.html"], ir_html, refresh=True)

    text = ir_html.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    wanted = []
    for a in soup.find_all("a", href=True):
        href = urljoin("https://www.tuc.com.tw", a["href"])
        if "_run.php" not in href:
            continue
        name = safe_filename_from_tuc_link(href)
        suffix = Path(name).suffix.lower()
        if suffix in {".pdf", ".ppt", ".pptx", ".zip"}:
            wanted.append((href, name))

    seen = set()
    unique = []
    for url, name in wanted:
        if name not in seen:
            seen.add(name)
            unique.append((url, name))

    for url, name in unique:
        dest = IR_DIR / name
        try:
            changed = download(url, dest, refresh=refresh, timeout=90)
            print(("downloaded" if changed else "cached"), dest.relative_to(ROOT))
        except Exception as exc:
            print("WARN IR", name, exc)


def fetch_core_snapshots(refresh: bool = False):
    for filename, url in CORE_SNAPSHOTS.items():
        dest = RAW / filename
        try:
            changed = download(url, dest, refresh=refresh)
            print(("downloaded" if changed else "cached"), dest.relative_to(ROOT))
        except Exception as exc:
            print("WARN CORE", filename, exc)


def tuc_ir_source_map() -> dict[str, str]:
    """Map cached TUC IR filenames to their exact attachment download URLs."""
    ir_html = RAW / "tuc_investor_relations.html"
    if not ir_html.exists():
        return {}

    text = ir_html.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    mapping: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin("https://www.tuc.com.tw", a["href"])
        if "_run.php" not in href:
            continue
        name = safe_filename_from_tuc_link(href)
        suffix = Path(name).suffix.lower()
        if suffix in {".pdf", ".ppt", ".pptx", ".zip"}:
            mapping[name] = href
    return mapping


def infer_source_url(path: Path, ir_sources: dict[str, str] | None = None) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("data/raw/mops_monthly/"):
        m = re.search(r"t21sc03_(\d+)_(\d+)_0\.html$", path.name)
        if m:
            return (
                "https://mopsov.twse.com.tw/nas/t21/otc/"
                f"t21sc03_{m.group(1)}_{m.group(2)}_0.html"
            )

    if rel.startswith("data/raw/mops_financials/"):
        m = re.search(r"6274_(\d{4})_Q([1-4])_C\.html$", path.name)
        if m:
            return (
                "https://mopsov.twse.com.tw/server-java/t164sb01"
                f"?step=1&CO_ID=6274&SYEAR={m.group(1)}"
                f"&SSEASON={m.group(2)}&REPORT_ID=C"
            )

    if rel.startswith("data/raw/exogenous/mops_sii_peer_rows/"):
        m = re.search(r"peers_(\d{4})_(\d{2})\.html$", path.name)
        if m:
            roc_year = int(m.group(1)) - 1911
            month = int(m.group(2))
            return (
                "https://mopsov.twse.com.tw/nas/t21/sii/"
                f"t21sc03_{roc_year}_{month}_0.html"
            )

    if path.name == "cbc_ntd_usd_monthly_historical.html":
        return "https://www.cbc.gov.tw/en/cp-480-58820-D92DC-2.html"
    if path.name == "cbc_ntd_usd_monthly_current.html":
        return "https://www.cbc.gov.tw/en/cp-480-58819-C8475-2.html"

    if path.name in CORE_SNAPSHOTS:
        return CORE_SNAPSHOTS[path.name]

    if path.parent == IR_DIR:
        ir_sources = ir_sources or tuc_ir_source_map()
        return ir_sources.get(path.name, CORE_SNAPSHOTS["tuc_investor_relations.html"])
    return ""


def read_existing_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def file_mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def build_manifest_rows(
    files: list[Path] | None = None,
    *,
    existing_rows: dict[str, dict[str, str]] | None = None,
    download_events: dict[str, dict[str, str]] | None = None,
    ir_sources: dict[str, str] | None = None,
) -> list[dict[str, str | int]]:
    """Build manifest rows while preserving capture times for unchanged cached files."""
    if files is None:
        files = [
            p
            for p in RAW.rglob("*")
            if p.is_file()
            and p.name not in {
                "manifest.csv",
                "README.md",
                "peer_revenue_manifest.csv",
                "source_inventory.csv",
            }
            and not p.name.endswith(".part")
        ]

    existing_rows = existing_rows if existing_rows is not None else read_existing_manifest()
    download_events = download_events if download_events is not None else DOWNLOAD_EVENTS
    ir_sources = ir_sources if ir_sources is not None else tuc_ir_source_map()

    rows: list[dict[str, str | int]] = []
    for path in sorted(files):
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        size = path.stat().st_size
        existing = existing_rows.get(rel)
        event = download_events.get(rel)

        if event is not None:
            captured_at = event["captured_at_utc"]
            source_url = event["source_url"]
        else:
            unchanged = (
                existing is not None
                and existing.get("sha256") == digest
                and str(existing.get("size_bytes", "")) == str(size)
            )
            captured_at = (
                existing["captured_at_utc"]
                if unchanged and existing.get("captured_at_utc")
                else file_mtime_utc(path)
            )
            source_url = infer_source_url(path, ir_sources)

        rows.append(
            {
                "path": rel,
                "source_url": source_url,
                "captured_at_utc": captured_at,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return rows


def write_manifest():
    RAW.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows()
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "source_url", "captured_at_utc", "sha256", "size_bytes"],
        )
        w.writeheader()
        w.writerows(rows)
    print("manifest:", MANIFEST.relative_to(ROOT), f"({len(rows)} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--start-month", type=int, default=12)
    parser.add_argument(
        "--as-of",
        type=parse_as_of,
        help="Freeze the monthly-revenue cutoff at YYYY-MM. Default: previous month.",
    )
    parser.add_argument("--end-year", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--end-month", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--skip-ir", action="store_true")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    MOPS_MONTHLY.mkdir(parents=True, exist_ok=True)
    IR_DIR.mkdir(parents=True, exist_ok=True)

    if args.as_of and (args.end_year is not None or args.end_month is not None):
        parser.error("use either --as-of or legacy --end-year/--end-month, not both")
    if (args.end_year is None) != (args.end_month is None):
        parser.error("legacy --end-year and --end-month must be supplied together")

    if args.as_of:
        end_year, end_month = args.as_of
    elif args.end_year is not None:
        end_year, end_month = args.end_year, args.end_month
    else:
        end_year, end_month = default_as_of()

    fetch_core_snapshots(refresh=args.refresh)
    fetch_mops_monthly(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=end_year,
        end_month=end_month,
        refresh=args.refresh,
    )
    if not args.skip_ir:
        fetch_tuc_ir_attachments(refresh=args.refresh)
    write_manifest()


if __name__ == "__main__":
    main()
