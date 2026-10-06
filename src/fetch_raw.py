from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import datetime, timezone
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
    return True


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            month = 1
            year += 1


def fetch_mops_monthly(
    start_year: int = 2003,
    start_month: int = 12,
    end_year: int = 2026,
    end_month: int = 9,
    refresh: bool = False,
):
    """Download MOPS OTC historical monthly revenue HTML."""
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
    if not ir_html.exists():
        download(CORE_SNAPSHOTS["tuc_investor_relations.html"], ir_html, refresh=True)

    text = ir_html.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    wanted = []
    for a in soup.find_all("a", href=True):
        href = urljoin("https://www.tuc.com.tw", a["href"])
        if "_run.php" not in href:
            continue
        name = safe_filename_from_tuc_link(href)
        if (
            re.search(r"202[1-6]Q[1-4]", name, re.I)
            or "財報2022-2025Q3" in name
            or "合併財報" in name
        ):
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


def infer_source_url(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("data/raw/mops_monthly/"):
        m = re.search(r"t21sc03_(\d+)_(\d+)_0\.html$", path.name)
        if m:
            return (
                "https://mopsov.twse.com.tw/nas/t21/otc/"
                f"t21sc03_{m.group(1)}_{m.group(2)}_0.html"
            )
    if path.name in CORE_SNAPSHOTS:
        return CORE_SNAPSHOTS[path.name]
    if path.parent == IR_DIR:
        return CORE_SNAPSHOTS["tuc_investor_relations.html"]
    return ""


def write_manifest():
    RAW.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in RAW.rglob("*")
        if p.is_file() and p.name != "manifest.csv" and not p.name.endswith(".part")
    ]
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for p in sorted(files):
        rows.append(
            {
                "path": p.relative_to(ROOT).as_posix(),
                "source_url": infer_source_url(p),
                "captured_at_utc": now,
                "sha256": sha256(p),
                "size_bytes": p.stat().st_size,
            }
        )
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
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-month", type=int, default=9)
    parser.add_argument("--skip-ir", action="store_true")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    MOPS_MONTHLY.mkdir(parents=True, exist_ok=True)
    IR_DIR.mkdir(parents=True, exist_ok=True)

    fetch_core_snapshots(refresh=args.refresh)
    fetch_mops_monthly(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        refresh=args.refresh,
    )
    if not args.skip_ir:
        fetch_tuc_ir_attachments(refresh=args.refresh)
    write_manifest()


if __name__ == "__main__":
    main()
