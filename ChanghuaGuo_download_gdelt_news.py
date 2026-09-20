"""Download crude-oil headlines for the last year.

Default source is GDELT 2.0 event dumps on data.gdeltproject.org (works for a
full year, no DOC API quota). Optional --source doc uses the article-title API,
which only covers ~3 months and rate-limits to 1 request / 5 seconds.

Person2 rules:
- at most 10 unique items kept per calendar day
- n_unique is the same-day deduped count before the cap
- near-duplicate titles (first 8 words) and URLs are dropped
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests

from cn_names import FILE_NAMES, col_header, headers, row_zh

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

EVENT_BASE = "http://data.gdeltproject.org/gdeltv2"
DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
OIL_QUERY = (
    '("crude oil" OR WTI OR OPEC OR "oil inventory" OR "oil supply" OR "oil demand") '
    "sourcelang:english"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}
WORD_RE = re.compile(r"[a-z0-9]+")
JUNK_PATH = {
    "news",
    "article",
    "articles",
    "business",
    "energy",
    "world",
    "articleshow",
    "story",
    "index",
    "details",
    "details.aspx",
    "wire",
    "headline",
    "headlines",
    "pntruncate",
    "national",
    "international",
    "markets",
    "commodities",
    "industry",
}
OIL_URL_RE = re.compile(
    r"(crude[\s\-_]?oil|\bopec\+?|\boapec\b|\bwti\b|\bbrent\b|petroleum|"
    r"oil[\s\-_]?(price|gas|export|pipeline|inventory|supply|demand|market|field)|"
    r"gasoline|refiner|tanker|hormuz|aramco|adnoc|pdvsa|rosneft|petrobras|"
    r"oilprice|oilandgas)",
    re.I,
)
OIL_STRONG_RE = re.compile(
    r"\b("
    r"crude|oil|opec|oapec|wti|brent|petroleum|petrol|gasoline|diesel|naphtha|"
    r"kerosene|refiner(?:y|ies)?|tanker|barrels?|bpd|mbpd|"
    r"pipeline|lng|lpg|shale|upstream|downstream|oilfield|"
    r"aramco|adnoc|pdvsa|rosneft|lukoil|petrobras|exxon(?:mobil)?|chevron|"
    r"conocophillips|totalenergies|equinor|petronas|sinopec|petrochina|cnooc|"
    r"nnpc|pemex|sonatrach|qatarenergy|socar|inpex|transneft|kazmunai|"
    r"hormuz|kirkuk|permian|bakken|orinoco|petrocedeno|rumaila|"
    r"oilprice|oilandgas"
    r")\b",
    re.I,
)
QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)
BRENT_NOT_OIL_RE = re.compile(
    r"\bbrent\s+(cook|smith|howard|davison|renaud|unruh|fitz)\b|"
    r"\bbrent\b.{0,48}\b(assembly|award|winner|honorary|street)\b",
    re.I,
)
NOT_A_TITLE = {
    "comment page",
    "comment page 1",
    "gail tverberg",
    "details.aspx",
    "index.html",
}
# Four UTC snapshots per day. Together they sample the day without
# downloading all 96 fifteen-minute files.
EVENT_TIMES = (
    "000000",
    "030000",
    "060000",
    "090000",
    "120000",
    "150000",
    "180000",
    "210000",
)


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def daterange(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def near_dup_key(title: str) -> str:
    words = WORD_RE.findall((title or "").lower())
    if not words:
        return (title or "").strip().lower()
    return " ".join(words[:8])


FILE_EXT_RE = re.compile(r"\.(html?|cms|php|aspx|asp|ece|shtml)$", re.I)
CMS_ID_RE = re.compile(r"^(article|story|item|news|wcm)?[-_]?\d+$", re.I)
SHORT_ID_RE = re.compile(r"^a[-_]?\d+$", re.I)
UUID_RE = re.compile(
    r"^[0-9a-f]{8}[- ][0-9a-f]{4}[- ][0-9a-f]{4}[- ][0-9a-f]{4}[- ][0-9a-f]{12}$",
    re.I,
)


def english_only(title: str) -> str:
    text = (title or "").strip()
    if "（" in text:
        text = text.split("（", 1)[0].strip()
    return text


def is_pure_numeric_title(title: str) -> bool:
    text = english_only(title)
    if not text:
        return True
    return bool(re.fullmatch(r"[\d\s._-]+", text)) and bool(re.search(r"\d", text))


def is_garbled_title(title: str) -> bool:
    """CMS filenames, UUIDs, and other non-English slugs that are not real headlines."""
    text = re.sub(r"\s+", " ", english_only(title))
    if not text:
        return True
    if is_pure_numeric_title(text):
        return True
    lower = text.lower()
    if lower.startswith("http") or lower in JUNK_PATH:
        return True
    base = FILE_EXT_RE.sub("", lower).strip(" .-_")
    compact = re.sub(r"[\s_-]+", "", base)
    if CMS_ID_RE.fullmatch(base) or CMS_ID_RE.fullmatch(compact):
        return True
    if SHORT_ID_RE.fullmatch(base) or SHORT_ID_RE.fullmatch(compact):
        return True
    if UUID_RE.fullmatch(base):
        return True
    tokens = WORD_RE.findall(base)
    hex_tokens = [t for t in tokens if re.fullmatch(r"[0-9a-f]{4,}", t)]
    real_words = [
        t
        for t in tokens
        if re.search(r"[a-z]", t)
        and len(t) >= 3
        and not t.isdigit()
        and not re.fullmatch(r"[0-9a-f]+", t)
    ]
    if len(hex_tokens) >= 4 and len(real_words) < 3:
        return True
    if len(tokens) >= 3 and all(re.fullmatch(r"[0-9a-f]{4,}", t) for t in tokens):
        return True
    if len(real_words) < 2 and re.search(r"\d{5,}", compact):
        return True
    return False


def is_broken_encoding(title: str) -> bool:
    text = english_only(title).translate(QUOTE_MAP)
    if not text or "\ufffd" in text:
        return True
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    return any(ord(ch) > 127 for ch in text)


def is_hash_title(title: str) -> bool:
    text = english_only(title).strip()
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 16 and re.fullmatch(r"[A-Za-z0-9]+", compact) and " " not in text:
        vowels = len(re.findall(r"[aeiouAEIOU]", compact))
        digits = sum(ch.isdigit() for ch in compact)
        if vowels <= 3 or digits >= 6:
            return True
    return False


def is_not_a_title(title: str) -> bool:
    text = re.sub(r"\s+", " ", english_only(title)).strip()
    lower = text.lower()
    if lower in NOT_A_TITLE or lower.startswith("comment page"):
        return True
    if is_hash_title(text):
        return True
    if len(text.split()) <= 2 and not OIL_STRONG_RE.search(text):
        return True
    return False


def strip_cms_noise(title: str) -> str:
    text = FILE_EXT_RE.sub("", english_only(title)).translate(QUOTE_MAP)
    text = re.sub(r"\s+", " ", text).strip(" .-_")
    if not re.match(r"^\d[\d,]*\s+(barrels?|million|billion|percent|bpd|mbpd)\b", text, re.I):
        text = re.sub(r"^20\d{2}[\s._-]?(?:0[1-9]|1[0-2])[\s._-]?(?:0[1-9]|[12]\d|3[01])\s+", "", text)
        text = re.sub(r"^20\d{6}\s+", "", text)
        text = re.sub(r"^\d{5,}\s+", "", text)
    text = re.sub(r"\s+article$", "", text, flags=re.I)
    text = re.sub(r"\s+(?:article\s*)?\d{6,}$", "", text, flags=re.I)
    text = re.sub(r"\s+\d+\.\d{4,}$", "", text)
    text = re.sub(r"\s+\d{5,}$", "", text)
    return re.sub(r"\s+", " ", text).strip(" .-_")


def is_strong_oil_related(title: str, url: str = "") -> bool:
    text = english_only(title)
    if BRENT_NOT_OIL_RE.search(text) and not re.search(r"\b(oil|crude|petroleum|opec|wti)\b", text, re.I):
        return False
    if OIL_STRONG_RE.search(text):
        return True
    if re.search(r"\bbps?\b", text, re.I) and re.search(
        r"\b(iraq|oil|petroleum|crude|refiner|upstream|field)\b", text, re.I
    ):
        return True
    if re.search(r"\bshell\b", text, re.I) and re.search(
        r"\b(oil|petroleum|lng|refiner|gas|crude)\b", text, re.I
    ):
        return True
    return False


def slug_usable(text: str) -> bool:
    slug = re.sub(r"\s+", " ", english_only(text))
    if len(slug) < 8:
        return False
    if is_pure_numeric_title(slug) or is_garbled_title(slug):
        return False
    if slug.lower() in JUNK_PATH or slug.lower().startswith("http"):
        return False
    return True


def _slug_score(slug: str) -> int:
    words = WORD_RE.findall(slug.lower())
    real = [w for w in words if re.search(r"[a-z]", w) and len(w) >= 3]
    return len(real) * 10 + min(len(slug), 80)


def title_from_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("title", "headline"):
        raw = (query.get(key) or [""])[0]
        text = unquote(raw.replace("+", " ")).strip()
        if slug_usable(text):
            return text[:240]

    path = unquote(parsed.path)
    parts = [p for p in path.split("/") if p]
    candidates: list[str] = []
    for part in parts:
        base = FILE_EXT_RE.sub("", part)
        if not base or base.lower() in JUNK_PATH or re.fullmatch(r"\d+", base):
            continue
        slug = re.sub(r"[-_]+", " ", base)
        slug = re.sub(r"\s+", " ", slug).strip()
        if slug_usable(slug):
            candidates.append(slug)
    if not candidates:
        return ""
    return max(candidates, key=_slug_score)[:240]


def parse_seen(seendate: str) -> tuple[str, str]:
    raw = (seendate or "").strip()
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.date().isoformat(), dt.strftime("%H:%M:%S")
    except ValueError:
        return "", ""


def normalize_row(title: str, source: str, url: str, day: date, time_str: str) -> dict[str, str] | None:
    title = (title or "").strip()
    url = (url or "").strip()
    if not url:
        return None
    if not title:
        title = title_from_url(url)
    if not title or is_pure_numeric_title(title):
        return None
    return {
        "date": day.isoformat(),
        "time": time_str,
        "title": title,
        "source": (source or urlparse(url).netloc).strip(),
        "url": url,
        "language": "English",
        "sourcecountry": "",
    }


def dedup_same_day(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen_title: set[str] = set()
    seen_url: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        title_key = near_dup_key(row["title"])
        url_key = row["url"].split("?")[0].rstrip("/").lower()
        if title_key and title_key in seen_title:
            continue
        if url_key and url_key in seen_url:
            continue
        if title_key:
            seen_title.add(title_key)
        if url_key:
            seen_url.add(url_key)
        unique.append(row)
    return unique


def fetch_event_zip(session: requests.Session, stamp: str, timeout: int) -> bytes | None:
    url = f"{EVENT_BASE}/{stamp}.export.CSV.zip"
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        print(f"  miss {stamp}: {exc}")
        return None


def rows_from_event_zip(blob: bytes, day: date, hhmmss: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    time_str = f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return rows
    name = zf.namelist()[0]
    with zf.open(name) as fh:
        for raw in fh:
            try:
                line = raw.decode("utf-8", "replace")
            except Exception:
                continue
            if not OIL_URL_RE.search(line):
                continue
            parts = line.rstrip("\n").split("\t")
            url = parts[-1].strip() if parts else ""
            if not url.startswith("http"):
                continue
            if not OIL_URL_RE.search(url):
                continue
            source = urlparse(url).netloc.lower()
            item = normalize_row(title_from_url(url), source, url, day, time_str)
            if item:
                rows.append(item)
    return rows


def collect_events_for_day(session: requests.Session, day: date, timeout: int, sleep_s: float) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    ymd = day.strftime("%Y%m%d")
    for hhmmss in EVENT_TIMES:
        blob = fetch_event_zip(session, ymd + hhmmss, timeout)
        if blob:
            collected.extend(rows_from_event_zip(blob, day, hhmmss))
        time.sleep(sleep_s)
    return collected


def request_doc_articles(session: requests.Session, day: date, timeout: int) -> tuple[bool, list[dict[str, Any]], str]:
    params = {
        "query": OIL_QUERY,
        "mode": "ArtList",
        "maxrecords": "250",
        "startdatetime": day.strftime("%Y%m%d000000"),
        "enddatetime": day.strftime("%Y%m%d235959"),
        "format": "json",
        "sort": "DateAsc",
    }
    last_error = "unknown"
    for attempt in range(5):
        try:
            resp = session.get(DOC_API, params=params, headers=HEADERS, timeout=timeout)
            body = (resp.text or "").strip()
            if resp.status_code == 429 or body.startswith("Please limit requests"):
                wait = min(180, 60 * (attempt + 1))
                print(f"  DOC 429 on {day}, sleep {wait}s")
                time.sleep(wait)
                last_error = "HTTP 429"
                continue
            resp.raise_for_status()
            if not body or body[0] not in "{[":
                last_error = body[:160].replace("\n", " ")
                time.sleep(8 * (attempt + 1))
                continue
            payload = json.loads(body)
            articles = payload.get("articles") or []
            if isinstance(articles, dict):
                articles = [articles]
            return True, list(articles), ""
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(10 * (attempt + 1))
    return False, [], last_error


def collect_doc_for_day(session: requests.Session, day: date, timeout: int) -> tuple[str, list[dict[str, str]], str]:
    ok, articles, err = request_doc_articles(session, day, timeout)
    if not ok:
        return "error", [], err
    mapped: list[dict[str, str]] = []
    for raw in articles:
        title = (raw.get("title") or "").strip()
        url = (raw.get("url") or "").strip()
        source = (raw.get("domain") or "").strip()
        _, time_str = parse_seen(str(raw.get("seendate") or ""))
        item = normalize_row(title, source, url, day, time_str or "00:00:00")
        if item:
            item["language"] = (raw.get("language") or "English").strip()
            item["sourcecountry"] = (raw.get("sourcecountry") or "").strip()
            mapped.append(item)
    return "ok", mapped, ""


def cell(row: dict, name: str) -> str:
    return str(row.get(name) or row.get(col_header(name)) or "")


def load_done_dates(daily_path: Path) -> set[str]:
    if not daily_path.exists():
        return set()
    done: set[str] = set()
    with daily_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            day = cell(row, "date")
            status = cell(row, "status")
            if day and status in {"ok", "out_of_window"}:
                done.add(day)
    return done


def write_header_if_needed(path: Path, fieldnames: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.DictWriter(f, fieldnames=headers(fieldnames)).writeheader()


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers(fieldnames), extrasaction="ignore")
        writer.writerows(row_zh(r, fieldnames) for r in rows)


def summarize(daily_path: Path) -> None:
    if not daily_path.exists():
        return
    with daily_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    queried = [r for r in rows if cell(r, "status") == "ok"]
    zero = [r for r in queried if int(cell(r, "n_unique") or 0) == 0]
    unique_sum = sum(int(cell(r, "n_unique") or 0) for r in queried)
    kept_sum = sum(int(cell(r, "n_kept") or 0) for r in queried)
    print("\n=== coverage ===")
    print(f"calendar days: {len(rows)}")
    print(f"queried days:  {len(queried)}")
    print(f"zero-news days: {len(zero)}")
    if queried:
        print(f"zero-news share: {len(zero) / len(queried):.1%}")
    print(f"unique headlines (pre-cap): {unique_sum}")
    print(f"kept headlines (max 10/day): {kept_sum}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download oil headlines, max 10 unique per day.")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--max-per-day", type=int, default=10)
    p.add_argument("--source", choices=("events", "doc"), default="events")
    p.add_argument("--sleep", type=float, default=0.35, help="pause between event-file downloads")
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.days < 1 or args.max_per_day < 1:
        print("--days and --max-per-day must be >= 1", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    headlines_path = out_dir / FILE_NAMES["headlines_raw"]
    daily_path = out_dir / FILE_NAMES["news_daily"]
    coverage_path = out_dir / FILE_NAMES["coverage_calendar"]
    progress_path = out_dir / "progress.json"

    headline_fields = ["date", "time", "title", "source", "url", "language", "sourcecountry"]
    daily_fields = ["date", "n_returned", "n_unique", "n_kept", "status", "note"]
    write_header_if_needed(headlines_path, headline_fields)
    write_header_if_needed(daily_path, daily_fields)
    write_header_if_needed(coverage_path, daily_fields)

    end = utc_today()
    start = end - timedelta(days=args.days - 1)
    days = daterange(start, end)
    done = load_done_dates(daily_path)
    session = requests.Session()

    print(
        f"source={args.source} {start} -> {end}; keep <= {args.max_per_day}/day; "
        f"already done {len(done)}/{len(days)}"
    )

    for i, day in enumerate(days, start=1):
        key = day.isoformat()
        if key in done:
            continue
        print(f"[{i}/{len(days)}] {key}", flush=True)
        note = ""
        status = "ok"
        if args.source == "doc":
            time.sleep(max(5.2, args.sleep))
            status, mapped, note = collect_doc_for_day(session, day, args.timeout)
        else:
            mapped = collect_events_for_day(session, day, args.timeout, args.sleep)
            note = "sampled 8 UTC GDELT event files"

        unique = dedup_same_day(mapped) if status == "ok" else []
        kept = unique[: args.max_per_day]
        row = {
            "date": key,
            "n_returned": len(mapped) if status == "ok" else 0,
            "n_unique": len(unique),
            "n_kept": len(kept),
            "status": status,
            "note": note if status != "ok" else ("" if unique else "no matching headlines"),
        }
        if kept:
            append_rows(headlines_path, headline_fields, kept)
        append_rows(daily_path, daily_fields, [row])
        append_rows(coverage_path, daily_fields, [row])
        if status == "ok":
            done.add(key)
        progress_path.write_text(
            json.dumps(
                {
                    "last_date": key,
                    "done_days": len(done),
                    "source": args.source,
                    "range": [start.isoformat(), end.isoformat()],
                    "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  unique={len(unique)} kept={len(kept)}", flush=True)

    summarize(daily_path)
    print(f"\nWrote:\n  {headlines_path}\n  {daily_path}\n  {coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
