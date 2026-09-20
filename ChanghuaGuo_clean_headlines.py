"""Build an English-only headline set for later scoring/training.

Keeps every remaining row, including duplicates. Drops numeric junk, CMS
filenames, encoding-broken text, and strings that are not headlines. Requires
a strong oil-related keyword in the cleaned English title.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from cn_names import FILE_NAMES, headers, row_zh
from download_gdelt_news import (
    english_only,
    is_broken_encoding,
    is_garbled_title,
    is_not_a_title,
    is_pure_numeric_title,
    is_strong_oil_related,
    near_dup_key,
    slug_usable,
    strip_cms_noise,
    title_from_url,
)


def cell(row: dict, name: str) -> str:
    if name in row and row[name] not in (None, ""):
        return str(row[name])
    for key, value in row.items():
        if str(key).split("（")[0] == name:
            return str(value or "")
    return ""


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [{str(k): (v if v is not None else "") for k, v in row.items()} for row in csv.DictReader(fh)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_zh(r, fieldnames) for r in rows)


def recover_title(title: str, url: str) -> tuple[str, bool]:
    current = strip_cms_noise(english_only(title))
    if looks_like_headline(current):
        return current, False
    recovered = strip_cms_noise(title_from_url(url))
    if looks_like_headline(recovered):
        return recovered, True
    return current, False


def looks_like_headline(title: str) -> bool:
    text = strip_cms_noise(english_only(title))
    if not text or is_pure_numeric_title(text) or is_garbled_title(text):
        return False
    if is_broken_encoding(text) or is_not_a_title(text) or not slug_usable(text):
        return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="English-only oil headlines, duplicates kept.")
    p.add_argument(
        "--headlines",
        type=Path,
        default=Path(__file__).with_name("data") / FILE_NAMES["headlines_raw"],
    )
    p.add_argument(
        "--daily",
        type=Path,
        default=Path(__file__).with_name("data") / FILE_NAMES["news_daily"],
    )
    p.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("data"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_rows = read_rows(args.headlines)
    daily_rows = read_rows(args.daily) if args.daily.exists() else []

    kept_rows: list[dict[str, str]] = []
    n_numeric_in = 0
    n_recovered = 0
    n_dropped = 0
    n_dropped_encoding = 0
    n_dropped_not_title = 0
    n_dropped_unrelated = 0
    dropped_by_day: Counter[str] = Counter()
    recovered_by_day: Counter[str] = Counter()

    for row in raw_rows:
        day = cell(row, "date")
        title = cell(row, "title")
        url = cell(row, "url")
        if is_pure_numeric_title(english_only(title)):
            n_numeric_in += 1
        cleaned, did_recover = recover_title(title, url)
        if did_recover:
            n_recovered += 1
            recovered_by_day[day] += 1
        if is_broken_encoding(cleaned):
            n_dropped += 1
            n_dropped_encoding += 1
            dropped_by_day[day] += 1
            continue
        if not looks_like_headline(cleaned):
            n_dropped += 1
            n_dropped_not_title += 1
            dropped_by_day[day] += 1
            continue
        if not is_strong_oil_related(cleaned, url):
            n_dropped += 1
            n_dropped_unrelated += 1
            dropped_by_day[day] += 1
            continue
        kept_rows.append(
            {
                "date": day,
                "time": cell(row, "time"),
                "title": cleaned,
                "title_raw": english_only(title),
                "source": cell(row, "source"),
                "url": url,
                "language": "English",
                "sourcecountry": cell(row, "sourcecountry"),
            }
        )

    key_counts = Counter(near_dup_key(r["title"]) for r in kept_rows)
    seen_first: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    dup_rows: list[dict[str, str]] = []
    day_keys: dict[str, list[str]] = defaultdict(list)

    for row in kept_rows:
        key = near_dup_key(row["title"])
        extra_copy = key in seen_first
        flagged = dict(row)
        flagged["is_duplicate"] = "1" if extra_copy else "0"
        dup_rows.append(flagged)
        day_keys[row["date"]].append(key)
        if extra_copy:
            continue
        seen_first.add(key)
        unique_rows.append(flagged)

    daily_out: list[dict] = []
    scrape_by_day = {}
    for row in daily_rows:
        day = cell(row, "date")
        n_returned = int(cell(row, "n_returned") or 0)
        n_unique = int(cell(row, "n_unique") or 0)
        scrape_dup = n_returned - n_unique
        scrape_by_day[day] = {
            "n_returned": n_returned,
            "n_unique": n_unique,
            "n_kept": int(cell(row, "n_kept") or 0),
            "status": cell(row, "status"),
            "note": cell(row, "note"),
            "n_scrape_dup": scrape_dup,
            "n_scrape_unique": n_unique,
            "scrape_dup_minus_unique": scrape_dup - n_unique,
        }

    all_days = sorted(set(scrape_by_day) | set(day_keys) | set(dropped_by_day))
    for day in all_days:
        keys = day_keys.get(day, [])
        counts = Counter(keys)
        dup_titles = sum(1 for key, n in counts.items() if key_counts[key] > 1)
        once_titles = sum(1 for key, n in counts.items() if key_counts[key] == 1)
        scrape = scrape_by_day.get(
            day,
            {
                "n_returned": 0,
                "n_unique": 0,
                "n_kept": 0,
                "status": "",
                "note": "",
                "n_scrape_dup": 0,
                "n_scrape_unique": 0,
                "scrape_dup_minus_unique": 0,
            },
        )
        daily_out.append(
            {
                "date": day,
                **scrape,
                "n_numeric_dropped": dropped_by_day[day],
                "n_recovered": recovered_by_day[day],
                "n_articles": len(keys),
                "n_dup_titles": dup_titles,
                "n_once_titles": once_titles,
                "dup_minus_once": dup_titles - once_titles,
            }
        )

    out_dir: Path = args.out_dir
    headline_fields = [
        "date",
        "time",
        "title",
        "title_raw",
        "source",
        "url",
        "language",
        "sourcecountry",
        "is_duplicate",
    ]
    daily_fields = [
        "date",
        "n_returned",
        "n_unique",
        "n_kept",
        "status",
        "note",
        "n_scrape_dup",
        "n_scrape_unique",
        "scrape_dup_minus_unique",
        "n_numeric_dropped",
        "n_recovered",
        "n_articles",
        "n_dup_titles",
        "n_once_titles",
        "dup_minus_once",
    ]

    unique_path = out_dir / FILE_NAMES["headlines_clean"]
    dups_path = out_dir / FILE_NAMES["headlines_clean_dups"]
    train_path = out_dir / FILE_NAMES["headlines_train"]
    daily_path = out_dir / FILE_NAMES["news_daily_clean"]

    write_csv(unique_path, headline_fields, unique_rows)
    write_csv(dups_path, headline_fields, dup_rows)
    write_csv(train_path, headline_fields, dup_rows)
    write_csv(daily_path, daily_fields, daily_out)

    n_dup_keys = sum(1 for n in key_counts.values() if n > 1)
    n_once_keys = sum(1 for n in key_counts.values() if n == 1)
    print("=== clean headlines (English only) ===")
    print(f"raw rows: {len(raw_rows)}")
    print(f"pure-numeric titles in raw: {n_numeric_in}")
    print(f"titles recovered from URL: {n_recovered}")
    print(f"dropped encoding: {n_dropped_encoding}")
    print(f"dropped not-a-title/junk: {n_dropped_not_title}")
    print(f"dropped weak/unrelated: {n_dropped_unrelated}")
    print(f"dropped total: {n_dropped}")
    print(f"kept rows (duplicates retained): {len(dup_rows)}")
    print(f"unique titles: {len(unique_rows)}")
    print(f"duplicate title keys: {n_dup_keys}")
    print(f"once-only title keys: {n_once_keys}")
    print(f"Wrote {train_path}")
    print(f"Wrote {dups_path}")
    print(f"Wrote {unique_path}")
    print(f"Wrote {daily_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
