# -*- coding: utf-8 -*-
"""
Price data pipeline for our CS760 group project (Group 2):
"Crude oil direction prediction - does news sentiment add value?"

This script builds the price dataset everyone else in the group builds on:
price_daily_20200102_20260915_labeled.csv (1,686 trading days x 14 columns),
daily WTI front-month futures from 2020-01-02 to 2026-09-15, with OHLCV,
7 technical features and the next-day direction label `label_up`.

My contribution: the whole pipeline below is my own work - the download with
a fallback source, cleaning, the 7 features, the QA checks, two volume-gap
patches I verified by hand, and the labelling. External stuff I only used,
never modified: Yahoo Finance (via yfinance) and stooq for the raw prices,
the two volume values cross-checked on investing.com, and the usual open
source libs (pandas / numpy / yfinance / matplotlib).

How it works, in order:
  1. download  - yfinance first, falls back to stooq if that fails; raw dump
                 goes to prices_raw.csv
  2. clean     - normalise dates, drop duplicates, drop rows with no close
                 (the negative closes in Apr 2020 are real history, we keep them)
  3. features  - ret_1d / ma5_dev / ma20_dev / vol_20d / gap / range_pct /
                 volume_chg, all computed from today and earlier only (no shift,
                 the lagging is handled later when we merge with the news side)
  4. QA + export price_daily.csv (full history, 6 decimals)
  5. patches   - two days where Yahoo reported volume = 0 (VOL_PATCH) and one
                 row on the last day that was an intraday snapshot, not the
                 settled close (ROW_FIX)
  6. recompute all features on the full series first, THEN slice the window
     (so the rolling windows don't get weird at the window edges)
  7. slice 2020-01-01 .. 2026-09-15
  8. label_up = 1 if close(t) > close(t-1) else 0, first row left blank
  9. export + a bunch of self-checks printed to the console

Dependencies: pandas and numpy are required; yfinance and matplotlib are
optional (no yfinance -> stooq fallback kicks in automatically, and a missing
matplotlib only skips the bonus price chart).

Run it with:
    python Xiaoran_price_pipeline_20200102_20260915.py

The path constants are near the top, tweak them if your folders differ.
"""
import os
import numpy as np
import pandas as pd

# ---------------- paths ----------------
RAW_FILE = "prices_raw.csv"          # raw download, same folder
OUT_FILE = "price_daily.csv"         # full-history feature table
CHART_FILE = "price_chart.png"       # bonus close-price chart for the slides

SRC = r"D:\Study\S3\CS760\pre2\price_work\price_daily.csv"
REF = r"D:\Study\S3\CS760\pre2\price_work\price_daily_20250917_20260916_labeled.csv"
OUT_DIR = r"D:\Study\S3\CS760\Final"
OUT = os.path.join(OUT_DIR, "price_daily_20200102_20260915_labeled.csv")

COLS = ["date", "open", "high", "low", "close", "volume",
        "ret_1d", "ma5_dev", "ma20_dev", "vol_20d", "gap", "range_pct", "volume_chg", "label_up"]

START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2026-09-15")

# extra features switch: gap / range_pct / volume_chg
# the group agreed to include them; flip to False if we ever need the lean version
INCLUDE_EXTRA = True

# volume gaps Yahoo left at zero; real values checked by hand on investing.com
VOL_PATCH = {"2025-11-03": 236250, "2026-04-02": 552460}
# the last row was an intraday snapshot rather than the settled close,
# re-checked against Yahoo Finance and fixed here
ROW_FIX = {"2026-09-15": {"open": 102.010002, "high": 106.75, "low": 101.209999,
                          "close": 105.830002, "volume": 371490}}


# ================= 1. download =================
def download_prices():
    """Try yfinance (CL=F) first, fall back to stooq (cl.f) if it misbehaves."""
    try:
        import yfinance as yf
        df = yf.download("CL=F", period="max", interval="1d",
                         auto_adjust=False, progress=False)
        # newer yfinance sometimes returns multi-level columns, flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is not None and len(df) > 100:
            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            print(f"[ok] yfinance returned {len(df)} rows")
            return df
        raise RuntimeError("yfinance returned too few rows")
    except Exception as e:
        print(f"[warn] yfinance failed ({e}), switching to stooq")
        url = "https://stooq.com/q/d/l/?s=cl.f&i=d"
        df = pd.read_csv(url)
        df.columns = [c.lower() for c in df.columns]
        df = df[["date", "open", "high", "low", "close", "volume"]]
        print(f"[ok] stooq returned {len(df)} rows")
        return df


# ================= 2. clean =================
def clean(df):
    """Normalise dates, dedupe, drop missing closes (we list them first).
    The negative oil price days in Apr 2020 stay in - that really happened."""
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    missing = df[df["close"].isna()]["date"].dt.strftime("%Y-%m-%d").tolist()
    if missing:
        print(f"[warn] {len(missing)} days with missing close, dropped: {missing[:10]}")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# ================= 3. features =================
def add_features(df):
    """The four agreed features plus the optional extras.
    Everything uses today's data and earlier only - no shift anywhere."""
    df["ret_1d"] = df["close"].pct_change()
    df["ma5_dev"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["ma20_dev"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    if INCLUDE_EXTRA:
        df["gap"] = df["open"] / df["close"].shift(1) - 1
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
        df["volume_chg"] = df["volume"] / df["volume"].rolling(20).mean() - 1
    return df


# ================= 4. QA + chart =================
def qa(df):
    """Quick sanity pass before handing the table over: row count, date range,
    missing days, duplicates, NaN columns, and a hand-check of one feature."""
    print("\n===== QA =====")
    print(f"rows: {len(df)}  range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    bdays = pd.bdate_range(df["date"].min(), df["date"].max())
    missing = sorted(set(bdays) - set(df["date"]))
    print(f"{len(missing)} business days missing vs the calendar (holidays are fine), "
          f"e.g. {[str(d.date()) for d in missing[:5]]}")
    nan_cols = df.columns[df.isna().any()].tolist()
    print(f"columns with NaN (rolling warm-up / missing volume, that's normal): {nan_cols}")
    dup = int(df["date"].duplicated().sum())
    print(f"duplicate dates: {dup}")
    assert dup == 0, "found duplicate dates, go back to the clean step"
    # spot-check: hand-compute ma5_dev for row 100, should match exactly
    # (only today's data and earlier - no peeking at the future)
    i = 100
    expect = df["close"].iloc[i] / df["close"].iloc[i - 4:i + 1].mean() - 1
    assert abs(df["ma5_dev"].iloc[i] - expect) < 1e-9, "ma5_dev doesn't match the hand calc"
    print("feature spot-check passed (ma5_dev on row 100 matches the hand calculation)")


def save_chart(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(pd.to_datetime(out["date"]), out["close"], lw=0.8)
        plt.title("WTI front-month daily close")
        plt.tight_layout()
        plt.savefig(CHART_FILE, dpi=150)
        print(f"[done] chart saved to {CHART_FILE} (for the slides)")
    except Exception as e:
        print(f"[warn] chart failed (not a big deal for the deliverable): {e}")


# ================= 5. full-history table price_daily.csv =================
def main_price_pipeline():
    df = download_prices()
    df.to_csv(RAW_FILE, index=False)
    print(f"[done] raw data saved to {RAW_FILE}")

    df = clean(df)
    df = add_features(df)
    qa(df)

    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_FILE, index=False, float_format="%.6f")
    print(f"\n[done] exported {OUT_FILE} ({len(out)} rows, columns: {list(out.columns)})")
    save_chart(out)


# ================= 6. final dataset (patch -> window -> labels -> export) =================
def main_final_dataset():
    df = pd.read_csv(SRC)
    df["date"] = pd.to_datetime(df["date"])
    orig = df.copy()

    # 1) apply the patches / fixes, and keep a record of what changed
    changes = []
    for d, v in VOL_PATCH.items():
        for ix in df.index[df["date"] == pd.Timestamp(d)]:
            old = int(df.loc[ix, "volume"])
            if old != v:
                changes.append((d, "volume", old, v, "investing.com"))
            df.loc[ix, "volume"] = v
    for d, vals in ROW_FIX.items():
        for ix in df.index[df["date"] == pd.Timestamp(d)]:
            for k, v in vals.items():
                old = float(df.loc[ix, k])
                if abs(old - float(v)) > 1e-9:
                    changes.append((d, k, round(old, 6), v, "Yahoo Finance re-check"))
                df.loc[ix, k] = v

    # 2) recompute the features (same definitions as price_daily.csv;
    #    compute on the full series first, slice the window afterwards)
    df["ret_1d"] = df["close"].pct_change()
    df["ma5_dev"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["ma20_dev"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["gap"] = df["open"] / df["close"].shift(1) - 1
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["volume_chg"] = df["volume"] / df["volume"].rolling(20).mean() - 1

    # 3) slice the window
    w = df[(df["date"] >= START) & (df["date"] <= END)].copy()

    # 4) label_up: 1 if today closed above the previous trading day, else 0;
    #    the first row has no previous day so it stays blank
    lab = (w["close"].values[1:] > w["close"].values[:-1]).astype(float)
    label_col = np.concatenate([[np.nan], lab])
    w["label_up"] = label_col
    w.iloc[0, w.columns.get_loc("label_up")] = np.nan

    out = w.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["label_up"] = out["label_up"].apply(lambda x: "" if pd.isna(x) else str(int(x)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out[COLS].to_csv(OUT, index=False, float_format="%.6f", encoding="utf-8")

    # ---------- self-checks ----------
    print("=== patches applied ===")
    for c in changes:
        print("  ", c)
    print()
    new = pd.read_csv(OUT)
    print("=== self-check ===")
    print("rows:", len(new), "| range:", new["date"].iloc[0], "~", new["date"].iloc[-1])
    print("no BOM:", open(OUT, "rb").read(3) == b"dat")
    print("first row label_up blank:", new["label_up"].iloc[0] == "" or pd.isna(new["label_up"].iloc[0]))
    print("label_up values:", sorted(set(new["label_up"].iloc[1:].tolist())))
    print("label counts: 1 =", int((new["label_up"] == "1").sum()), " 0 =", int((new["label_up"] == "0").sum()))
    # decimal check: just look at the raw text of a few fields on line 2
    line2 = open(OUT, encoding="utf-8").read().split("\n")[1].split(",")
    print("line 2 fields:", line2)
    print("all 6 decimals:", all(len(v.split(".")[1]) == 6 for v in line2[1:8] if "." in v))
    # did the patches actually land?
    print("2025-11-03 volume =", int(new.loc[new["date"] == "2025-11-03", "volume"].iloc[0]))
    print("2026-04-02 volume =", int(new.loc[new["date"] == "2026-04-02", "volume"].iloc[0]))
    last = new.iloc[-1]
    print("last row:", last["date"], "close =", last["close"], "volume =", int(last["volume"]))
    print("volume==0 days left in the window:", int((new["volume"] == 0).sum()))
    print("output path:", OUT)
    if os.path.exists(REF):
        ref = pd.read_csv(REF)
        print("columns match the reference file:", list(new.columns) == list(ref.columns))


if __name__ == "__main__":
    main_price_pipeline()
    main_final_dataset()
