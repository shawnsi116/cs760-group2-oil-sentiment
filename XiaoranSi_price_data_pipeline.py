import os
import csv
from datetime import datetime, date

import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "price_work")
if not os.path.isdir(WORK):
    WORK = BASE

RAW_FILE = "prices_raw.csv"
OUT_FILE = "price_daily.csv"
CHART_FILE = "price_chart.png"
SUBSET_FILE = "price_daily_20250917_20260916.csv"
LABELED_FILE = "price_daily_20250917_20260916_labeled.csv"
INCLUDE_EXTRA = True
START = date(2025, 9, 17)
END = date(2026, 9, 16)
PATCH = {"2025-11-03": 236250, "2026-04-02": 552460}


# price_work/person1_price_pipeline.py
def download_prices():
    """主选 yfinance (CL=F)，失败自动切 stooq (cl.f)。"""
    try:
        import yfinance as yf
        df = yf.download("CL=F", period="max", interval="1d",
                         auto_adjust=False, progress=False)
        # 新版 yfinance 可能返回多级列名，统一拍平
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is not None and len(df) > 100:
            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            print(f"[ok] yfinance 下载 {len(df)} 行")
            return df
        raise RuntimeError("yfinance 返回数据过少")
    except Exception as e:
        print(f"[warn] yfinance 失败（{e}），改用 stooq")
        url = "https://stooq.com/q/d/l/?s=cl.f&i=d"
        df = pd.read_csv(url)
        df.columns = [c.lower() for c in df.columns]
        df = df[["date", "open", "high", "low", "close", "volume"]]
        print(f"[ok] stooq 下载 {len(df)} 行")
        return df


def clean(df):
    """日期规范化、去重排序、缺失收盘价列出再删。负油价（2020-04）保留。"""
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    missing = df[df["close"].isna()]["date"].dt.strftime("%Y-%m-%d").tolist()
    if missing:
        print(f"[warn] 收盘价缺失 {len(missing)} 天，已剔除并记录：{missing[:10]}")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def add_features(df):
    """接口四特征 + 可选附加特征。全部只用当天及之前的数据，不做 shift。"""
    df["ret_1d"] = df["close"].pct_change()
    df["ma5_dev"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["ma20_dev"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    if INCLUDE_EXTRA:
        df["gap"] = df["open"] / df["close"].shift(1) - 1
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
        df["volume_chg"] = df["volume"] / df["volume"].rolling(20).mean() - 1
    return df


def qa(df):
    """交付前质检：行数、区间、缺失、重复、NaN、特征口径抽查。"""
    print("\n===== 质检 =====")
    print(f"行数: {len(df)}  区间: {df['date'].min().date()} ~ {df['date'].max().date()}")
    bdays = pd.bdate_range(df["date"].min(), df["date"].max())
    missing = sorted(set(bdays) - set(df["date"]))
    print(f"相对工作日历缺失 {len(missing)} 天（节假日属正常），示例: "
          f"{[str(d.date()) for d in missing[:5]]}")
    nan_cols = df.columns[df.isna().any()].tolist()
    print(f"含 NaN 的列（rolling 预热期/volume 缺失属正常）: {nan_cols}")
    dup = int(df["date"].duplicated().sum())
    print(f"重复日期: {dup}")
    assert dup == 0, "有重复日期，回去查 clean 步骤"
    # 口径抽查：第 100 行 ma5_dev 手算应一致（只用当天及之前数据，无未来泄露）
    i = 100
    expect = df["close"].iloc[i] / df["close"].iloc[i - 4:i + 1].mean() - 1
    assert abs(df["ma5_dev"].iloc[i] - expect) < 1e-9, "ma5_dev 口径对不上"
    print("特征口径抽查通过（第 100 行 ma5_dev 手算一致）")


def save_chart(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(pd.to_datetime(out["date"]), out["close"], lw=0.8)
        plt.title("WTI front-month daily close")
        plt.tight_layout()
        plt.savefig(os.path.join(WORK, CHART_FILE), dpi=150)
        print(f"[done] 已导出 {CHART_FILE}（slides P3 用）")
    except Exception as e:
        print(f"[warn] 画图失败（不影响交付）：{e}")


def main_pipeline():
    df = download_prices()
    df.to_csv(os.path.join(WORK, RAW_FILE), index=False)
    print(f"[done] 原始数据已存 {RAW_FILE}")

    df = clean(df)
    df = add_features(df)
    qa(df)

    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(os.path.join(WORK, OUT_FILE), index=False, float_format="%.6f")
    print(f"\n[done] 已导出 {OUT_FILE}（{len(out)} 行，列: {list(out.columns)}）")
    save_chart(out)


# price_work/download_test.py
def download_test():
    import yfinance as yf
    df = yf.download("CL=F", period="max", interval="1d", progress=False)
    print("行数:", len(df))
    print(df.head(3))
    print(df.tail(3))


# price_work/subset_price_daily.py
DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]


def try_parse(s):
    s = s.strip().strip('"')
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    return None


def detect_date_col(header, first_row):
    # 1) 按列名匹配
    for i, name in enumerate(header):
        if name.strip().strip('"').lower() in ("date", "day", "trade_date", "datetime", "time", "timestamp"):
            return i
    # 2) 按内容匹配：第一行数据里能解析成日期的列
    for i, val in enumerate(first_row):
        if try_parse(val) is not None:
            return i
    raise SystemExit("没找到日期列，请人工检查文件表头")


def main_subset():
    src = os.path.join(WORK, OUT_FILE)
    dst = os.path.join(WORK, SUBSET_FILE)
    with open(src, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    col = detect_date_col(header, data[0])
    print(f"日期列自动识别为：第 {col + 1} 列 '{header[col]}'")

    kept = [r for r in data
            if r and any(c.strip() for c in r)
            and (d := try_parse(r[col])) is not None and START <= d <= END]

    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(kept)

    dates = [try_parse(r[col]) for r in kept]
    print(f"原文件数据行数: {len(data)}")
    print(f"保留行数: {len(kept)}")
    if dates:
        print(f"子集区间: {min(dates)} ~ {max(dates)}")
    print(f"已写出: {SUBSET_FILE}")


# price_work/add_label.py
def main_label():
    sub = pd.read_csv(os.path.join(WORK, SUBSET_FILE)).reset_index(drop=True)
    full = pd.read_csv(os.path.join(WORK, OUT_FILE))

    # ---- 1) 生成标签 ----
    full["date"] = pd.to_datetime(full["date"])
    prev_close_first = full.loc[full["date"] < "2025-09-17", "close"].iloc[-1]  # 2025-09-16 真实前收

    labels = []
    for i in range(len(sub)):
        prev_close = prev_close_first if i == 0 else sub.loc[i - 1, "close"]
        labels.append(1 if sub.loc[i, "close"] > prev_close else 0)
    sub["label_up"] = labels

    # 持平检查（规则：持平记 0；实测本窗口 0 天）
    ties = int((sub["close"].values[1:] == sub["close"].values[:-1]).sum())

    # ---- 2) volume==0 的处理：volume_chg 置空，原始列不动 ----
    zero_days = sub.loc[sub["volume"] == 0, "date"].tolist()
    sub.loc[sub["volume"] == 0, "volume_chg"] = pd.NA

    med_vol = int(sub["volume"].median())

    sub.to_csv(os.path.join(WORK, LABELED_FILE), index=False, float_format="%.6f")

    up, down = sum(labels), len(labels) - sum(labels)
    print(f"标签分布：涨={up} 跌={down}（共 {len(labels)} 行，涨占比 {up / len(labels) * 100:.1f}%）")
    print(f"首日 2025-09-17 标签 = {labels[0]}（母文件真实前收 2025-09-16 = {prev_close_first:.6f}）")
    print(f"持平天数：{ties}（规则约定：持平记 0）")
    print(f"窗口内成交量中位数：{med_vol} 手/天")
    print(f"volume==0 日期：{zero_days} → 这些天的 volume_chg 已置空，原始 volume 列保持原样")
    print(f"已写出：{LABELED_FILE}（{len(sub)} 行 × {len(sub.columns)} 列）")


# price_work/patch_volume.py
def main_patch():
    # 1) 母序列上补丁 volume，重算 volume_chg（20 日均值要用完整历史，且补丁会影响随后 19 天）
    full = pd.read_csv(os.path.join(WORK, OUT_FILE))
    for d, v in PATCH.items():
        full.loc[full["date"] == d, "volume"] = v
    full["volume_chg"] = full["volume"] / full["volume"].rolling(20).mean() - 1

    # 2) 更新 labeled 文件：打 volume 补丁 + 按日期整体替换 volume_chg
    # 注意：文件可能被外部编辑器另存为 xlsx（ZIP 头 PK），自动识别读取方式
    labeled_path = os.path.join(WORK, LABELED_FILE)
    with open(labeled_path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        print("[warn] labeled 文件被外部编辑器转成了 xlsx，按 Excel 读取并恢复为 CSV")
        lab = pd.read_excel(labeled_path)
    else:
        lab = pd.read_csv(labeled_path)
    # date 列统一回 YYYY-MM-DD 字符串（Excel 往返后可能变成 datetime）
    lab["date"] = pd.to_datetime(lab["date"]).dt.strftime("%Y-%m-%d")
    for d, v in PATCH.items():
        lab.loc[lab["date"] == d, "volume"] = v
    lab = lab.drop(columns=["volume_chg"]).merge(
        full[["date", "volume_chg"]], on="date", how="left")
    # 恢复列顺序（volume_chg 放回 label_up 之前）
    cols = list(lab.columns)
    cols.insert(cols.index("label_up"), cols.pop(cols.index("volume_chg")))
    lab = lab[cols]
    lab.to_csv(labeled_path, index=False, float_format="%.6f")

    print(lab[lab["date"].isin(PATCH)][["date", "volume", "volume_chg", "label_up"]].to_string(index=False))
    print("volume_chg 剩余 NaN:", int(lab["volume_chg"].isna().sum()))
    print("volume==0 剩余天数:", int((lab["volume"] == 0).sum()))
    print(f"已更新: {LABELED_FILE}（{len(lab)} 行 × {len(lab.columns)} 列）")


if __name__ == "__main__":
    download_test()
    main_pipeline()
    main_subset()
    main_label()
    main_patch()
