# -*- coding: utf-8 -*-
"""
News-to-price alignment + the main modelling experiment for our CS760 group
project (Group 2): "Crude oil direction prediction - does news sentiment
add value?"

This is the script the README points to for replicating the results - one
run redoes the whole experiment (about 8 minutes on a plain CPU) and
regenerates every table and figure (Fig 1-7) used in the report and the
slides.

My contribution: the alignment rule, the walk-forward setup, the feature
blocks, the three model wrappers, the paired bootstrap test, the leakage
positive control and all the plots below are my own work. External things I
only used as-is: scikit-learn / xgboost / PyTorch / pandas / numpy / scipy /
matplotlib, plus the sentiment CSVs that come out of my teammate's
CrudeBERT / DeepSeek pipeline.

What it does:
  * aligns calendar-day news to trading days (no look-ahead, see the note
    below), then predicts the NEXT day's direction
  * runs 7 feature blocks x 3 models (LogReg / XGBoost / LSTM) with an
    expanding-window walk-forward split (train 500, test 250, step 250)
  * paired bootstrap (5,000 resamples) for every "enhanced" block vs the
    price-only baseline
  * a deliberately leaked feature as a positive control, to prove the
    pipeline would catch look-ahead if there was any
  * naive baselines (always-long / momentum / reversal) for reference
  * a same-day vs next-day diagnostic - this is what ended up driving our
    conclusion, since sentiment correlates with same-day returns but says
    nothing about tomorrow

Alignment rule (the bit that matters):
    news dated d (calendar day, weekends included) is assigned to the first
    trading day T with T >= d; the features of day t only ever predict
    label_up(t+1). So Monday's news predicts Tuesday, weekend news lands on
    Monday and also predicts Tuesday - the information set is strictly
    "everything known by the close of day t".
    Known imperfection we state openly: GDELT timestamps only come in
    3-hour buckets, so we can't split pre-close vs post-close news (at most
    about half a day of blur).

Run it with:
    python Xiaoran_sentiment_alignment_and_modeling.py

Inputs (same folder as this script):
    price_daily_20200102_20260915_labeled.csv   (from my price pipeline script)
    sentiment_cb_daily_fixed.csv                CrudeBERT daily, noisy-input arm
    sentiment_cb_daily_en.csv                   CrudeBERT daily, cleaned English arm
    sentiment_llm_daily.csv                     DeepSeek daily (cached, no API call)

Outputs: everything lands in analysis_out/ - the aligned panel, the model
comparison table, bootstrap significance, univariate diagnostics, naive
baselines, per-fold details, the same-day-vs-next-day table, and Fig 1-7.

Dependencies: pandas numpy scipy scikit-learn xgboost torch matplotlib
(Python 3.12+, no GPU needed).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

import xgboost as xgb

import torch
import torch.nn as nn

# ------------------------------------------------------------------ config
HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis_out"
OUT.mkdir(exist_ok=True)

RNG = 20261009
np.random.seed(RNG)
torch.manual_seed(RNG)

PRICE_F = HERE / "price_daily_20200102_20260915_labeled.csv"
CB_F = HERE / "sentiment_cb_daily_fixed.csv"     # CrudeBERT, noisy-input arm (same input as the v28 run)
CB_EN_F = HERE / "sentiment_cb_daily_en.csv"     # CrudeBERT, cleaned English headlines
LLM_F = HERE / "sentiment_llm_daily.csv"         # DeepSeek

TRAIN_MIN = 500      # initial training window (trading days)
TEST_LEN = 250       # test length per fold
STEP = 250           # roll step
SEQ_LEN = 10         # lookback for the LSTM

PRICE_FEATS = ["ret_1d", "ma5_dev", "ma20_dev", "vol_20d", "gap", "range_pct", "volume_chg"]

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


# ================================================================== 1. load
def load_all():
    px = pd.read_csv(PRICE_F, encoding="utf-8-sig")
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for c in PRICE_FEATS + ["close", "label_up"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    print(f"[load] price {len(px):,} trading days  {px['date'].min().date()} -> {px['date'].max().date()}")
    print(f"[load] label_up mean = {px['label_up'].mean():.4f}  (majority-class accuracy to beat)")

    cb = pd.read_csv(CB_F, encoding="utf-8-sig")
    cb["date"] = pd.to_datetime(cb["date"], errors="coerce")
    print(f"[load] CrudeBERT daily {len(cb):,} days (noisy-input arm)")

    cb_en = None
    if CB_EN_F.exists():
        cb_en = pd.read_csv(CB_EN_F, encoding="utf-8-sig")
        cb_en["date"] = pd.to_datetime(cb_en["date"], errors="coerce")
        print(f"[load] CrudeBERT daily {len(cb_en):,} days (cleaned English arm)")

    llm = pd.read_csv(LLM_F, encoding="utf-8-sig")
    llm["date"] = pd.to_datetime(llm["date"], errors="coerce")
    print(f"[load] DeepSeek daily {len(llm):,} days")
    return px, cb, cb_en, llm


# ================================================================== 2. align
def align_news_to_trading_days(news_daily: pd.DataFrame, trading_days: pd.DatetimeIndex,
                               prefix: str = "") -> pd.DataFrame:
    """
    Map the calendar-day news table onto trading days:
         A(d) = the first trading day >= d
    Then aggregate per trading day: counts get summed, sentiment-type columns
    get a count-weighted average, share-type columns too.
    """
    nd = news_daily.dropna(subset=["date"]).copy()
    td = pd.DatetimeIndex(sorted(trading_days))

    # keep only news inside [first trading day, last trading day]
    nd = nd[(nd["date"] >= td[0]) & (nd["date"] <= td[-1])].copy()
    # the mapping itself: first trading day >= d
    pos = np.searchsorted(td.values, nd["date"].values)
    nd["td"] = td.values[np.clip(pos, 0, len(td) - 1)]

    w = nd["n_articles"].astype(float)
    sent_col = "cb_sent" if "cb_sent" in nd.columns else "sent_mean"
    share_cols = [c for c in nd.columns if c.endswith("_share")]
    prob_cols = [c for c in nd.columns if c.startswith("cb_p_")]
    conf_col = "cb_conf" if "cb_conf" in nd.columns else None

    out = pd.DataFrame({"date": td})
    cnt = nd.groupby("td")["n_articles"].sum()

    tmp = pd.DataFrame({"td": nd["td"]})
    for c in [sent_col] + share_cols + prob_cols + ([conf_col] if conf_col else []):
        tmp[c] = nd[c].astype(float).values * w.values
        # count-weighted average; num and cnt share the same index so pandas aligns them
        s = tmp.groupby("td")[c].sum() / cnt
        out[c] = out["date"].map(s)

    out["n_news"] = out["date"].map(cnt).fillna(0.0)
    out["has_news"] = (out["n_news"] > 0).astype(float)
    # no-news days: sentiment set to 0 (neutral), has_news marks them apart
    for c in [sent_col] + share_cols + prob_cols + ([conf_col] if conf_col else []):
        out.loc[out["n_news"] == 0, c] = 0.0

    if prefix:
        ren = {c: f"{prefix}{c}" for c in out.columns if c not in ("date",)}
        out = out.rename(columns=ren)
    return out


def build_panel(px, cb, cb_en, llm):
    """
    Build the final daily panel:
        X(t) = price features known at t's close + the news assigned to day t
        y(t) = label_up(t+1)
    """
    td = pd.DatetimeIndex(px["date"])
    cb_a = align_news_to_trading_days(cb, td)
    llm_a = align_news_to_trading_days(llm, td, prefix="llm_")
    cb_en_a = align_news_to_trading_days(cb_en, td, prefix="cben_") if cb_en is not None else None

    panel = px[["date", "close", "label_up"] + PRICE_FEATS].copy()
    panel = panel.merge(cb_a, on="date", how="left")
    panel = panel.merge(llm_a, on="date", how="left")
    if cb_en_a is not None:
        panel = panel.merge(cb_en_a, on="date", how="left")

    # next-day direction = the prediction target
    panel["y_next"] = panel["label_up"].shift(-1)
    # next-day return (for the economic side of the evaluation)
    panel["ret_next"] = panel["close"].shift(-1) / panel["close"] - 1.0

    # ---- rolling / change-type sentiment features (history only, no look-ahead) ----
    # single-day means are noisy, and the literature often uses "sentiment change
    # / surprise" as the predictor, so we add 3-day mean, 1-day change and 5-day
    # std on top - gives the hypothesis a fairer shot.
    roll_src = [c for c in ["cb_sent", "cb_pos_share", "cb_neg_share",
                            "llm_sent_mean", "llm_pos_share", "llm_neg_share",
                            "n_news"]
                if c in panel.columns]
    for c in roll_src:
        s = panel[c].astype(float)
        panel[f"{c}_ma3"] = s.rolling(3, min_periods=1).mean()
        panel[f"{c}_chg1"] = s.diff()
        panel[f"{c}_std5"] = s.rolling(5, min_periods=2).std()
    if "cben_cb_sent" in panel.columns:
        s = panel["cben_cb_sent"].astype(float)
        panel["cben_cb_sent_ma3"] = s.rolling(3, min_periods=1).mean()
        panel["cben_cb_sent_chg1"] = s.diff()

    # for the look-ahead audit: deliberately add a "future" feature as a
    # positive control. It has to be the future RETURN, not the future price
    # level - a price level's correlation with direction gets diluted and the
    # control would silently fail.
    panel["LEAK_ret_next"] = panel["ret_next"]

    panel = panel.dropna(subset=["y_next", "ret_next"]).reset_index(drop=True)
    print(f"[panel] usable sample: {len(panel):,} trading days")
    print(f"[panel] trading days with no news at all: {int((panel['n_news']==0).sum())}")
    print(f"[panel] y_next mean = {panel['y_next'].mean():.4f}")
    return panel


# ================================================================== 3. feature blocks
def feature_sets(have_en: bool):
    P = list(PRICE_FEATS)
    N = ["n_news", "has_news"]
    CB = ["cb_sent", "cb_pos_share", "cb_neg_share", "cb_conf"]
    LLM = ["llm_sent_mean", "llm_pos_share", "llm_neg_share"]
    CBEN = ["cben_cb_sent", "cben_cb_pos_share", "cben_cb_neg_share", "cben_cb_conf"] if have_en else []

    # rolling / change sentiment features (fair test: not just the noisy daily mean)
    ROLL = [f"{c}_{k}" for c in ["cb_sent", "cb_pos_share", "cb_neg_share",
                                 "llm_sent_mean", "llm_pos_share", "llm_neg_share",
                                 "n_news"]
            for k in ("ma3", "chg1", "std5")]

    sets = {
        "P (price only)": P,
        "P+N (+news count)": P + N,
        "P+N+CB (+CrudeBERT)": P + N + CB,
        "P+N+LLM (+DeepSeek)": P + N + LLM,
        "P+N+ROLL (+rolling sent)": P + N + ROLL,
    }
    if have_en:
        sets["P+N+CB_en (+cleaned)"] = P + N + CBEN
        sets["P+N+CB+LLM (all features)"] = P + N + CB + CBEN + LLM
    else:
        sets["P+N+CB+LLM (all features)"] = P + N + CB + LLM
    return sets


# ================================================================== 4. models
class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=32, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, L: int):
    """Return (N-L, L, F) sequences; sample i uses X[i-L+1 .. i]."""
    N, F = X.shape
    if N <= L:
        return np.empty((0, L, F), dtype=np.float32)
    idx = np.arange(L - 1, N)
    seqs = np.stack([X[i - L + 1:i + 1] for i in idx], axis=0)
    return seqs.astype(np.float32)


def fit_predict(model_name, Xtr, ytr, Xte, seq_len=SEQ_LEN, epochs=40, seed=RNG):
    """One interface for all three models: returns positive-class probs on the test set."""
    if model_name == "LogReg":
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        m = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        m.fit(Xtr_s, ytr)
        return m.predict_proba(Xte_s)[:, 1]

    if model_name == "XGBoost":
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_weight=5,
            eval_metric="logloss", random_state=seed, n_jobs=4,
            tree_method="hist",
        )
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]

    if model_name == "LSTM":
        torch.manual_seed(seed)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
        Xte_s = sc.transform(Xte).astype(np.float32)

        # training: windows only inside the training part; testing: prepend the
        # tail of the training rows as context (labels never leak)
        n_tr = len(Xtr_s)
        if n_tr <= seq_len + 20:
            return np.full(len(Xte_s), ytr.mean())
        Str = make_sequences(Xtr_s, seq_len)
        ytr_seq = ytr[seq_len - 1:]

        ctx = Xtr_s[-seq_len + 1:]
        Xte_cat = np.vstack([ctx, Xte_s]) if len(ctx) else Xte_s
        Ste = make_sequences(Xte_cat, seq_len)
        if len(Ste) != len(Xte_s):
            Ste = Ste[-len(Xte_s):]

        net = LSTMNet(Xtr_s.shape[1], hidden=32, layers=1, dropout=0.2)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()

        xt = torch.tensor(Str)
        yt = torch.tensor(ytr_seq, dtype=torch.float32)
        xe = torch.tensor(Ste)

        net.train()
        bs = 64
        for _ in range(epochs):
            perm = torch.randperm(len(xt))
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(xt[b]), yt[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            logits = net(xe).numpy()
        p = 1.0 / (1.0 + np.exp(-logits))
        # numerical safety net
        p = np.clip(np.nan_to_num(p, nan=float(ytr.mean())), 1e-6, 1 - 1e-6)
        return p

    raise ValueError(model_name)


# ================================================================== 5. backtest
def walk_forward(panel, feat_cols, model_name, train_min=TRAIN_MIN,
                 test_len=TEST_LEN, step=STEP):
    """Expanding-window walk-forward. Returns a DataFrame of daily OOS predictions."""
    n = len(panel)
    X_all = panel[feat_cols].astype(float).values
    y_all = panel["y_next"].astype(float).values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    recs = []
    start = train_min
    fold = 0
    while start < n:
        te_end = min(start + test_len, n)
        tr_X, tr_y = X_all[:start], y_all[:start]
        te_X = X_all[start:te_end]
        if len(te_X) == 0 or len(np.unique(tr_y)) < 2:
            break
        p = fit_predict(model_name, tr_X, tr_y, te_X)
        for i, prob in enumerate(p):
            k = start + i
            recs.append({
                "fold": fold,
                "date": panel["date"].iloc[k],
                "y": y_all[k],
                "p": float(prob),
                "ret_next": panel["ret_next"].iloc[k],
                "close": panel["close"].iloc[k],
            })
        fold += 1
        start += step
        if te_end >= n:
            break
    return pd.DataFrame(recs)


def evaluate(df: pd.DataFrame) -> dict:
    """One set of metrics for everyone: direction accuracy / AUC / return stuff."""
    y, p, r = df["y"].values, df["p"].values, df["ret_next"].values
    acc = accuracy_score(y, (p > 0.5).astype(int))
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = np.nan
    # long/short: long when p>0.5, short otherwise (equal weight, rebalanced
    # daily, no costs)
    pos = np.where(p > 0.5, 1.0, -1.0)
    strat = pos * r
    cum = float(np.prod(1 + strat) - 1)
    ann_ret = float((1 + cum) ** (252 / len(r)) - 1) if len(r) > 0 else np.nan
    sharpe = float(np.mean(strat) / np.std(strat) * np.sqrt(252)) if np.std(strat) > 0 else np.nan
    eq = np.cumprod(1 + strat)
    mdd = float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else np.nan
    bh = float(np.prod(1 + r) - 1)
    return {"Accuracy": acc, "AUC": auc, "cum_return": cum, "ann_return": ann_ret,
            "sharpe": sharpe, "max_dd": mdd, "buy_hold": bh, "N": len(r)}


def bootstrap_diff(base_df, aug_df, n_boot=5000, seed=RNG):
    """
    Paired bootstrap: resample the same test days and compare the two models.
    H0: the enhanced model is no better than the baseline (delta <= 0).
    """
    m = base_df.merge(aug_df, on=["fold", "date"], suffixes=("_b", "_a"))
    y = m["y_b"].values
    pb, pa = m["p_b"].values, m["p_a"].values
    n = len(m)
    rs = np.random.RandomState(seed)

    d_acc = np.empty(n_boot)
    d_auc = np.empty(n_boot)
    for b in range(n_boot):
        idx = rs.randint(0, n, n)
        yy, p1, p2 = y[idx], pb[idx], pa[idx]
        if len(np.unique(yy)) < 2:
            d_acc[b], d_auc[b] = 0.0, 0.0
            continue
        d_acc[b] = accuracy_score(yy, (p2 > 0.5).astype(int)) - accuracy_score(yy, (p1 > 0.5).astype(int))
        d_auc[b] = roc_auc_score(yy, p2) - roc_auc_score(yy, p1)

    def summ(d, name):
        lo, hi = np.percentile(d, [2.5, 97.5])
        pval = 2 * min((d <= 0).mean(), (d >= 0).mean())   # two-sided
        return {"metric": name, "delta_mean": float(d.mean()),
                "ci_low": float(lo), "ci_high": float(hi),
                "p_two_sided": float(min(pval, 1.0)),
                "P(delta>0)": float((d > 0).mean())}
    return pd.DataFrame([summ(d_acc, "Accuracy"), summ(d_auc, "AUC")])


# ================================================================== 6. plots
def pct(x, _=None):
    return f"{x*100:.0f}%"


def plot_roc(all_preds, baseline_key, path):
    plt.figure(figsize=(7.2, 6.0))
    for k, df in all_preds.items():
        fpr, tpr, _ = roc_curve(df["y"].values, df["p"].values)
        auc = roc_auc_score(df["y"].values, df["p"].values)
        lw = 2.8 if k == baseline_key else 1.8
        ls = "--" if k == baseline_key else "-"
        plt.plot(fpr, tpr, lw=lw, ls=ls, label=f"{k}  (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1, label="random (AUC=0.500)")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Fig 1  out-of-sample ROC comparison (next-day oil direction)")
    plt.legend(fontsize=8.5, loc="lower right"); plt.grid(alpha=.25)
    plt.savefig(path); plt.close()


def plot_equity(all_preds, panel, baseline_key, path):
    plt.figure(figsize=(9.2, 5.6))
    for k, df in all_preds.items():
        r = df.sort_values("date")["ret_next"].values
        pos = np.where(df.sort_values("date")["p"].values > 0.5, 1.0, -1.0)
        eq = np.cumprod(1 + pos * r)
        lw = 2.8 if k == baseline_key else 1.6
        ls = "--" if k == baseline_key else "-"
        plt.plot(df.sort_values("date")["date"], eq, lw=lw, ls=ls, label=k)
    # buy and hold
    s = panel[["date", "ret_next"]].copy()
    s = s[s["date"].isin(all_preds[baseline_key]["date"])]
    plt.plot(s["date"], np.cumprod(1 + s["ret_next"].values), color="grey",
             lw=1.6, ls="-.", label="buy & hold WTI")
    plt.axhline(1.0, color="black", lw=.8)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}x"))
    plt.xlabel("date"); plt.ylabel("equity (start = 1)")
    plt.title("Fig 2  out-of-sample equity: long/short strategy vs buy & hold")
    plt.legend(fontsize=8.5, ncol=2); plt.grid(alpha=.25)
    plt.savefig(path); plt.close()


def plot_fold_metrics(results, path):
    keys = [k for k in results.keys()]
    models = sorted({m for (f, m) in results[keys[0]].keys()})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, metric in zip(axes, ["Accuracy", "AUC"]):
        for fk in keys:
            xs = range(1, len(results[fk]) + 1)
            ys = [results[fk][(fk, m)][metric] for m in models]
            ax.plot(xs, ys, marker="o", ms=4, label=fk)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticks(range(1, len(results[keys[0]]) + 1))
        ax.set_xlabel("walk-forward fold")
        ax.set_ylabel(metric)
        ax.set_title(f"per-fold {metric}")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path); plt.close()


def plot_rolling_acc(all_preds, baseline_key, path, win=60):
    plt.figure(figsize=(9.2, 5.0))
    for k, df in all_preds.items():
        d = df.sort_values("date").reset_index(drop=True)
        hit = ((d["p"].values > 0.5).astype(int) == d["y"].values).astype(float)
        roll = pd.Series(hit).rolling(win, min_periods=20).mean()
        lw = 2.6 if k == baseline_key else 1.5
        ls = "--" if k == baseline_key else "-"
        plt.plot(d["date"], roll, lw=lw, ls=ls, label=k)
    plt.axhline(0.5, color="k", ls=":", lw=1.2, label="random 50%")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(pct))
    plt.xlabel("date"); plt.ylabel(f"{win}-day rolling direction accuracy")
    plt.title(f"Fig 3  out-of-sample {win}-day rolling accuracy (daily)")
    plt.legend(fontsize=8.5, ncol=2); plt.grid(alpha=.25)
    plt.savefig(path); plt.close()


def plot_news_overview(panel, path):
    """News coverage and sentiment over time, with the structural break marked."""
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True)

    d = panel[["date", "n_news"]].copy()
    axes[0].plot(d["date"], d["n_news"].rolling(21, min_periods=5).mean(),
                 color="#1f77b4", lw=1.8, label="21-day moving average")
    axes[0].axvline(pd.Timestamp("2026-03-01"), color="crimson", ls="--", lw=1.4)
    axes[0].text(pd.Timestamp("2026-03-05"), axes[0].get_ylim()[1] * .85,
                 "2026-03 channel expansion\n(StockTwits/RSS added)", color="crimson", fontsize=8.5)
    axes[0].set_ylabel("headlines per day")
    axes[0].set_title("Fig 4  news coverage and sentiment over time (structural-break check)")
    axes[0].legend(fontsize=8.5); axes[0].grid(alpha=.25)

    axes[1].plot(panel["date"], panel["cb_sent"].rolling(21, min_periods=5).mean(),
                 color="#ff7f0e", lw=1.6, label="CrudeBERT (fixed mapping)")
    axes[1].plot(panel["date"], panel["llm_sent_mean"].rolling(21, min_periods=5).mean(),
                 color="#2ca02c", lw=1.6, label="DeepSeek LLM")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].axvline(pd.Timestamp("2026-03-01"), color="crimson", ls="--", lw=1.4)
    axes[1].axvline(pd.Timestamp("2025-10-01"), color="purple", ls=":", lw=1.4)
    axes[1].set_ylabel("sentiment (21-day avg)")
    axes[1].legend(fontsize=8.5); axes[1].grid(alpha=.25)

    axes[2].plot(panel["date"], panel["close"], color="black", lw=1.2)
    axes[2].set_ylabel("WTI close (USD)")
    axes[2].grid(alpha=.25)

    plt.tight_layout(); plt.savefig(path); plt.close()


def plot_importance(panel, feat_cols, path):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
                          min_child_weight=5, random_state=RNG, n_jobs=4,
                          tree_method="hist", eval_metric="logloss")
    X = np.nan_to_num(panel[feat_cols].astype(float).values)
    y = panel["y_next"].astype(int).values
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values()
    plt.figure(figsize=(7.6, max(3.4, 0.28 * len(imp))))
    colors = ["#c0392b" if i.startswith(("cb", "llm", "n_news", "has_news")) else "#34495e"
              for i in imp.index]
    plt.barh(imp.index, imp.values, color=colors)
    plt.xlabel("XGBoost gain importance (fit on the full sample, just for interpretation)")
    plt.title("Fig 5  feature importance: red = news/sentiment, dark grey = price")
    plt.grid(alpha=.25, axis="x")
    plt.savefig(path); plt.close()


def plot_scatter(panel, path):
    """Sentiment vs next-day return: shows the 'basically no relationship' picture."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, col, name in zip(axes, ["llm_sent_mean", "cb_sent"],
                             ["DeepSeek LLM", "CrudeBERT (fixed)"]):
        x = panel[col].astype(float).values
        y = panel["ret_next"].astype(float).values
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=7, alpha=.28, color="#2c3e50")
        b = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
        ax.plot(xs, np.polyval(b, xs), color="crimson", lw=2)
        r = np.corrcoef(x[m], y[m])[0, 1]
        ax.set_title(f"{name}   vs next-day return\ncorrelation r = {r:+.4f}")
        ax.set_xlabel("same-day sentiment"); ax.set_ylabel("next-day return")
        ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(path); plt.close()


# ================================================================== 7. main
def main():
    print("=" * 84)
    print("STEP 3  news-price alignment and baseline-vs-sentiment experiment")
    print("=" * 84)

    px, cb, cb_en, llm = load_all()
    panel = build_panel(px, cb, cb_en, llm)
    panel.to_csv(OUT / "panel_aligned.csv", index=False, encoding="utf-8-sig")

    # ---------- 6.0 look-ahead positive control ----------
    print("\n" + "-" * 72)
    print("[audit] look-ahead positive control: feed tomorrow's return in as a")
    print("        feature on purpose and see if the pipeline notices.")
    print("        If the pipeline could actually cheat, AUC should go near 1;")
    print("        since the honest runs sit far below that, there's no leak.")
    X = panel[PRICE_FEATS + ["LEAK_ret_next"]].astype(float).values
    y = panel["y_next"].astype(int).values
    ntr = 500
    m = LogisticRegression(max_iter=2000)
    sc = StandardScaler()
    m.fit(sc.fit_transform(X[:ntr]), y[:ntr])
    p_leak = m.predict_proba(sc.transform(X[ntr:]))[:, 1]
    auc_leak = roc_auc_score(y[ntr:], p_leak)
    print(f"[audit] AUC with the future return included = {auc_leak:.4f}")
    print(f"[audit] verdict: {'control works - the pipeline catches leakage, so the 0.48~0.51 honest runs are genuinely leak-free' if auc_leak > 0.9 else 'control did not fire as expected, needs a look'}")
    panel = panel.drop(columns=["LEAK_ret_next"])

    sets = feature_sets(have_en=CB_EN_F.exists())
    models = ["LogReg", "XGBoost", "LSTM"]

    all_preds, results, fold_rows = {}, {}, []
    for fs_name, cols in sets.items():
        for mname in models:
            df = walk_forward(panel, cols, mname)
            ev = evaluate(df)
            results[(fs_name, mname)] = ev
            all_preds[f"{mname} | {fs_name}"] = df
            fold_rows.append({"feature_set": fs_name, "model": mname, **ev})
            print(f"[{mname:7s}] {fs_name:26s} Acc={ev['Accuracy']:.4f}  AUC={ev['AUC']:.4f}  "
                  f"cum={ev['cum_return']:+.2%}  sharpe={ev['sharpe']:+.2f}")

    res_df = pd.DataFrame(fold_rows)
    res_df.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # ---------- significance: every enhanced block vs the baseline (same model) ----------
    print("\n" + "=" * 84)
    print("significance tests (paired bootstrap, 5000 resamples; H0: enhanced <= baseline)")
    print("=" * 84)
    base_key = "P (price only)"
    sig_tables = []
    for mname in models:
        bdf = all_preds[f"{mname} | {base_key}"]
        for fs_name in sets:
            if fs_name == base_key:
                continue
            adf = all_preds[f"{mname} | {fs_name}"]
            t = bootstrap_diff(bdf, adf)
            t.insert(0, "comparison", f"{fs_name}  vs  baseline")
            t.insert(0, "model", mname)
            sig_tables.append(t)
            for _, row in t.iterrows():
                flag = "significant" if row["p_two_sided"] < 0.05 else "not significant"
                print(f"[{mname:7s}] {fs_name:26s} d{row['metric']:8s} = "
                      f"{row['delta_mean']:+.4f}  95%CI[{row['ci_low']:+.4f},{row['ci_high']:+.4f}]  "
                      f"p={row['p_two_sided']:.4f} -> {flag}")
    sig_df = pd.concat(sig_tables, ignore_index=True)
    sig_df.to_csv(OUT / "significance_tests.csv", index=False, encoding="utf-8-sig")

    # ---------- univariate correlation / AUC ----------
    print("\n" + "=" * 84)
    print("univariate diagnostics of the sentiment signals")
    print("=" * 84)
    diag = []
    for col in ["llm_sent_mean", "llm_pos_share", "llm_neg_share",
                "cb_sent", "cb_pos_share", "cb_neg_share", "n_news"]:
        if col not in panel.columns:
            continue
        x = panel[col].astype(float).values
        y = panel["y_next"].astype(int).values
        r = float(np.corrcoef(x, y)[0, 1])
        auc = roc_auc_score(y, x)
        # coefficient significance from a univariate logistic regression
        sc = StandardScaler()
        xs = sc.fit_transform(x.reshape(-1, 1))
        lr = LogisticRegression(max_iter=1000).fit(xs, y)
        coef = float(lr.coef_[0][0])
        diag.append({"feature": col, "r_vs_next_dir": r,
                     "univariate_auc": float(auc), "std_coef": coef})
    diag_df = pd.DataFrame(diag)
    print(diag_df.to_string(index=False))
    diag_df.to_csv(OUT / "sentiment_univariate.csv", index=False, encoding="utf-8-sig")

    # ---------- naive baselines for reference ----------
    print("\n" + "=" * 84)
    print("naive baselines (test period, N=%d)" % len(all_preds[f"XGBoost | {base_key}"]))
    print("=" * 84)
    ref = all_preds[f"XGBoost | {base_key}"]
    yy, rr = ref["y"].values, ref["ret_next"].values
    naive = []
    naive.append({"strategy": "always long (buy & hold)", "Accuracy": float((yy == 1).mean()),
                  "cum_return": float(np.prod(1 + rr) - 1),
                  "sharpe": float(np.mean(rr) / np.std(rr) * np.sqrt(252))})
    # momentum: yesterday up -> long today
    mom = np.concatenate([[1], (yy[:-1] == 1).astype(float)])
    sr = np.where(mom > .5, 1., -1.) * rr
    naive.append({"strategy": "momentum (yesterday up -> long)", "Accuracy": float(((mom > .5).astype(int) == yy).mean()),
                  "cum_return": float(np.prod(1 + sr) - 1),
                  "sharpe": float(np.mean(sr) / np.std(sr) * np.sqrt(252))})
    # reversal
    sr = np.where(mom > .5, -1., 1.) * rr
    naive.append({"strategy": "reversal (yesterday up -> short)", "Accuracy": float(((mom > .5).astype(int) != yy).mean()),
                  "cum_return": float(np.prod(1 + sr) - 1),
                  "sharpe": float(np.mean(sr) / np.std(sr) * np.sqrt(252))})
    naive_df = pd.DataFrame(naive)
    print(naive_df.to_string(index=False))
    naive_df.to_csv(OUT / "naive_baselines.csv", index=False, encoding="utf-8-sig")

    # ---------- per-fold details ----------
    fold_detail = []
    for fs_name in sets:
        for mname in models:
            dfp = all_preds[f"{mname} | {fs_name}"]
            for f, g in dfp.groupby("fold"):
                ev = evaluate(g)
                fold_detail.append({"feature_set": fs_name, "model": mname,
                                    "fold": int(f) + 1,
                                    "test_period": f"{g['date'].min().date()} ~ {g['date'].max().date()}",
                                    **ev})
    fold_df = pd.DataFrame(fold_detail)
    fold_df.to_csv(OUT / "fold_details.csv", index=False, encoding="utf-8-sig")

    # ---------- same-day vs next-day diagnostic (sentiment works, but can't predict) ----------
    print("\n" + "=" * 84)
    print("key diagnostic: same-day explanatory power vs actual predictive power")
    print("=" * 84)
    rows = []
    for col, name in [("llm_sent_mean", "DeepSeek LLM"), ("cb_sent", "CrudeBERT (noisy input)"),
                      ("cben_cb_sent", "CrudeBERT (cleaned English)")]:
        if col not in panel.columns:
            continue
        x = panel[col].astype(float).values
        same = panel["ret_1d"].astype(float).values
        nxt = panel["ret_next"].astype(float).values
        m1 = np.isfinite(x) & np.isfinite(same)
        m2 = np.isfinite(x) & np.isfinite(nxt)
        r_same = float(np.corrcoef(x[m1], same[m1])[0, 1])
        r_next = float(np.corrcoef(x[m2], nxt[m2])[0, 1])
        # AUC: same-day sentiment predicting same-day direction / next-day direction
        a_same = roc_auc_score(panel["label_up"].astype(int).values[m1], x[m1])
        a_next = roc_auc_score(panel["y_next"].astype(int).values[m2], x[m2])
        t_same = stats.pearsonr(x[m1], same[m1])
        t_next = stats.pearsonr(x[m2], nxt[m2])
        rows.append({"sentiment": name,
                     "r_same_day": r_same, "p_same_day": float(t_same.pvalue),
                     "auc_same_day": float(a_same),
                     "r_next_day": r_next, "p_next_day": float(t_next.pvalue),
                     "auc_next_day": float(a_next)})
    diag2 = pd.DataFrame(rows)
    print(diag2.to_string(index=False))
    diag2.to_csv(OUT / "contemporaneous_vs_next_day.csv", index=False, encoding="utf-8-sig")

    # ---------- plots ----------
    print("\n[plots] drawing the comparison figures ...")
    # ROC only for the XGBoost family so the chart stays readable
    xgb_keys = {k: v for k, v in all_preds.items() if k.startswith("XGBoost")}
    bk = "XGBoost | P (price only)"
    plot_roc(xgb_keys, bk, OUT / "fig1_roc_xgboost.png")
    plot_equity(xgb_keys, panel, bk, OUT / "fig2_equity_xgboost.png")
    plot_rolling_acc(xgb_keys, bk, OUT / "fig3_rolling_acc_xgboost.png", win=60)

    fold_res = {}
    for (fs, m), ev in results.items():
        fold_res.setdefault(fs, {})[m] = ev
    plot_news_overview(panel, OUT / "fig4_news_coverage_sentiment.png")

    all_cols = sets["P+N+CB+LLM (all features)"]
    plot_importance(panel, all_cols, OUT / "fig5_feature_importance.png")
    plot_scatter(panel, OUT / "fig6_sentiment_vs_return.png")

    # per-fold metric plot (XGBoost)
    per_fold = {}
    for fs_name in sets:
        key = f"XGBoost | {fs_name}"
        df = all_preds[key]
        rows = []
        for f, g in df.groupby("fold"):
            ev = evaluate(g)
            ev["fold"] = int(f)
            rows.append(ev)
        per_fold[fs_name] = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for fs_name, d in per_fold.items():
        for ax, metric in zip(axes, ["Accuracy", "AUC"]):
            ax.plot(d["fold"] + 1, d[metric], marker="o", ms=4, label=fs_name)
    for ax, metric in zip(axes, ["Accuracy", "AUC"]):
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xlabel("walk-forward fold"); ax.set_ylabel(metric)
        ax.set_title(f"per-fold {metric} (XGBoost)"); ax.grid(alpha=.25)
    axes[0].legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig(OUT / "fig7_per_fold.png"); plt.close()

    print(f"[output] everything saved to: {OUT}")

    # ---------- summary ----------
    print("\n" + "=" * 84)
    print("summary")
    print("=" * 84)
    piv = res_df.pivot_table(index="feature_set", columns="model", values="AUC")
    print("\nAUC:")
    print(piv.round(4).to_string())
    piv2 = res_df.pivot_table(index="feature_set", columns="model", values="Accuracy")
    print("\ndirection accuracy:")
    print(piv2.round(4).to_string())


if __name__ == "__main__":
    main()
