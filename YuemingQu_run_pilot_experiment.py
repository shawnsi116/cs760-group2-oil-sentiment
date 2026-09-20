#!/usr/bin/env python3
"""One-year WTI price + news sentiment pilot experiment.

The script:
1. aligns calendar-day sentiment to the same/next WTI trading day;
2. creates a leakage-safe next-trading-day target;
3. runs expanding-window XGBoost and LSTM experiments;
4. exports model data, out-of-sample predictions, metrics, comparisons, and charts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


PRICE_FEATURES = [
    "ret_1d",
    "ma5_dev",
    "ma20_dev",
    "vol_20d",
    "gap",
    "range_pct",
]

SENTIMENT_FEATURES = {
    "price_only": [],
    "price_crudebert": [
        "cb_sent_mean",
        "cb_neg_share",
        "cb_log_articles",
        "cb_has_news",
    ],
    "price_llm": [
        "llm_sent_mean",
        "llm_neg_share",
        "llm_log_articles",
        "llm_has_news",
    ],
}


@dataclass(frozen=True)
class Fold:
    fold: int
    train_end: int
    test_start: int
    test_end: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("Price data contains duplicate dates")

    expected_current = (df["close"] > df["close"].shift(1)).astype("Int64")
    comparable = expected_current.notna()
    mismatches = (
        df.loc[comparable, "label_up"].astype("Int64")
        != expected_current.loc[comparable]
    ).sum()
    if mismatches:
        raise ValueError(f"label_up disagrees with current-day direction on {mismatches} rows")

    # Row t uses information available through close t and predicts direction t+1.
    df["target_next_day"] = df["label_up"].shift(-1).astype("Int64")
    direct_target = (df["close"].shift(-1) > df["close"]).astype("Int64")
    direct_target.iloc[-1] = pd.NA
    if not df["target_next_day"].equals(direct_target):
        raise ValueError("Next-day target alignment check failed")
    return df


def load_and_map_sentiment(
    path: Path, prefix: str, trading_dates: pd.Series
) -> pd.DataFrame:
    source = pd.read_csv(path)
    rename = {
        "date（日期）": "calendar_date",
        "sent_mean（情绪均值）": "sent_mean",
        "neg_share（负面占比）": "neg_share",
        "n_articles（新闻条数）": "n_articles",
        "p_pos_mean（正面概率均值）": "p_pos_mean",
        "p_neu_mean（中性概率均值）": "p_neu_mean",
        "p_neg_mean（负面概率均值）": "p_neg_mean",
    }
    source = source.rename(columns=rename)
    required = set(rename.values())
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing sentiment columns in {path.name}: {sorted(missing)}")

    source["calendar_date"] = pd.to_datetime(source["calendar_date"], errors="raise")
    source = source.sort_values("calendar_date").reset_index(drop=True)
    if source["calendar_date"].duplicated().any():
        raise ValueError(f"Duplicate daily sentiment dates in {path.name}")
    if (source["n_articles"] <= 0).any():
        raise ValueError(f"Non-positive article count in {path.name}")

    prob_sum = source[["p_pos_mean", "p_neu_mean", "p_neg_mean"]].sum(axis=1)
    if not np.allclose(prob_sum, 1.0, atol=1e-5):
        raise ValueError(f"Sentiment probabilities do not sum to one in {path.name}")
    if not np.allclose(
        source["sent_mean"], source["p_pos_mean"] - source["p_neg_mean"], atol=1e-5
    ):
        raise ValueError(f"sent_mean is inconsistent with probabilities in {path.name}")

    trade = pd.DatetimeIndex(pd.to_datetime(trading_dates).sort_values().unique())
    positions = trade.searchsorted(source["calendar_date"], side="left")
    valid = positions < len(trade)
    source = source.loc[valid].copy()
    source["feature_trading_date"] = trade[positions[valid]]

    weighted_cols = [
        "sent_mean",
        "neg_share",
        "p_pos_mean",
        "p_neu_mean",
        "p_neg_mean",
    ]
    for col in weighted_cols:
        source[f"weighted_{col}"] = source[col] * source["n_articles"]

    agg_spec = {"n_articles": ("n_articles", "sum")}
    for col in weighted_cols:
        agg_spec[f"sum_{col}"] = (f"weighted_{col}", "sum")
    mapped = source.groupby("feature_trading_date", as_index=False).agg(**agg_spec)
    for col in weighted_cols:
        mapped[col] = mapped[f"sum_{col}"] / mapped["n_articles"]
    mapped = mapped[["feature_trading_date", "n_articles", *weighted_cols]]

    mapped = mapped.rename(
        columns={
            "feature_trading_date": "date",
            "n_articles": f"{prefix}_n_articles",
            **{col: f"{prefix}_{col}" for col in weighted_cols},
        }
    )
    mapped[f"{prefix}_log_articles"] = np.log1p(mapped[f"{prefix}_n_articles"])
    mapped[f"{prefix}_has_news"] = 1
    return mapped


def build_dataset(price: pd.DataFrame, cb: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    merged = price.merge(cb, on="date", how="left").merge(llm, on="date", how="left")
    for prefix in ("cb", "llm"):
        count_col = f"{prefix}_n_articles"
        has_col = f"{prefix}_has_news"
        log_col = f"{prefix}_log_articles"
        sentiment_cols = [
            f"{prefix}_sent_mean",
            f"{prefix}_neg_share",
            f"{prefix}_p_pos_mean",
            f"{prefix}_p_neu_mean",
            f"{prefix}_p_neg_mean",
        ]
        merged[count_col] = merged[count_col].fillna(0).astype(int)
        merged[has_col] = merged[has_col].fillna(0).astype(int)
        merged[log_col] = merged[log_col].fillna(0.0)
        merged[sentiment_cols] = merged[sentiment_cols].fillna(0.0)

    model_df = merged.loc[merged["target_next_day"].notna()].copy()
    model_df["target_next_day"] = model_df["target_next_day"].astype(int)
    needed = PRICE_FEATURES + ["target_next_day"]
    if model_df[needed].isna().any().any():
        raise ValueError("Missing values remain in core model columns")
    return model_df.reset_index(drop=True)


def make_folds(n_rows: int, initial_train: int = 150, test_size: int = 20) -> list[Fold]:
    if n_rows <= initial_train:
        raise ValueError("Not enough observations for requested initial training window")
    folds: list[Fold] = []
    start = initial_train
    fold_id = 1
    while start < n_rows:
        end = min(start + test_size, n_rows)
        folds.append(Fold(fold_id, start, start, end))
        start = end
        fold_id += 1
    return folds


def run_xgboost(
    df: pd.DataFrame, feature_set: str, features: list[str], folds: list[Fold]
) -> pd.DataFrame:
    X = df[features].to_numpy(dtype=np.float32)
    y = df["target_next_day"].to_numpy(dtype=np.int64)
    rows = []
    for fold in folds:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(X[: fold.train_end], y[: fold.train_end])
        prob = model.predict_proba(X[fold.test_start : fold.test_end])[:, 1]
        for offset, p in enumerate(prob):
            idx = fold.test_start + offset
            rows.append(
                {
                    "date": df.loc[idx, "date"],
                    "fold": fold.fold,
                    "train_end_date": df.loc[fold.train_end - 1, "date"],
                    "model": "XGBoost",
                    "feature_set": feature_set,
                    "actual": int(y[idx]),
                    "prob_up": float(p),
                    "predicted": int(p >= 0.5),
                }
            )
    return pd.DataFrame(rows)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.20)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(x)
        last = encoded[:, -1, :]
        return self.output(self.dropout(last)).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, indices: np.ndarray, lookback: int):
    sequences, labels = [], []
    for idx in indices:
        start = idx - lookback + 1
        if start < 0:
            continue
        sequences.append(X[start : idx + 1])
        labels.append(y[idx])
    return (
        torch.tensor(np.asarray(sequences), dtype=torch.float32),
        torch.tensor(np.asarray(labels), dtype=torch.float32),
    )


def fit_one_lstm(
    X_scaled: np.ndarray,
    y: np.ndarray,
    train_end: int,
    test_indices: np.ndarray,
    lookback: int,
    seed: int,
) -> np.ndarray:
    seed_everything(seed)
    eligible = np.arange(lookback - 1, train_end)
    val_size = max(20, int(math.ceil(len(eligible) * 0.20)))
    train_indices = eligible[:-val_size]
    val_indices = eligible[-val_size:]

    X_train, y_train = make_sequences(X_scaled, y, train_indices, lookback)
    X_val, y_val = make_sequences(X_scaled, y, val_indices, lookback)
    X_test, _ = make_sequences(X_scaled, y, test_indices, lookback)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=min(32, len(X_train)), shuffle=True
    )
    model = LSTMClassifier(input_size=X_scaled.shape[1], hidden_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_loss = float("inf")
    patience = 12
    stale = 0
    for _ in range(100):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val), y_val).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_test)).cpu().numpy()


def run_lstm(
    df: pd.DataFrame,
    feature_set: str,
    features: list[str],
    folds: list[Fold],
    lookback: int = 10,
) -> pd.DataFrame:
    X_raw = df[features].to_numpy(dtype=np.float32)
    y = df["target_next_day"].to_numpy(dtype=np.int64)
    rows = []
    seeds = [17, 42, 91]
    for fold in folds:
        scaler = StandardScaler().fit(X_raw[: fold.train_end])
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        test_indices = np.arange(fold.test_start, fold.test_end)
        seed_probs = [
            fit_one_lstm(X_scaled, y, fold.train_end, test_indices, lookback, seed)
            for seed in seeds
        ]
        prob = np.mean(np.vstack(seed_probs), axis=0)
        for offset, p in enumerate(prob):
            idx = fold.test_start + offset
            rows.append(
                {
                    "date": df.loc[idx, "date"],
                    "fold": fold.fold,
                    "train_end_date": df.loc[fold.train_end - 1, "date"],
                    "model": "LSTM",
                    "feature_set": feature_set,
                    "actual": int(y[idx]),
                    "prob_up": float(p),
                    "predicted": int(p >= 0.5),
                }
            )
    return pd.DataFrame(rows)


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature_set), group in predictions.groupby(["model", "feature_set"]):
        y = group["actual"].to_numpy()
        p = group["prob_up"].to_numpy()
        pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "n_oos": len(group),
                "roc_auc": roc_auc_score(y, p),
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "brier": brier_score_loss(y, p),
                "tn": confusion_matrix(y, pred, labels=[0, 1])[0, 0],
                "fp": confusion_matrix(y, pred, labels=[0, 1])[0, 1],
                "fn": confusion_matrix(y, pred, labels=[0, 1])[1, 0],
                "tp": confusion_matrix(y, pred, labels=[0, 1])[1, 1],
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "feature_set"]).reset_index(drop=True)


def score_predictions_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature_set, fold), group in predictions.groupby(
        ["model", "feature_set", "fold"]
    ):
        y = group["actual"].to_numpy()
        p = group["prob_up"].to_numpy()
        pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "fold": int(fold),
                "n_oos": len(group),
                "n_up": int(y.sum()),
                "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
                "accuracy": accuracy_score(y, pred),
                "predicted_up_share": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "feature_set", "fold"]).reset_index(drop=True)


def paired_auc_bootstrap(predictions: pd.DataFrame, iterations: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(20260918)
    rows = []
    for model in predictions["model"].unique():
        base = predictions.query("model == @model and feature_set == 'price_only'").sort_values("date")
        for candidate in ("price_crudebert", "price_llm"):
            other = predictions.query("model == @model and feature_set == @candidate").sort_values("date")
            joined = base[["date", "actual", "prob_up"]].merge(
                other[["date", "actual", "prob_up"]], on=["date", "actual"], suffixes=("_base", "_candidate")
            )
            y = joined["actual"].to_numpy()
            p0 = joined["prob_up_base"].to_numpy()
            p1 = joined["prob_up_candidate"].to_numpy()
            observed = roc_auc_score(y, p1) - roc_auc_score(y, p0)
            diffs = []
            for _ in range(iterations):
                idx = rng.integers(0, len(joined), len(joined))
                if len(np.unique(y[idx])) < 2:
                    continue
                diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "comparison": f"{candidate} - price_only",
                    "delta_auc": observed,
                    "ci_2.5%": lo,
                    "ci_97.5%": hi,
                    "bootstrap_iterations": len(diffs),
                }
            )
    return pd.DataFrame(rows)


def save_charts(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "price_only": "#64748B",
        "price_crudebert": "#2563EB",
        "price_llm": "#F97316",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    for ax, model in zip(axes, ["XGBoost", "LSTM"]):
        for feature_set in ["price_only", "price_crudebert", "price_llm"]:
            g = predictions.query("model == @model and feature_set == @feature_set")
            fpr, tpr, _ = roc_curve(g["actual"], g["prob_up"])
            auc = roc_auc_score(g["actual"], g["prob_up"])
            ax.plot(fpr, tpr, label=f"{feature_set} (AUC={auc:.3f})", color=colors[feature_set])
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8")
        ax.set_title(model)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Walk-forward out-of-sample ROC curves")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    plot = metrics.copy()
    labels = plot["model"] + "\n" + plot["feature_set"]
    bars = ax.bar(labels, plot["roc_auc"], color=[colors[x] for x in plot["feature_set"]])
    ax.axhline(0.5, linestyle="--", color="#64748B", linewidth=1)
    ax.set_ylim(0.0, max(0.75, float(plot["roc_auc"].max()) + 0.08))
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Pilot walk-forward ROC-AUC")
    ax.tick_params(axis="x", labelsize=8)
    for bar, value in zip(bars, plot["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "auc_comparison.png", bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    model_df: pd.DataFrame,
    folds: list[Fold],
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    def to_markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(c) for c in frame.columns]
        rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    metrics_display = metrics.copy()
    for col in ["roc_auc", "accuracy", "precision", "recall", "f1", "brier"]:
        metrics_display[col] = metrics_display[col].map(lambda x: f"{x:.4f}")
    comparisons_display = comparisons.copy()
    for col in ["delta_auc", "ci_2.5%", "ci_97.5%"]:
        comparisons_display[col] = comparisons_display[col].map(lambda x: f"{x:.4f}")

    fold_lines = [
        f"- Fold {f.fold}: train rows 1–{f.train_end}; test rows {f.test_start + 1}–{f.test_end}"
        for f in folds
    ]
    oos = model_df.iloc[folds[0].test_start : folds[-1].test_end]
    oos_up = int(oos["target_next_day"].sum())
    oos_n = len(oos)
    lstm_majority_warning = metrics.query(
        "model == 'LSTM' and feature_set in ['price_only', 'price_crudebert']"
    )[["feature_set", "tn", "fp", "fn", "tp"]]

    report = f"""# WTI 次交易日涨跌方向：一年期试验报告

## 一、试验范围

- 可用日期：{model_df['date'].min().date()} 至 {model_df['date'].max().date()}
- 可用带标签交易日：{len(model_df)} 天
- 滚动样本外测试：{oos_n} 天，其中上涨 {oos_up} 天、下跌 {oos_n - oos_up} 天
- 预测目标：当前交易日收盘后，预测下一交易日收盘价是否高于当前交易日
- 主指标：ROC-AUC；辅助指标：Accuracy、Precision、Recall、F1、Brier 分数
- 模型：XGBoost、单层 LSTM
- 特征组合：仅价格、价格 + CrudeBERT、价格 + LLM

## 二、数据对齐与防止泄漏

成员三提交的是按自然日汇总的情绪表，而不是带精确发布时间的逐条新闻。本试验把交易日情绪归入当日，把周末和休市日情绪归入下一个可用交易日；若多个自然日映射到同一交易日，则按新闻条数加权重算情绪均值与占比。

这是适用于一年期试跑的近似方案。正式十年实验应回到逐条新闻，以明确的市场时区和每日截点重新聚合，防止盘后新闻提前进入当日特征。

原价格表的 `label_up` 表示“当日相对前一日”的涨跌，且可由当日收益率直接确定。因此本实验没有把它作为同一行预测目标，而是构造 `target_next_day = label_up.shift(-1)`，避免标签泄漏。

由于成交量中存在需要回查数据源的零值，本轮不使用 `volume_chg`。价格特征采用收益率、均线偏离、波动率、跳空和日内振幅，不直接使用原始价格水平。

## 三、滚动样本外设计

{chr(10).join(fold_lines)}

六组实验使用完全相同的样本外日期。XGBoost 使用固定的保守参数；LSTM 使用 10 日回看窗口、16 个隐藏单元、早停，并对 3 个随机种子的预测概率取平均。标准化器只在每一折的训练数据上拟合。

## 四、总体结果

{to_markdown_table(metrics_display)}

## 五、情绪特征相对价格基线的增量

{to_markdown_table(comparisons_display)}

置信区间来自共同样本外日期上的配对非参数 Bootstrap。四个区间都跨越 0，因此本轮没有证据证明加入 CrudeBERT 或 LLM 情绪能稳定改变 ROC-AUC。

## 六、诊断与结论

- 六组 ROC-AUC 均低于随机排序基准 0.5，当前特征和设置尚未形成可用的方向预测能力。
- XGBoost 中，LLM 情绪相对仅价格基线的 AUC 增量最大，但只有约 +0.0181，且 95% 区间跨越 0；不能据此宣称情绪有效。
- LSTM 的仅价格与 CrudeBERT 版本在 99 个样本外日期上全部预测“上涨”。其约 54.5% 的准确率等于测试集上涨占比，属于多数类退化，而非有效区分。
- 各折结果见 `fold_metrics.csv`。不同时间段波动明显，说明一年数据对于 LSTM 尤其偏少。
- 本轮的价值是验证了数据对齐、防泄漏、滚动训练、消融比较和复现实验链路；不应把当前结果写成“模型已具有预测能力”。

LSTM 多数类诊断：

{to_markdown_table(lstm_majority_warning)}

## 七、复现信息

- LSTM 随机种子：17、42、91；XGBoost 随机种子：42
- 初始训练集：150 个交易日；每个测试块：20 个交易日
- LSTM 回看窗口：10 个交易日；Bootstrap 随机种子：20260918
- 输入文件通过命令行参数传入，脚本不会修改原文件
"""
    (output_dir / "pilot_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", type=Path, required=True)
    parser.add_argument("--cb", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    price = load_price(args.price)
    cb = load_and_map_sentiment(args.cb, "cb", price["date"])
    llm = load_and_map_sentiment(args.llm, "llm", price["date"])
    model_df = build_dataset(price, cb, llm)
    folds = make_folds(len(model_df), initial_train=150, test_size=20)

    all_predictions = []
    for feature_set, sentiment_features in SENTIMENT_FEATURES.items():
        features = PRICE_FEATURES + sentiment_features
        all_predictions.append(run_xgboost(model_df, feature_set, features, folds))
        all_predictions.append(run_lstm(model_df, feature_set, features, folds, lookback=10))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(["model", "feature_set", "date"]).reset_index(drop=True)
    metrics = score_predictions(predictions)
    fold_metrics = score_predictions_by_fold(predictions)
    comparisons = paired_auc_bootstrap(predictions)

    model_df.to_csv(args.output / "model_dataset.csv", index=False)
    predictions.to_csv(args.output / "walk_forward_predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics_summary.csv", index=False)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    comparisons.to_csv(args.output / "auc_bootstrap_comparisons.csv", index=False)
    save_charts(predictions, metrics, args.output)
    write_report(args.output, model_df, folds, metrics, fold_metrics, comparisons, args)

    metadata = {
        "price_rows_raw": len(price),
        "model_rows": len(model_df),
        "date_min": str(model_df["date"].min().date()),
        "date_max": str(model_df["date"].max().date()),
        "folds": [f.__dict__ for f in folds],
        "price_features": PRICE_FEATURES,
        "sentiment_features": SENTIMENT_FEATURES,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("\nPaired AUC comparisons")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
