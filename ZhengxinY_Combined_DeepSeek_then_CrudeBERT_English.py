
# =====================================================================
# SECTION 1 — FILE 2 CODE: DEEPSEEK PIPELINE
# =====================================================================
import os
import re
import hashlib
import sqlite3
import asyncio
import warnings
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import requests
import aiohttp
import nest_asyncio
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
nest_asyncio.apply()

# =====================================================================
# API and core configuration
# =====================================================================
# Set DEEPSEEK_API_KEY in your environment before running; do not hard-code credentials.

@dataclass
class Config:
    raw_headlines_path: str = "headlines_clean_20200101_20260915_\u5168\u6e20\u9053_new.csv"
    output_dir: str = "person3_sentiment_outputs_v28"
    
    # Use a version-specific cache to avoid reusing results from older prompts.
    cache_db_path: str = "deepseek_sentiment_cache_v28.db" 
    
    crudebert_repo: str = "Captain-1337/CrudeBERT"
    deepseek_model: str = "deepseek-chat"
    deepseek_api_url: str = "https://api.deepseek.com/chat/completions"
    max_length: int = 96
    batch_size: int = 128 
    api_concurrency: int = 50
    api_timeout_seconds: int = 30
    api_retries: int = 8  # Increase retries to reduce failures that would otherwise fall back to Neutral.
    random_seed: int = 42

CFG = Config()
OUTPUT_DIR = Path(CFG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALID_LABELS = ("Positive", "Negative", "Neutral")

# =====================================================================
# Cache and API connectivity
# =====================================================================
def pre_flight_api_check():
    print("\n[API check] Verifying DeepSeek API connectivity...")
    headers = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}
    payload = {"model": CFG.deepseek_model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5}
    try:
        resp = requests.post(CFG.deepseek_api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            print("[API check] Validation passed; the service is reachable.\n")
        else:
            raise SystemExit(f"\nAPI request failed. HTTP status: {resp.status_code}")
    except Exception as e:
        raise SystemExit(f"\nNetwork connection failed: {str(e)}")

class LocalCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("CREATE TABLE IF NOT EXISTS cache (cache_key TEXT PRIMARY KEY, label TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    
    def get_many(self, keys: List[str]) -> Dict[str, str]:
        if not keys: return {}
        result = {}
        with sqlite3.connect(self.db_path) as conn:
            for start in range(0, len(keys), 999):
                chunk = keys[start:start + 999]
                placeholders = ",".join(["?"] * len(chunk))
                rows = conn.execute(f"SELECT cache_key, label FROM cache WHERE cache_key IN ({placeholders})", chunk).fetchall()
                result.update(dict(rows))
        return result

    def save_many(self, records: Dict[str, str]):
        if not records: return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("INSERT OR REPLACE INTO cache(cache_key, label, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)", list(records.items()))
            conn.commit()

# =====================================================================
# Probe test: infer the model's output-label mapping.
# =====================================================================
def infer_crudebert_mapping(model, tokenizer, device, config):
    raw_id2label = config.id2label
    idx2label = {}
    
    # 1. Try direct matching against label names.
    if any("pos" in str(v).lower() for v in raw_id2label.values()) and any("neg" in str(v).lower() for v in raw_id2label.values()):
        for k, v in raw_id2label.items():
            if "pos" in str(v).lower(): idx2label[int(k)] = "Positive"
            elif "neg" in str(v).lower(): idx2label[int(k)] = "Negative"
            else: idx2label[int(k)] = "Neutral"
        print(f"[Model Setup] Configuration labels parsed successfully: {idx2label}")
        return idx2label
        
    print("\n[Mapping check] The model config lacks clear label names; starting a probe benchmark...")
    # 2. Use representative headlines to probe the output mapping.
    probe_texts = [
        "OPEC+ agrees to massive oil production cuts, triggering a huge surge in crude prices.", # Expected Positive (bullish)
        "Global crude oil inventories build up massively amid severe economic recession.", # Expected Negative (bearish)
        "ExxonMobil holds its routine annual shareholder meeting today." # Expected Neutral
    ]
    
    inputs = tokenizer(probe_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        
    pos_idx, neg_idx, neu_idx = preds[0], preds[1], preds[2]
    
    if len(set([pos_idx, neg_idx, neu_idx])) == 3:
        idx2label = {pos_idx: "Positive", neg_idx: "Negative", neu_idx: "Neutral"}
        print(f"Probe calibration completed. Inferred mapping from the benchmark: {idx2label}\n")
    else:
        print("Probe results overlap; falling back to the conventional three-class mapping...")
        idx2label = {0: "Positive", 1: "Negative", 2: "Neutral"}
        
    return idx2label

# =====================================================================
# CrudeBERT engine
# =====================================================================
def load_crudebert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Loading CrudeBERT on {device.type.upper()}...")
    try:
        config = AutoConfig.from_pretrained(CFG.crudebert_repo)
        tokenizer = AutoTokenizer.from_pretrained(CFG.crudebert_repo)
        model = AutoModelForSequenceClassification.from_pretrained(CFG.crudebert_repo, config=config)
    except Exception:
        print("[INIT] Standard loading failed; attempting fallback loading...")
        try:
            config_path = hf_hub_download(repo_id=CFG.crudebert_repo, filename="config.json")
            model_path = hf_hub_download(repo_id=CFG.crudebert_repo, filename="pytorch_model.bin")
        except:
            config_path = hf_hub_download(repo_id=CFG.crudebert_repo, filename="crude_bert_config.json")
            model_path = hf_hub_download(repo_id=CFG.crudebert_repo, filename="crude_bert_model.bin")
        
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForSequenceClassification.from_config(config)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        # Remove an optional state-dict entry that may be absent from this model version.
        state_dict.pop("bert.embeddings.position_ids", None)
        model.load_state_dict(state_dict, strict=False)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model.to(device)
    model.eval()
    
    idx2label = infer_crudebert_mapping(model, tokenizer, device, config)
    return model, tokenizer, device, idx2label

def run_crudebert(df: pd.DataFrame):
    model, tokenizer, device, idx2label = load_crudebert()
    titles = df["title_clean"].tolist()
    
    labels, scores, max_probs = [], [], []
    pos_idx = next((k for k, v in idx2label.items() if v == "Positive"), -1)
    neg_idx = next((k for k, v in idx2label.items() if v == "Negative"), -1)

    with torch.inference_mode():
        for start in tqdm(range(0, len(titles), CFG.batch_size), desc="[CrudeBERT] Inference"):
            batch = titles[start:start + CFG.batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=CFG.max_length, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
            
            pred_indices = np.argmax(probs, axis=1)
            batch_max = np.max(probs, axis=1)
            
            # Calculate the sentiment score safely.
            if pos_idx != -1 and neg_idx != -1:
                batch_score = probs[:, pos_idx] - probs[:, neg_idx] 
            else:
                batch_score = np.zeros(len(batch))
            
            labels.extend([idx2label.get(int(i), "Neutral") for i in pred_indices])
            scores.extend(batch_score.tolist())
            max_probs.extend(batch_max.tolist())

    return {"label": labels, "score": scores, "max_prob": max_probs}

# =====================================================================
# DeepSeek inference engine (few-shot prompting; predictions are not manually edited).
# =====================================================================


def parse_llm(content: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', str(content), flags=re.DOTALL).strip().lower()
    # Remove extra symbols to make label parsing more robust.
    words = text.replace('"', ' ').replace("'", ' ').replace(".", " ").replace("=>", " ").replace(":", " ").split()
    
    # Search backwards to avoid being distracted by introductory wording such as "The label is...".
    for w in reversed(words):
        if "positive" in w: return "Positive"
        if "negative" in w: return "Negative"
        if "neutral" in w: return "Neutral"
    return "Neutral"

async def fetch_llm(session, semaphore, title, debug=False):
    payload = {
        "model": CFG.deepseek_model, 
        "temperature": 0.0,  # A temperature of 0.0 improves repeatability.
        "max_tokens": 10,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": f"Headline: {title}"}
        ]
    }
    for attempt in range(CFG.api_retries):
        try:
            async with semaphore:
                async with session.post(CFG.deepseek_api_url, headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}, json=payload, timeout=CFG.api_timeout_seconds) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw_output = data["choices"][0]["message"]["content"]
                        final_label = parse_llm(raw_output)
                        if debug:
                            print(f"[DEBUG API] Raw response: {raw_output.strip()} | Parsed label: {final_label} | Headline: {title[:40]}...")
                        return title, final_label
                    elif resp.status == 429:
                        await asyncio.sleep((1.5 ** attempt)) # Exponential backoff.
        except Exception:
            await asyncio.sleep(2)
            
    return title, "Neutral" 

async def run_deepseek(titles: List[str], cache_engine: LocalCache):
    unique = list(dict.fromkeys(titles))
    # Use a versioned cache key so results from older prompts are not reused.
    cache_keys = {t: hashlib.md5(f"v28_fewshot|{t}".encode()).hexdigest() for t in unique}
    existing = cache_engine.get_many(list(cache_keys.values()))
    
    results = {t: existing[k] for t, k in cache_keys.items() if k in existing}
    missing = [t for t in unique if t not in results]
    
    if missing:
        sem = asyncio.Semaphore(CFG.api_concurrency)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CFG.api_concurrency)) as session:
            debug_tasks = missing[:3]
            for t in debug_tasks:
                _, debug_label = await fetch_llm(session, sem, t, debug=True)
                results[t] = debug_label

            remaining = missing[3:]
            if remaining:
                tasks = [asyncio.create_task(fetch_llm(session, sem, t, debug=False)) for t in remaining]
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="[DeepSeek] API requests"):
                    t, l = await f
                    results[t] = l
                        
        cache_engine.save_many({cache_keys[t]: results[t] for t in missing if results.get(t) in VALID_LABELS})
    
    return [results.get(t, "Neutral") for t in titles]

# =====================================================================
# Model-agreement metrics. Agreement is not the same as accuracy.
# =====================================================================
def robust_academic_audit(df: pd.DataFrame):
    valid = df[(df["cb_label"].isin(VALID_LABELS)) & (df["llm_label"].isin(VALID_LABELS))]
    abs_baseline = (valid["cb_label"] == valid["llm_label"]).mean() * 100
    
    # Metric 1: directional agreement when neither model predicts Neutral.
    dir_subset_strict = valid[(valid["cb_label"] != "Neutral") & (valid["llm_label"] != "Neutral")]
    dir_agr_strict = (dir_subset_strict["cb_label"] == dir_subset_strict["llm_label"]).mean() * 100 if len(dir_subset_strict) > 0 else 0.0
    
    # Metric 2: agreement within the subset where CrudeBERT predicts a directional class.
    cb_directional = valid[valid["cb_label"] != "Neutral"]
    cb_dir_agr = (cb_directional["cb_label"] == cb_directional["llm_label"]).mean() * 100 if len(cb_directional) > 0 else 0.0

    print("\n" + "="*85)
    print("[Research note] Independent predictions have not been manually edited. Few-shot prompting changes DeepSeek instructions only.")
    print("="*85)
    print(f"1. Overall agreement (including Neutral, N={len(valid)}): {abs_baseline:.2f}%")
    print(f"2. Directional agreement (both models are non-neutral, N={len(dir_subset_strict)}): {dir_agr_strict:.2f}%")
    print(f"3. Agreement when CrudeBERT is non-neutral (N={len(cb_directional)}): {cb_dir_agr:.2f}%")
    
    for threshold in [0.80, 0.90]:
        subset = valid[valid["cb_max_prob"] >= threshold]
        if len(subset) > 0:
            agr = (subset["cb_label"] == subset["llm_label"]).mean() * 100
            print(f"4. High-confidence agreement (CrudeBERT confidence >= {threshold}): {agr:.2f}%")
            
    print("-" * 85)
    print("Interpretation: this comparison alone does not establish that few-shot prompting corrected CrudeBERT errors.")
    print("="*85 + "\n")
    return valid

# =====================================================================
# Safe CSV export helper (handles files locked by another application).
# =====================================================================
def safe_to_csv(df: pd.DataFrame, base_path: Path):
    try:
        df.to_csv(base_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {base_path.name}")
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        fallback_path = base_path.with_name(f"{base_path.stem}_safe_{timestamp}.csv")
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"Warning: {base_path.name} is locked; saved to an alternate file: {fallback_path.name}")

def aggregate_daily(df: pd.DataFrame, prefix: str):
    valid = df[df[f"{prefix}_label"].isin(VALID_LABELS)].copy()
    valid["is_neg"] = (valid[f"{prefix}_label"] == "Negative").astype(int)
    valid["is_pos"] = (valid[f"{prefix}_label"] == "Positive").astype(int)
    valid["is_neu"] = (valid[f"{prefix}_label"] == "Neutral").astype(int)
    
    agg_df = valid.groupby("date").agg(
        n_articles=("title_clean", "count"),
        sent_mean=(f"{prefix}_score", "mean"),
        neg_share=("is_neg", "mean"),
        pos_share=("is_pos", "mean"),
        neu_share=("is_neu", "mean")
    ).reset_index()
    
    agg_df = agg_df.sort_values(by="date").reset_index(drop=True)
    for col in ["sent_mean", "neg_share", "pos_share", "neu_share"]:
        agg_df[col] = agg_df[col].round(4)
        
    return agg_df

def main():
    print("[INIT] Pipeline V28 (few-shot prompt and mapping-probe version)...")
    pre_flight_api_check()
    
    df = pd.read_csv(CFG.raw_headlines_path)
    date_col = next(c for c in df.columns if 'date' in c.lower() or '\u65e5\u671f' in c)
    title_col = next(c for c in df.columns if 'title' in c.lower() or '\u6807\u9898' in c)
    
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df["title_clean"] = df[title_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["title_clean"].str.len() > 0].copy()

    # 1. Run CrudeBERT and infer its output-label mapping.
    cb_res = run_crudebert(df)
    df["cb_label"], df["cb_score"], df["cb_max_prob"] = cb_res["label"], cb_res["score"], cb_res["max_prob"]

    # 2. Run DeepSeek with the versioned few-shot prompt cache.
    loop = asyncio.get_event_loop()
    df["llm_label"] = loop.run_until_complete(run_deepseek(df["title_clean"].tolist(), LocalCache(CFG.cache_db_path)))
    
    df["llm_score"] = (df["llm_label"] == "Positive").astype(float) - (df["llm_label"] == "Negative").astype(float)

    # 3. Report agreement metrics.
    robust_academic_audit(df)
    
    # 4. Export outputs safely.
    cb_daily = aggregate_daily(df, "cb")
    safe_to_csv(cb_daily, OUTPUT_DIR / "sentiment_cb_daily.csv")
    
    llm_daily = aggregate_daily(df, "llm")
    safe_to_csv(llm_daily, OUTPUT_DIR / "sentiment_llm_daily.csv")
    
    safe_to_csv(df, OUTPUT_DIR / "news_sentiment_row_level.csv")

    print(f"\nCompleted. Full-run outputs were saved to {OUTPUT_DIR}/ directory.")

if __name__ == "__main__":
    main()


# =====================================================================
# SECTION 2 — FILE 1 CODE: STANDALONE CRUDEBERT DAILY PIPELINE
# =====================================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Edit these only if your actual input filename differs.
INPUT_CSV = Path("headlines_clean_20200101_20260915_\u5168\u6e20\u9053_new.csv")
OUTPUT_CSV = Path("person3_sentiment_outputs_crudebert") / "sentiment_cb_daily.csv"
MODEL_NAME = "Captain-1337/CrudeBERT"
TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 32
IDX_TO_LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}
LABELS = ["Positive", "Negative", "Neutral"]


def find_column(columns, exact_names, contains_names, description):
    cols = list(columns)
    for name in exact_names:
        if name in cols:
            return name
    lookup = {str(c).strip().lower(): c for c in cols}
    for name in exact_names:
        if name.strip().lower() in lookup:
            return lookup[name.strip().lower()]
    for col in cols:
        low = str(col).strip().lower()
        if any(part.lower() in low for part in contains_names):
            return col
    raise ValueError(f"Could not find {description} column. Available columns: {cols}")


def clean_title(value):
    """Remove Chinese translation text in parentheses; otherwise preserve headline text."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\([^()]*[\u4e00-\u9fff][^()]*\)", " ", text)
    text = re.sub(r"（[^（）]*[\u4e00-\u9fff][^（）]*）", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] device: {device}")
    print(f"[INIT] model: {MODEL_NAME}")
    # Use the model repository's standard Transformers loading path instead of
    # manually assembling config + state_dict. This avoids silently ignoring keys.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if int(getattr(model.config, "num_labels", -1)) != 3:
        raise RuntimeError(f"Expected 3 model classes; num_labels={getattr(model.config, 'num_labels', None)}")
    model.to(device)
    model.eval()
    print(f"[Label mapping] {IDX_TO_LABEL} (following the class_names order in the model-card example)")
    print(f"[Model config id2label] {getattr(model.config, 'id2label', None)}")
    return model, tokenizer, device


def predict(texts, model, tokenizer, device):
    labels, p_pos, p_neg, p_neu, max_probs, scores = [], [], [], [], [], []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), BATCH_SIZE), total=total_batches, desc="CrudeBERT inference"):
            batch = texts[start:start + BATCH_SIZE]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=MAX_LENGTH, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits.float()
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}; expected [batch, 3].")
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
                raise RuntimeError("Model probabilities contain NaN/Inf or the three probabilities do not sum to 1; saving stopped.")
            ids = probs.argmax(axis=1)
            labels.extend(IDX_TO_LABEL[int(i)] for i in ids)
            p_pos.extend(probs[:, 0].tolist())
            p_neg.extend(probs[:, 1].tolist())
            p_neu.extend(probs[:, 2].tolist())
            max_probs.extend(probs.max(axis=1).tolist())
            scores.extend((probs[:, 0] - probs[:, 1]).tolist())
    return {
        "label": labels, "p_positive": p_pos, "p_negative": p_neg,
        "p_neutral": p_neu, "max_prob": max_probs, "score": scores,
    }


def main():
    print("=" * 72)
    print("CrudeBERT natural predictions and daily aggregation (one CSV output)")
    print("No forced non-zero classes, no label edits, and no alignment to DeepSeek results.")
    print("=" * 72)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV.resolve()}\n"
            "Place the script and source CSV in the same directory, or update INPUT_CSV."
        )

    raw = pd.read_csv(INPUT_CSV, low_memory=False)
    date_col = find_column(raw.columns, ["date", "date（\u65e5\u671f）", "\u65e5\u671f"], ["date", "\u65e5\u671f"], "\u65e5\u671f")
    title_col = find_column(raw.columns, ["title", "title（\u6807\u9898）", "headline", "\u6807\u9898"], ["title", "headline", "\u6807\u9898"], "\u6807\u9898")
    print(f"[Data] Raw rows: {len(raw):,}")
    print(f"[Data] Date column: {date_col}; title column: {title_col}")

    df = raw[[date_col, title_col]].copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df["title_clean"] = df[title_col].map(clean_title)
    original_rows = len(df)
    df = df[df["date"].notna() & df["title_clean"].ne("") & df["title_clean"].str.lower().ne("nan")].copy()
    print(f"[Data] Valid headlines: {len(df):,}; excluded invalid dates/empty titles: {original_rows - len(df):,}")
    if df.empty:
        raise ValueError("No valid headlines found; execution stopped.")
    print(f"[Data] Date range: {df['date'].min()} to {df['date'].max()}")

    model, tokenizer, device = load_model()
    pred = predict(df["title_clean"].tolist(), model, tokenizer, device)
    for col, values in pred.items():
        df[col] = values

    # Report natural article-level results, including a class if its count is zero.
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    print("\n[Natural prediction distribution]")
    for label in LABELS:
        print(f"{label:>8}: {int(counts[label]):,} ({counts[label] / len(df) * 100:.2f}%)")
    print("\n[Mean class probabilities across all headlines]")
    print(f"Positive: {df['p_positive'].mean():.6f}")
    print(f"Negative: {df['p_negative'].mean():.6f}")
    print(f"Neutral : {df['p_neutral'].mean():.6f}")
    print(f"Mean maximum class probability: {df['max_prob'].mean():.6f}")
    print(f"Mean probability difference mean(P(Positive)-P(Negative)): {df['score'].mean():.6f}")

    daily = df.groupby("date", sort=True).agg(
        n_articles=("label", "size"),
        pos_count=("label", lambda s: int((s == "Positive").sum())),
        neg_count=("label", lambda s: int((s == "Negative").sum())),
        neu_count=("label", lambda s: int((s == "Neutral").sum())),
        sent_mean=("score", "mean"),
        p_positive_mean=("p_positive", "mean"),
        p_negative_mean=("p_negative", "mean"),
        p_neutral_mean=("p_neutral", "mean"),
        max_prob_mean=("max_prob", "mean"),
    ).reset_index()
    daily["pos_share"] = daily["pos_count"] / daily["n_articles"]
    daily["neg_share"] = daily["neg_count"] / daily["n_articles"]
    daily["neu_share"] = daily["neu_count"] / daily["n_articles"]
    daily = daily[[
        "date", "n_articles", "pos_count", "neg_count", "neu_count",
        "pos_share", "neg_share", "neu_share", "sent_mean",
        "p_positive_mean", "p_negative_mean", "p_neutral_mean", "max_prob_mean",
    ]]
    daily[daily.columns.drop("date")] = daily[daily.columns.drop("date")].round(6)

    if not (daily["pos_count"] + daily["neg_count"] + daily["neu_count"]).equals(daily["n_articles"]):
        raise RuntimeError("Daily class counts do not equal n_articles; saving stopped.")
    if not np.allclose(daily["pos_share"] + daily["neg_share"] + daily["neu_share"], 1.0, atol=2e-5):
        raise RuntimeError("Daily class shares do not sum to 1; saving stopped.")
    if int(daily["n_articles"].sum()) != len(df):
        raise RuntimeError("The total number of daily articles does not match the number of predictions; saving stopped.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Only output file] {OUTPUT_CSV.resolve()}")
    print(f"[Check] Number of dates: {len(daily):,} ; aggregated headlines: {int(daily['n_articles'].sum()):,}")
    print("[Check] Daily class counts equal n_articles; the three class shares sum to approximately 1.")
    print("Completed. Predictions were not manually changed; any class with zero natural predictions is retained as zero.")


if __name__ == "__main__":
    main()