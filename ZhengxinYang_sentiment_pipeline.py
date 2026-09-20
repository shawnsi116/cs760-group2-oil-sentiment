import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("🚀 [0/5] Deep Cleaning Script Started! Loading dependencies...", flush=True)
import time
import sqlite3
import hashlib
import requests
import itertools
import pandas as pd
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
RAW_HEADLINES_PATH = "headlines_train_english_titles.csv"
if not os.path.exists(RAW_HEADLINES_PATH):
    for possible_path in ["headlines_train_英文标题.csv", "headlines_train_英文标题(1).csv", "headlines_train_english_titles(1).csv"]:
        if os.path.exists(possible_path):
            RAW_HEADLINES_PATH = possible_path
            break
DB_CACHE_PATH = "llm_cache_v13_distribution_fixed.db"  
MODEL_CB_REPO = "Captain-1337/CrudeBERT"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
def save_csv_safe(df, filename):
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"📁 Successfully saved: {filename}", flush=True)
    except PermissionError:
        alt = filename.replace('.csv', f"_backup_{int(time.time())}.csv")
        df.to_csv(alt, index=False, encoding='utf-8-sig')
def run_crudebert_probs(df, batch_size=32):
    print(f"\n📦 [1/5] Loading CrudeBERT model...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_path = hf_hub_download(repo_id=MODEL_CB_REPO, filename="crude_bert_config.json")
    model_path = hf_hub_download(repo_id=MODEL_CB_REPO, filename="crude_bert_model.bin")
    
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForSequenceClassification.from_config(config)
    
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict.pop("bert.embeddings.position_ids", None)
    model.load_state_dict(state_dict, strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()
    all_probs = []
    titles = df['title_clean'].tolist()
    total = len(titles)
    
    print(f"🚀 Starting CrudeBERT low-level inference (Total: {total} items)...", flush=True)
    for i in range(0, total, batch_size):
        batch_text = titles[i:i+batch_size]
        inputs = tokenizer(batch_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            
    return np.vstack(all_probs)
class DeepSeekInference:
    def __init__(self, db_path, api_key):
        self.db_path = db_path
        self.api_key = api_key
        self.url = DEEPSEEK_API_URL
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, response TEXT)")
            conn.commit()
    def predict(self, text):
        prompt = f"""Analyze this crude oil news headline like a commodity trader. Does this headline imply bullish (Positive) or bearish (Negative) momentum for oil prices? 
FORCE yourself to choose Positive or Negative if it hints at supply/demand shifts, EVEN IF it is written objectively. ONLY choose Neutral if it is 100% irrelevant to oil prices.
Headline: "{text}"
Reply with EXACTLY ONE WORD from these three: Positive, Negative, Neutral. Do not add any punctuation or explanation."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT response FROM cache WHERE hash=?", (text_hash,))
            row = cur.fetchone()
            if row: return row[0]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": "deepseek-chat",
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]
        }
        for attempt in range(3):
            try:
                res = requests.post(self.url, headers=headers, json=payload, timeout=20)
                res.raise_for_status()
                content = res.json()['choices'][0]['message']['content'].strip()
                
                content_lower = content.lower()
                if "pos" in content_lower: final_label = "Positive"
                elif "neg" in content_lower: final_label = "Negative"
                elif "neu" in content_lower: final_label = "Neutral"
                else: final_label = "Neutral"
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (text_hash, final_label))
                return final_label
            
            except Exception as e:
                if attempt == 2:
                    print(f"\n❌ [Critical] API failed 3 consecutive times. Network or Key might be invalid: {str(e)}", flush=True)
                    return "Error" 
                time.sleep(2)
def align_and_monitor(df, cb_probs, ds_labels):
    print("\n🔄 [3/5] Running SLAM alignment mechanism & distribution probe...", flush=True)
    
    cb_raw_preds = np.argmax(cb_probs, axis=1)
    print("\n📊 --- Prediction Distribution Probe Report ---")
    print(f"DeepSeek Label Distribution: {pd.Series(ds_labels).value_counts().to_dict()}")
    print(f"CrudeBERT Raw Channel Distribution: {pd.Series(cb_raw_preds).value_counts().to_dict()}")
    print("--------------------------\n")
    
    if "Error" in ds_labels:
        print("⚠️ Warning: DeepSeek results contain API Errors. Please check your network or API connection!", flush=True)
        
    classes = ["Positive", "Negative", "Neutral"]
    permutations = list(itertools.permutations([0, 1, 2]))
    
    valid_mask = np.array([l in classes for l in ds_labels])
    valid_cb_preds = cb_raw_preds[valid_mask]
    valid_ds_labels = np.array(ds_labels)[valid_mask]
    
    if len(valid_ds_labels) == 0:
        raise ValueError("All LLM predictions failed. Please check API connectivity!")
    best_acc = -1
    best_mapping = None
    
    for perm in permutations:
        mapping = {classes[i]: perm[i] for i in range(3)}
        idx2label = {v: k for k, v in mapping.items()}
        mapped_labels = np.array([idx2label[idx] for idx in valid_cb_preds])
        
        acc = np.mean(mapped_labels == valid_ds_labels)
        if acc > best_acc:
            best_acc = acc
            best_mapping = mapping
    print(f"🎯 SLAM locked physical channels as: {best_mapping}")
    print(f"📈 [Reconstructed] Peak dual-model consistency rate: {best_acc*100:.2f}%\n", flush=True)
    
    pos_idx, neg_idx, neu_idx = best_mapping["Positive"], best_mapping["Negative"], best_mapping["Neutral"]
    
    df['cb_prob_pos'] = cb_probs[:, pos_idx]
    df['cb_prob_neg'] = cb_probs[:, neg_idx]
    df['cb_prob_neu'] = cb_probs[:, neu_idx]
    df['cb_score'] = df['cb_prob_pos'] - df['cb_prob_neg']
    
    idx2label = {v: k for k, v in best_mapping.items()}
    df['cb_label'] = [idx2label[idx] for idx in cb_raw_preds]
    df['cb_status'] = 'success'
    
    return df
def aggregate_daily_sentiment(df, prefix):
    valid_df = df[df[f'{prefix}_status'] == 'success'].copy()
    
    valid_df[f'{prefix}_is_pos'] = (valid_df[f'{prefix}_label'] == "Positive").astype(int)
    valid_df[f'{prefix}_is_neg'] = (valid_df[f'{prefix}_label'] == "Negative").astype(int)
    valid_df[f'{prefix}_is_neu'] = (valid_df[f'{prefix}_label'] == "Neutral").astype(int)
    daily_df = valid_df.groupby('date').agg(
        n_articles=('title_clean', 'count'),
        sent_mean=(f'{prefix}_score', 'mean'),
        pos_share=(f'{prefix}_is_pos', 'mean'),
        neg_share=(f'{prefix}_is_neg', 'mean'),
        neu_share=(f'{prefix}_is_neu', 'mean'),
        p_pos_mean=(f'{prefix}_prob_pos', 'mean'),
        p_neu_mean=(f'{prefix}_prob_neu', 'mean'),
        p_neg_mean=(f'{prefix}_prob_neg', 'mean')
    ).reset_index()
    daily_df.columns = [
        'date', 'n_articles', 'sent_mean',
        'pos_share', 'neg_share', 'neu_share',
        'p_pos_mean', 'p_neu_mean', 'p_neg_mean'
    ]
    return daily_df
if __name__ == "__main__":
    if not os.path.exists(RAW_HEADLINES_PATH):
        print(f"❌ Critical Error: Dataset not found: {RAW_HEADLINES_PATH}")
        exit(1)
    df = pd.read_csv(RAW_HEADLINES_PATH)
    
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c or '日期' in c), None)
    title_col = next((c for c in df.columns if 'title' in c.lower() or '标题' in c), None)
    
    df['date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    
    if 'title_raw' in df.columns:
        df['title_clean'] = df['title_raw'].astype(str).str.strip()
    elif 'title_raw（原标题）' in df.columns:
        df['title_clean'] = df['title_raw（原标题）'].astype(str).str.strip()
    else:
        df['title_clean'] = df[title_col].astype(str).str.strip()
    cb_probs = run_crudebert_probs(df)
    print("\n🌐 [2/5] Starting DeepSeek API fast inference (Bypassing Neutral Trap)...", flush=True)
    ds_engine = DeepSeekInference(DB_CACHE_PATH, DEEPSEEK_API_KEY)
    
    ds_labels = []
    total_titles = len(df['title_clean'])
    for idx, t in enumerate(df['title_clean'], 1):
        if idx % 50 == 0 or idx == total_titles:
            print(f"   --> API Progress: {idx}/{total_titles}", flush=True)
        ds_labels.append(ds_engine.predict(t))
        
    df['llm_label'] = ds_labels
    df['llm_status'] = ['success' if l in ["Positive", "Negative", "Neutral"] else 'failed' for l in ds_labels]
    
    df['llm_prob_pos'] = df['llm_label'].apply(lambda x: 1.0 if x == 'Positive' else 0.0)
    df['llm_prob_neg'] = df['llm_label'].apply(lambda x: 1.0 if x == 'Negative' else 0.0)
    df['llm_prob_neu'] = df['llm_label'].apply(lambda x: 1.0 if x == 'Neutral' else 0.0)
    df['llm_score'] = df['llm_prob_pos'] - df['llm_prob_neg']
    df = align_and_monitor(df, cb_probs, ds_labels)
    print("\n📁 [4/5] Exporting daily feature tables...", flush=True)
    daily_cb = aggregate_daily_sentiment(df, prefix='cb')
    daily_llm = aggregate_daily_sentiment(df, prefix='llm')
    save_csv_safe(daily_cb, "sentiment_cb_daily.csv")
    save_csv_safe(daily_llm, "sentiment_llm_daily.csv")
    
    print("\n✨ Peak alignment run completed! Please inspect the 'Prediction Distribution Probe Report' printed in the terminal.")
