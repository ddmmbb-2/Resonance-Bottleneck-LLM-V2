import pandas as pd
import numpy as np
import json
from tokenizers import Tokenizer
from tqdm import tqdm
import os
from opencc import OpenCC

# ==========================================
# 🎯 參數設定
# ==========================================
tokenizer_path = "bpe_tokenizer_v12.json"
parquet_file = "train-00000-of-00001-5fcf805680823132.parquet"
json_file = "alpaca_gpt4_data_zh.json"
wiki_file = "wiki_zh.parquet"
output_bin = "corpus_v20_twllm.bin"

EOS_TOKEN = "<|endoftext|>"

# 初始化 OpenCC：使用 s2twp (簡體轉台灣正體，包含慣用語轉換)
cc = OpenCC('s2twp')

# 載入 Tokenizer
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"❌ 找不到 {tokenizer_path}")
tokenizer = Tokenizer.from_file(tokenizer_path)

all_tokens = []

def process_text(text):
    """清理文字並轉為繁體"""
    if not text:
        return ""
    # 轉繁體並去除多餘空格
    return cc.convert(text.strip())

# --- 1. 處理原有 Parquet 資料 ---
if os.path.exists(parquet_file):
    print(f"📖 讀取 Parquet: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parquet 轉繁體中"):
        conversations = row['conversations']
        if isinstance(conversations, np.ndarray):
            conversations = conversations.tolist()
            
        prompt = ""
        for turn in conversations:
            role = turn.get('role', '')
            # 轉換內容
            content = process_text(turn.get('content', ''))
            
            if role == "human":
                prompt += f"User: {content}\n"
            elif role == "gpt":
                prompt += f"Assistant: {content}{EOS_TOKEN}\n\n"
        
        all_tokens.extend(tokenizer.encode(prompt).ids)

# --- 2. 處理 Alpaca JSON 資料 ---
if os.path.exists(json_file):
    print(f"📖 讀取 JSON: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        alpaca_data = json.load(f)
        
    for item in tqdm(alpaca_data, desc="JSON 轉繁體中"):
        instruction = process_text(item.get("instruction", ""))
        user_input = process_text(item.get("input", ""))
        output = process_text(item.get("output", ""))
        
        full_user_content = f"{instruction}\n{user_input}".strip()
        prompt = f"User: {full_user_content}\n"
        prompt += f"Assistant: {output}{EOS_TOKEN}\n\n"
        
        all_tokens.extend(tokenizer.encode(prompt).ids)

# --- 3. 處理 Wikipedia 資料 (800MB 取 400MB) ---
if os.path.exists(wiki_file):
    print(f"📖 讀取 Wiki: {wiki_file}")
    df_wiki = pd.read_parquet(wiki_file)
    
    # 🎯 隨機取樣 50% (約 400MB)
    df_wiki = df_wiki.sample(frac=0.5, random_state=42).reset_index(drop=True)
    
    for _, row in tqdm(df_wiki.iterrows(), total=len(df_wiki), desc="Wiki 轉繁體中"):
        title = process_text(row.get('title', ''))
        text = process_text(row.get('text', ''))
        
        # 簡單過濾掉太短的內容（通常是無意義的條目）
        if len(text) < 50:
            continue
            
        prompt = f"User: 請詳細介紹關於{title}的知識。\n"
        prompt += f"Assistant: {text}{EOS_TOKEN}\n\n"
        
        all_tokens.extend(tokenizer.encode(prompt).ids)

# ==========================================
# 💾 儲存為 Binary 檔案
# ==========================================
print(f"💾 正在儲存為 {output_bin} ...")
arr = np.array(all_tokens, dtype=np.uint16)
arr.tofile(output_bin)

print(f"🎉 處理完成！")
print(f"📊 總 Tokens 數: {len(arr):,}")
print(f"📦 預估檔案大小: {len(arr) * 2 / 1024 / 1024:.2f} MB")