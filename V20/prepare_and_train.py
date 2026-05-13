import pandas as pd
import json
import os
from opencc import OpenCC
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ==========================================
# 🎯 參數設定
# ==========================================
parquet_file = "train-00000-of-00001-5fcf805680823132.parquet"
json_file = "alpaca_gpt4_data_zh.json"
wiki_file = "wiki_zh.parquet"
temp_corpus_txt = "full_corpus_for_tokenizer.txt"
output_tokenizer = "bpe_tokenizer_v12.json"

cc = OpenCC('s2twp')  # 轉台灣繁體慣用語

def clean_and_convert(text):
    if not text: return ""
    return cc.convert(text.strip())

# ==========================================
# 1. 提取並合併純文字語料
# ==========================================
print("📝 開始提取語料並轉為繁體...")

with open(temp_corpus_txt, 'w', encoding='utf-8') as f_out:
    # --- 處理 Parquet ---
    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="處理 Parquet"):
            for turn in row['conversations']:
                f_out.write(clean_and_convert(turn.get('content', '')) + "\n")

    # --- 處理 JSON (Alpaca) ---
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in tqdm(data, desc="處理 JSON"):
                text = f"{item.get('instruction','')}\n{item.get('input','')}\n{item.get('output','')}"
                f_out.write(clean_and_convert(text) + "\n")

    # --- 處理 Wikipedia (取 50% 訓練 Tokenizer 就夠了) ---
    if os.path.exists(wiki_file):
        df_wiki = pd.read_parquet(wiki_file).sample(frac=0.5, random_state=42)
        for _, row in tqdm(df_wiki.iterrows(), total=len(df_wiki), desc="處理 Wiki"):
            text = f"{row.get('title','')}\n{row.get('text','')}"
            f_out.write(clean_and_convert(text) + "\n")

print(f"✅ 語料提取完成，已存至 {temp_corpus_txt}")

# ==========================================
# 2. 訓練 Tokenizer
# ==========================================
print("⏳ 開始訓練 BPE Tokenizer (預計耗時數分鐘)...")

# 初始化 BPE
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 設定訓練器
# 16384 是一個適合小型模型且平衡性能的 Vocab Size
trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>"]
)

# 開始訓練
tokenizer.train([temp_corpus_txt], trainer)

# 儲存
tokenizer.save(output_tokenizer)
print(f"🎉 成功！Tokenizer 已儲存為: {output_tokenizer}")

# (選做) 刪除暫存檔
# os.remove(temp_corpus_txt)