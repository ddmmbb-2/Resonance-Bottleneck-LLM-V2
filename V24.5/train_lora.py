import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import math
from tokenizers import Tokenizer
from tqdm import tqdm

# 載入你原本 chat.py 的配置與模型架構
from chat import config, D2V20HybridModel, RMSNorm, ResonanceOptimizerCore, D2V18AttentionBlock, D2V20SSMBlock

# ==================== 1. 自定義 LoRA 模組 ====================
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        
        # 凍結原本的預訓練權重
        for param in self.original_linear.parameters():
            param.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original_linear(x) + (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scale

def apply_lora_to_v245(model, r=8, alpha=16):
    lora_count = 0
    for name, module in model.named_modules():
        if "attn" in name and hasattr(module, "qkv_expand"):
            if isinstance(module.qkv_expand, nn.Linear):
                module.qkv_expand = LoRALinear(module.qkv_expand, r=r, alpha=alpha)
                lora_count += 1
        if "blocks" in name and hasattr(module, "init_proj") and hasattr(module, "latent_to_model"):
            if isinstance(module.init_proj, nn.Linear):
                module.init_proj = LoRALinear(module.init_proj, r=r, alpha=alpha)
                module.latent_to_model = LoRALinear(module.latent_to_model, r=r, alpha=alpha)
                lora_count += 2
    print(f"🚀 LoRA 注入完成！共成功替換了 {lora_count} 個線性層。")
    return model

# ==================== 2. SFT 專屬資料集 (帶 Label Masking) ====================
class SftChatDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 套用日常對話 Prompt 模板
        system_prompt = "<|system|>你是一個親切、幽默且善於聊天的臺灣日常對話助手。\n"
        user_part = f"<|user|>{item['user']}\n<|assistant|>"
        assistant_part = f"{item['assistant']}"
        
        # 分段編碼，用來精確計算長度並做 Mask
        prompt_ids = self.tokenizer.encode(system_prompt + user_part).ids
        assistant_ids = self.tokenizer.encode(assistant_part).ids
        
        # 結合 input_ids
        input_ids = prompt_ids + assistant_ids
        
        # 核心：建立 labels，將 prompt 區域填上 -100 (不計算 Loss)
        labels = [-100] * len(prompt_ids) + assistant_ids
        
        # 截斷
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            
        return torch.tensor(input_ids), torch.tensor(labels)

# Collate Function: 負責動態 Padding
def sft_collate_fn(batch):
    input_ids_list, labels_list = zip(*batch)
    # 使用 0 填充 input_ids，使用 -100 填充 labels
    padded_inputs = nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    padded_labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return padded_inputs, padded_labels

# ==================== 3. 訓練主程式 ====================
def train_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = Tokenizer.from_file(config["vocab_name"])
    config['vocab_size'] = tokenizer.get_vocab_size()
    
    base_model_path = "d2_v24_samba_latent.pth"
    print(f"📦 正在載入 V24.5 終極底座: {base_model_path}...")
    
    model = D2V20HybridModel(config['vocab_size'], config['d_model'], config['n_layers']) 
    checkpoint = torch.load(base_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 全面凍結底座
    for param in model.parameters():
        param.requires_grad = False
    
    # 注入 LoRA
    model = apply_lora_to_v245(model, r=8, alpha=16)
    model = model.to(device)
    model.train()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # 載入新生成的純淨日常對話
    dataset = SftChatDataset("daily_chat.jsonl", tokenizer, max_seq_len=config["max_seq_len"])
    
    # 💡 核心修正：batch_size=1，完全不需要 shuffle 後的 collate_fn 補零！零污染！
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 💡 核心修正：更為溫和的學習率，保護原本大腦的知識結構
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
    
    epochs = 1  # 溫和微調 3 個 Epoch 即可
    print("\n🎬 開始進行高精度日常對話 LoRA 微調...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(inputs)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, config['vocab_size']), 
                shift_labels.view(-1), 
                ignore_index=-100
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        print(f"✨ Epoch {epoch+1} 結束，平均對話 Loss: {total_loss / len(dataloader):.4f}")
        
    # 儲存
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    save_path = "v24_5_daily_chat_lora.pth"
    torch.save({"lora_state_dict": lora_state_dict}, save_path)
    print(f"🎉 恭喜！日常對話 LoRA 微調完成。權重已覆蓋儲存至: {save_path}")

if __name__ == "__main__":
    train_lora()