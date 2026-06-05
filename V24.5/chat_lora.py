import torch
import torch.nn as nn
import os
import math
from tokenizers import Tokenizer

# 載入你原本 chat.py 的配置與模型架構
from chat import config, D2V20HybridModel

# 1. 必須引入相同的 LoRA 結構，否則權重會對不上
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.original_linear(x) + (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scale

def apply_lora_to_v245(model, r=8, alpha=16):
    for name, module in model.named_modules():
        if "attn" in name and hasattr(module, "qkv_expand"):
            if isinstance(module.qkv_expand, nn.Linear):
                module.qkv_expand = LoRALinear(module.qkv_expand, r=r, alpha=alpha)
        if "blocks" in name and hasattr(module, "init_proj") and hasattr(module, "latent_to_model"):
            if isinstance(module.init_proj, nn.Linear):
                module.init_proj = LoRALinear(module.init_proj, r=r, alpha=alpha)
                module.latent_to_model = LoRALinear(module.latent_to_model, r=r, alpha=alpha)
    return model

# 2. 初始化與動態外掛載入
def init_lora_chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = Tokenizer.from_file(config["vocab_name"])
    config['vocab_size'] = tokenizer.get_vocab_size()
    
    # A. 載入底座
    base_model_path = "d2_v24_samba_latent.pth"
    model = D2V20HybridModel(config['vocab_size'], config['d_model'], config['n_layers'])
    checkpoint = torch.load(base_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # B. 注入空 LoRA 結構
    model = apply_lora_to_v245(model, r=8, alpha=16)
    
    # C. 填入剛練好的 LoRA 微調權重
    lora_weight_path = "v24_5_daily_chat_lora.pth"
    if os.path.exists(lora_weight_path):
        print(f"✅ 成功融合日常對話 LoRA 權重: {lora_weight_path}")
        lora_checkpoint = torch.load(lora_weight_path, map_location=device)
        # strict=False 很重要，因為我們只載入帶有 "lora_" 關鍵字的參數
        model.load_state_dict(lora_checkpoint['lora_state_dict'], strict=False)
    else:
        print(f"⚠️ 警告：找不到 {lora_weight_path}，將使用未微調的純底座進行對話！")
        
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

# 3. 帶有 SFT 模板與 Stop Token 的生成演算法
def generate_response(model, tokenizer, device, user_input, max_new_tokens=150, temperature=0.85, rep_penalty=1.8):
    # 順應 57k 底座記憶的黃金 Template
    prompt = f"User: {user_input}\nAssistant:"
    
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)
            next_token_logits = logits[:, -1, :].clone()
            
            # 💡 1. 強力重複懲罰：只要字出現過，機率直接打骨折，徹底消滅「選擇合適的選擇」與「王國興」
            for token_id in set(generated_ids):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= rep_penalty
                else:
                    next_token_logits[0, token_id] *= rep_penalty
            
            # 💡 2. 溫度採樣
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token_id)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=-1)
            
            # 💡 3. 全域字串防禦（核心修正）：直接解碼整段已生成的文字來檢查！
            # 這樣就算 Token 被切成 U -> s -> e -> r，只要拼起來有 User，就立刻中斷！
            current_output_text = tokenizer.decode(generated_ids).replace(" ", "")
            
            if "User" in current_output_text or "Assistant" in current_output_text or next_token_id == 0:
                # 剔除掉最後不小心吐出來的垃圾標籤，保持輸出純淨
                break
                
    # 重新解碼最終文字
    final_response = tokenizer.decode(generated_ids).replace(" ", "")
    # 清理結尾可能殘留的標籤片段
    if "User" in final_response:
        final_response = final_response.split("User")[0]
    return final_response

# # 4. 對話主迴圈
def main():
    print("🔥 V24.5 Samba-Latent 對話解鎖版啟動中...")
    model, tokenizer, device = init_lora_chat()
    print("\n🎤 臺灣本土日常對話模式就緒！輸入 'quit' 或 'exit' 結束。")
    print("--------------------------------------------------")
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                print("👋 下次見！")
                break
                
            print("🤖 模型思考中...")
            model.cache.clear()
            
            # 🚀 完美的降溫嚴謹版（只留這一個呼叫即可）：
            reply = generate_response(
                model, 
                tokenizer, 
                device, 
                user_input, 
                temperature=0.3,    # 讓模型專注，只挑機率最高的正確字，拒絕胡思亂想
                rep_penalty=1.15    # 微幅懲罰，允許模型重複使用「的、是、在」等基礎字
            )
            print(f"🤖 助手: {reply}")
            
        except KeyboardInterrupt:
            print("\n👋 下次見！")
            break

if __name__ == "__main__":
    main()