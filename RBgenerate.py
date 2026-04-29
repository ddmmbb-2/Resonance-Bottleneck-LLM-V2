import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import os
import math

# ==========================================
# 🚀 V18.1 推論配置 (與訓練腳本完全同步)
# ==========================================
config = {
    "d_model": 768,          
    "n_heads": 12,           
    "n_layers": 24,          
    "latent_dim": 512,       
    "max_seq_len": 2048,
    "block_size": 512,
    "vocab_name": "bpe_tokenizer_v12.json",     
    "save_model": "d2_v18_resonance_pro.pth", 
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🌟 V18.1 核心組件實作
# ==========================================

class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, :, None, :]) 
        self.register_buffer("sin", emb.sin()[None, :, None, :])

    def forward(self, x):
        L = x.shape[1]
        cos, sin = self.cos[:, :L, :, :], self.sin[:, :L, :, :]
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rot * sin

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x_fp32 = x.float() 
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(x.dtype)

class CausalConv1d(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., :-2].transpose(1, 2)

class SwiGLU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden_dim = int(d_model * 8 / 3) 
        hidden_dim = (hidden_dim + 63) // 64 * 64 
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.ln = RMSNorm(d_model)
    def forward(self, x):
        x_norm = self.ln(x)
        return self.w3(F.silu(self.w1(x_norm)) * self.w2(x_norm))

class LatentResonanceAttentionV18(nn.Module):
    def __init__(self, d_model, latent_dim=512):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        self.ln = RMSNorm(d_model)
        self.latent_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.qkv_expand = nn.Linear(latent_dim, d_model * 3, bias=False)
        self.reso_expand = nn.Linear(latent_dim, self.n_heads * 4, bias=False) 
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.rope = RoPE(self.d_head, max_seq_len=config["max_seq_len"])
        self.out_gate = nn.Linear(latent_dim, d_model, bias=False)
        self.head_decay = nn.Parameter(torch.linspace(-3.0, 1.0, self.n_heads))
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_norm = RMSNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x) 
        latent = F.silu(self.latent_compress(x_norm))
        q, k, v = self.qkv_expand(latent).chunk(3, dim=-1)
        q = self.rope(self.q_norm(q.view(B, L, self.n_heads, self.d_head)))
        k = self.rope(self.k_norm(k.view(B, L, self.n_heads, self.d_head)))
        v = v.view(B, L, self.n_heads, self.d_head)
        
        q_f, k_f = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0
        v_f = v.float()
        
        params = self.reso_expand(latent).view(B, L, self.n_heads, 4)
        sem_amp, sem_phase, ctx_amp, ctx_phase = params.unbind(-1)
        sem_phase, ctx_phase = torch.sigmoid(sem_phase) * math.pi, torch.sigmoid(ctx_phase) * math.pi
        
        decay_rate = torch.clamp(0.3 + 0.65 * torch.sigmoid(self.head_decay.view(1, 1, self.n_heads)), 1e-5, 0.999)
        gate = torch.clamp(torch.sigmoid((torch.sigmoid(sem_amp) * torch.sigmoid(ctx_amp) * torch.cos(sem_phase - ctx_phase)) * self.temperature) * 1.2 - 0.1, 0.05, 0.95)
        
        kv_input = (k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)) * gate.unsqueeze(-1).unsqueeze(-1) * (1.0 - decay_rate).view(1, 1, self.n_heads, 1, 1)
        z_input = k_f * (1.0 - decay_rate).view(1, 1, self.n_heads, 1)

        log_decay = torch.log(decay_rate).view(1, 1, self.n_heads, 1)
        cum_log_decay = torch.cumsum(log_decay.expand(B, L, -1, -1), dim=1)
        df = torch.exp(cum_log_decay)
        
        kv_states = torch.cumsum(kv_input / (df.unsqueeze(-1) + 1e-8), dim=1) * df.unsqueeze(-1)
        z_states = torch.cumsum(z_input / (df + 1e-8), dim=1) * df
        
        out_num = (q_f.unsqueeze(-2) @ kv_states.to(q_f.dtype)).squeeze(-2)
        den = torch.clamp((q_f * z_states.to(q_f.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5)
        
        out = (out_num / den).view(B, L, D)
        return self.proj(self.mem_norm(out)) * F.silu(self.out_gate(latent))

# ==========================================
# 🌟 V18.1 區塊與模型封裝
# ==========================================

class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, latent_dim=config["latent_dim"])
        self.ffn = SwiGLU(d_model)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class D2V18ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.conv = CausalConv1d(d_model)
        self.ffn = SwiGLU(d_model)
    def forward(self, x):
        x = x + self.conv(self.ln(x))
        x = x + self.ffn(x)
        return x

class D2V18Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            D2V18AttentionBlock(d_model) if i % 2 == 0 else D2V18ConvBlock(d_model)
            for i in range(n_layers)
        ])
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.out_ln(x))

# ==========================================
# 3. 推論交互介面
# ==========================================
def generate():
    tokenizer = Tokenizer.from_file(config["vocab_name"])
    vocab_size = tokenizer.get_vocab_size()
    model = D2V18Model(vocab_size, config["d_model"], config["n_layers"]).to(device)
    
    # 🎯 修改這裡：優先讀取你剛剛中斷的 Step 6000 檔案
    model_path = "d2_v18_step_6000.pth"
    if not os.path.exists(model_path):
        model_path = config["save_model"]
        
    if os.path.exists(model_path):
        print(f"♻️ 正在載入 V18.1 權重: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        # 訓練腳本存檔時有包一層 model_state_dict
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("❌ 找不到權重檔案！")
        return

    model.eval()
    print(f"🚀 V18.1 測試模式 (Temp: 0.7, Top-K: 40)")
    
    while True:
        prompt = input("\n💡 請輸入開頭 (輸入 'q' 退出): ")
        if prompt.lower() == 'q': break
        
        input_ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
        generated = input_ids
        print(f"🤖 V18.1 生成中: ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(150): 
                context = generated[:, -config["block_size"]:]
                logits = model(context)
                logits = logits[:, -1, :] / 0.3 # 稍微提高溫度，讓 6k 步的模型更有創意
        
                # 重複懲罰 (稍微降低一點，避免模型不敢說話)
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= 1.05
                
                v, _ = torch.topk(logits, 10)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                
                word = tokenizer.decode([next_token.item()])
                print(word, end="", flush=True)
                
                if next_token.item() == tokenizer.token_to_id("<|endoftext|>"): break
        print("\n" + "-"*30)

if __name__ == "__main__":
    generate()