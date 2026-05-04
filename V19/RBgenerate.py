import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizers import Tokenizer

# ==========================================
# 1. 完整且原汁原味的模型架構定義
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x_fp32 = x.float() 
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(x.dtype)

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
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x_norm = self.ln(x)
        out = self.w3(F.silu(self.w1(x_norm)) * self.w2(x_norm))
        return self.dropout(out)

class LatentResonanceAttentionV18(nn.Module):
    def __init__(self, d_model, latent_dim, dropout=0.1):
        super().__init__()
        self.n_heads = 8  # 依據你的 config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        self.ln = RMSNorm(d_model)
        self.latent_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.qkv_expand = nn.Linear(latent_dim, d_model * 3, bias=False)
        self.reso_expand = nn.Linear(latent_dim, self.n_heads * 4, bias=False) 
        
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.rope = RoPE(self.d_head, max_seq_len=512)
        
        self.out_gate = nn.Linear(latent_dim, d_model, bias=False)
        self.head_decay = nn.Parameter(torch.linspace(-3.0, 1.0, self.n_heads))
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward_with_context(self, context, query):
        return self.forward(context + query)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x) 
        latent = F.silu(self.latent_compress(x_norm))
        
        q, k, v = self.qkv_expand(latent).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head)
        k = k.view(B, L, self.n_heads, self.d_head)
        v = v.view(B, L, self.n_heads, self.d_head)
        
        q, k = self.rope(self.q_norm(q)), self.rope(self.k_norm(k))
        
        q_f, k_f, v_f = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0, v.float()
        
        params = self.reso_expand(latent).view(B, L, self.n_heads, 4)
        sem_amp, sem_phase, ctx_amp, ctx_phase = params.unbind(-1)
        sem_amp, ctx_amp = torch.sigmoid(sem_amp), torch.sigmoid(ctx_amp)
        sem_phase, ctx_phase = torch.sigmoid(sem_phase) * math.pi, torch.sigmoid(ctx_phase) * math.pi
        
        raw_decay = 0.3 + 0.65 * torch.sigmoid(self.head_decay.view(1, 1, self.n_heads))
        decay_rate = torch.clamp(raw_decay, min=1e-5, max=0.999)
        
        dt_kv, dt_z = (1.0 - decay_rate).unsqueeze(-1).unsqueeze(-1), (1.0 - decay_rate).unsqueeze(-1)
        cos_diff = torch.cos(sem_phase - ctx_phase)
        base_gate = torch.sigmoid((sem_amp * ctx_amp * cos_diff) * self.temperature) 
        gate = torch.clamp(base_gate * 1.2 - 0.1, min=0.05, max=0.95)
        
        kv_input = (k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)) * gate.unsqueeze(-1).unsqueeze(-1) * dt_kv 
        z_input = k_f * dt_z

        log_decay = torch.log(decay_rate).unsqueeze(-1) 
        cum_log_decay = torch.cumsum(log_decay.expand(B, L, -1, -1), dim=1) 
        safe_df_z = torch.exp(cum_log_decay) + 1e-8 
        safe_df_kv = safe_df_z.unsqueeze(-1)        
        
        kv_states = torch.cumsum(kv_input.float() / safe_df_kv, dim=1) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = torch.cumsum(z_input.float() / safe_df_z, dim=1) * torch.exp(cum_log_decay)

        out_num = (q_f.unsqueeze(-2) @ kv_states.to(x.dtype)).squeeze(-2) 
        den = torch.clamp((q_f * z_states.to(x.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5) 
        
        out = self.mem_norm((out_num / den).contiguous().view(B, L, D))
        gate_val = F.silu(self.out_gate(latent))
        return self.dropout(self.proj(out) * gate_val)

class ResonanceReasoningCore(nn.Module):
    def __init__(self, d_model, latent_dim, think_steps=2):
        super().__init__()
        self.steps = think_steps
        self.step_modulator = nn.Embedding(think_steps, latent_dim * 2)
        self.latent_to_model = nn.Linear(latent_dim, d_model, bias=False)
        self.model_to_latent = nn.Linear(d_model, latent_dim, bias=False)
        self.reason_attn = LatentResonanceAttentionV18(d_model, latent_dim)
        self.gate = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = RMSNorm(latent_dim)
        self.init_proj = nn.Linear(d_model, latent_dim)
        self.register_buffer("avg_gate_val", torch.zeros(1))

    def forward(self, x):
        h_latent = self.init_proj(x)
        for i in range(self.steps):
            step_ids = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
            mod = self.step_modulator(step_ids).unsqueeze(1) 
            scale, bias = mod.chunk(2, dim=-1)
            
            h_input = h_latent * (1.0 + scale) + bias
            h_query = self.latent_to_model(h_input)
            
            delta_model = self.reason_attn.forward_with_context(context=x, query=h_query)
            delta_latent = self.norm(self.model_to_latent(delta_model))
            
            gate_val = torch.sigmoid(self.gate(torch.cat([h_latent, delta_latent], dim=-1)) * 1.2)
            h_latent = h_latent + gate_val * torch.tanh(delta_latent)
            
        return self.latent_to_model(self.norm(h_latent))

class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, latent_dim=256)
        self.ffn = SwiGLU(d_model, dropout=0.1)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class D2V18ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.conv = CausalConv1d(d_model)
        self.ffn = SwiGLU(d_model, dropout=0.1)
    def forward(self, x):
        x = x + self.conv(self.ln(x))
        x = x + self.ffn(x)
        return x

class D2V19StableModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                self.blocks.append(ResonanceReasoningCore(d_model, 256, 2))
            elif i % 2 == 0:
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                self.blocks.append(D2V18ConvBlock(d_model))
                
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 
        
    def forward(self, x):
        x = self.emb_dropout(self.embedding(x))
        for block in self.blocks:
            x = block(x) if not isinstance(block, ResonanceReasoningCore) else x + block(x)
        return self.head(self.out_ln(x))

# ==========================================
# 2. 載入與推論配置
# ==========================================

config = {
    "d_model": 512,
    "n_layers": 12,
    "vocab_size": 16384,
    "vocab_name": "bpe_tokenizer_v12.json",
    "save_model": "d2_v19_stable.pth",
    "max_seq_len": 512,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"啟動推論設備: {device}")

tokenizer = Tokenizer.from_file(config["vocab_name"])
model = D2V19StableModel(config["vocab_size"], config["d_model"], config["n_layers"]).to(device)

# 載入權重
try:
    ckpt = torch.load(config["save_model"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ V19-Stable 權重載入成功 (訓練進度: {ckpt.get('step', 'unknown')} 步)")
except Exception as e:
    print(f"❌ 載入失敗: {e}")
    exit()

# ==========================================
# 3. 對話生成函數
# ==========================================

@torch.no_grad()
def generate_response(prompt, temperature=0.75, top_p=0.85):
    input_text = f"User: {prompt}\nAssistant: "
    input_ids = tokenizer.encode(input_text).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print("V19: ", end="", flush=True)
    
    for _ in range(256): 
        x_cond = x[:, -config["max_seq_len"]:]
        logits = model(x_cond)[:, -1, :] / temperature
        
        # Nucleus Sampling (Top-p)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        
        char = tokenizer.decode([next_token.item()])
        if next_token.item() == tokenizer.token_to_id("<|endoftext|>") or char == "\n":
            break
        print(char, end="", flush=True)
    print("\n")

# ==========================================
# 4. 啟動對話介面
# ==========================================

if __name__ == "__main__":
    print("\n--- 💡 已進入 V19-Stable 獨立對話模式 ---")
    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip(): continue
            generate_response(user_input)
        except KeyboardInterrupt:
            break
    print("\n👋 已結束。")