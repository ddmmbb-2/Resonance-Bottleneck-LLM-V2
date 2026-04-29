import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from tokenizers import Tokenizer

# ==========================================
# 🎯 V19-Mini 實驗配置 (3060 友善版)
# ==========================================
config = {
    "d_model": 256,          
    "n_heads": 4,           
    "n_layers": 6,           # 局部推理實驗：只做 6 層
    "latent_dim": 128,       
    "dropout": 0.1,          
    "max_seq_len": 512,      
    "batch_size": 16,        
    "block_size": 256,       
    "accum_steps": 4,        
    "think_steps": 2,        # 每次推理迭代 2 次
    "lr": 3e-4,              
    "epochs": 100000,        
    "warmup_steps": 1000,    
    "bin_data": "corpus_v15_clean.bin", 
    "save_model": "d2_v19_mini.pth", 
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,                      
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V19-Mini 實驗啟動中 | 設備: {device}")

# ==========================================
# 1. 資料加載
# ==========================================
if not os.path.exists(config["bin_data"]):
    raise FileNotFoundError(f"❌ 找不到 {config['bin_data']}！請確認檔案位置。")

tokenizer = Tokenizer.from_file(config["vocab_name"])
vocab_size = tokenizer.get_vocab_size() 
data = np.memmap(config["bin_data"], dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([torch.from_numpy(data[i:i+config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+config["block_size"]+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 2. 基礎組件 (RMSNorm, RoPE, CausalConv1d, SwiGLU)
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

# ==========================================
# 3. 核心 Attention (加入簡化版 Context 互動)
# ==========================================
class LatentResonanceAttentionV18(nn.Module):
    def __init__(self, d_model, latent_dim, dropout=0.1):
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
        self.dropout = nn.Dropout(dropout)

    def forward_with_context(self, context, query):
        # 簡易版資訊交互：直接疊加
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

# ==========================================
# 4. V19 推理模塊與主模型
# ==========================================
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
        
        # Latent 初始化投影 (給個暖啟動)
        self.init_proj = nn.Linear(d_model, latent_dim)
        self.register_buffer("avg_gate_val", torch.zeros(1))

    def _step(self, x, h_latent, step_idx):
        step_ids = torch.full((x.size(0),), step_idx, device=x.device, dtype=torch.long)
        mod = self.step_modulator(step_ids).unsqueeze(1) 
        scale, bias = mod.chunk(2, dim=-1)
        
        h_input = h_latent * (1.0 + scale) + bias
        h_query = self.latent_to_model(h_input)
        
        delta_model = self.reason_attn.forward_with_context(context=x, query=h_query)
        delta_latent = self.norm(self.model_to_latent(delta_model))
        
        gate_val = torch.sigmoid(self.gate(torch.cat([h_latent, delta_latent], dim=-1)) * 1.2)
        
        if self.training:
            self.avg_gate_val = 0.9 * self.avg_gate_val + 0.1 * gate_val.detach().mean()
            
        h_next = h_latent + gate_val * torch.tanh(delta_latent)
        
        if self.training:
            h_next = h_next + torch.randn_like(h_next) * 0.01
            
        return h_next

    def forward(self, x):
        h_latent = self.init_proj(x) # [B, L, Latent_dim]
        for i in range(self.steps):
            h_latent = checkpoint(self._step, x, h_latent, i, use_reentrant=False)
        return self.latent_to_model(self.norm(h_latent))

class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, latent_dim=config["latent_dim"])
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class D2V18ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.conv = CausalConv1d(d_model)
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])
    def forward(self, x):
        x = x + self.conv(self.ln(x))
        x = x + self.ffn(x)
        return x

class D2V19MiniModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [2, 4]: # 🎯 指定局部使用 Reasoning Core
                self.blocks.append(ResonanceReasoningCore(d_model, config["latent_dim"], config["think_steps"]))
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
            # 加入殘差確保 Reasoning Block 不會破壞主幹梯度
            x = x + checkpoint(block, x, use_reentrant=False) if isinstance(block, ResonanceReasoningCore) else checkpoint(block, x, use_reentrant=False)
        return self.head(self.out_ln(x))

# ==========================================
# 5. 訓練與監控迴圈
# ==========================================
model = D2V19MiniModel(config["vocab_size"], config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)

# 👇 新增這三行：強制告訴 PyTorch 初始學習率是多少 👇
for param_group in optimizer.param_groups:
    param_group['initial_lr'] = config["lr"]
    param_group['lr'] = config["lr"]

warmup_scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / config["warmup_steps"]), last_epoch=global_step)

print(f"🌟 V19-Mini 模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"🚀 已喚醒 V19-Mini 模型！局部推理層在 Block 2 與 4，準備起飛！")

model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V19-Mini 訓練中")

while global_step < config["epochs"]:
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0 
    
    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = ce_loss / config["accum_steps"]
        
        loss.backward()
        total_loss += ce_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    avg_loss = total_loss / config["accum_steps"]
    
    if global_step < config["warmup_steps"]:
        warmup_scheduler.step()

    global_step += 1
    
    # 📊 抓取 Gate 的活躍度來觀察 Reasoning Core
    gate_vals = [b.avg_gate_val.item() for b in model.blocks if isinstance(b, ResonanceReasoningCore)]
    gate_str = f"[{','.join([f'{g:.3f}' for g in gate_vals])}]" if gate_vals else "N/A"

    pbar.update(1)
    pbar.set_postfix({
        "Loss": f"{avg_loss:.4f}", 
        "LR": f"{optimizer.param_groups[0]['lr']:.1e}",
        "Gate": gate_str
    })

    if global_step % 2000 == 0:
        ckpt = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(ckpt, config["save_model"])  
        print(f"\n🚩 Step {global_step} 存檔成功！Gate狀態: {gate_str}")