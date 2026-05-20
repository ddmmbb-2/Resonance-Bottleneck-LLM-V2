import torch
import torch.nn as nn
import os
import math
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from tokenizers import Tokenizer
import csv 
import shutil

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 🎯 V22-Optimizer 實驗配置 (新增 Phase 3 設定)
# ==========================================
config = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 12,
    "latent_dim": 256,
    "dropout": 0.1,
    "max_seq_len": 512,
    "batch_size": 4,
    "block_size": 256,
    "chunk_size": 64,          # 🎬 Phase 3: Micro-Chunking 大小
    "cache_capacity": 512,     # 🎬 Phase 3: Keyframe Cache 容量
    "accum_steps": 8,
    "think_steps": 3,
    "lr": 3e-4,              
    "min_lr": 3e-5,          
    "warmup_steps": 500,     
    "max_steps": 20000,      
    "epochs": 100000,        
    "bin_data": "corpus_v20_twllm.bin", 
    "save_model": "d2_v22_twllm_optimizer.pth", 
    "log_csv": "v22_twllm_optimizer_log.csv",   
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,
    "halt_tau": 0.05,                  
    "inference_exit_threshold": 0.85   
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V22 Latent Optimizer (含 Phase 3 長文本共振) | 設備: {device}")

# ==========================================
# 1. 資料加載與優化型 get_batch
# ==========================================
if not os.path.exists(config["bin_data"]):
    raise FileNotFoundError(f"❌ 找不到 {config['bin_data']}！請確認檔案位置。")

tokenizer = Tokenizer.from_file(config["vocab_name"])
config["vocab_size"] = tokenizer.get_vocab_size() 
vocab_size = config["vocab_size"]
print(f"🔥 詞表大小: {vocab_size}")

data = np.memmap(config["bin_data"], dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x_list, y_list = [], []
    for i in ix:
        x_list.append(torch.from_numpy(data[i:i+config["block_size"]].astype(np.int64)))
        y_list.append(torch.from_numpy(data[i+1:i+config["block_size"]+1].astype(np.int64)))
        
    x = torch.stack(x_list).pin_memory().to(device, non_blocking=True)
    y = torch.stack(y_list).pin_memory().to(device, non_blocking=True)
    return x, y

if not os.path.exists(config["log_csv"]):
    with open(config["log_csv"], mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Final_CE", "Step0_CE", "Improvement", "LR", "Diffs", "Halts"])

# ==========================================
# 2. 基礎組件
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
# 🎬 Phase 3: 濃縮與共振模組
# ==========================================
class CausalKeyframeCache(nn.Module):
    def __init__(self, capacity, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.capacity = capacity
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        # 🔴 修復 2：引入 RoPE 為外部 Cache 提供絕對位置資訊
        self.rope = RoPE(self.d_head, max_seq_len=max_seq_len)
        self.super_tokens = [] # 儲存格式改為 tuple: (vector, end_pos)

    def clear(self):
        self.super_tokens = []

    def add(self, vector, end_pos):
        # 🔴 修復 2：同時記錄該 Super Token 對應的結束位置
        self.super_tokens.append((vector.detach(), end_pos))
        if len(self.super_tokens) > self.capacity:
            self.super_tokens.pop(0)

    def get_all_as_kv(self, batch_size, current_start_pos):
        # 🔴 修復 2：嚴格因果過濾，只取 end_pos <= current_start_pos 的 tokens
        valid_tokens = [(v, p) for v, p in self.super_tokens if p <= current_start_pos]
        if not valid_tokens:
            return None

        # (B, M, D)
        vectors = torch.stack([v for v, _ in valid_tokens], dim=1) 
        
        # 🟢 修復 5：動態處理 Batch Size 不匹配的情況 (推理期防呆機制)
        if vectors.shape[0] != batch_size:
            if vectors.shape[0] > batch_size:
                vectors = vectors[:batch_size]
            else:
                padding = vectors.new_zeros(batch_size - vectors.shape[0], *vectors.shape[1:])
                vectors = torch.cat([vectors, padding], dim=0)

        positions = torch.tensor([p for _, p in valid_tokens], device=vectors.device, dtype=torch.long) # (M,)

        kv = self.kv_proj(vectors) # (B, M, 2*D)
        k, v = kv.chunk(2, dim=-1)
        B, M, _ = k.shape
        
        k = k.view(B, M, self.n_heads, self.d_head)
        v = v.view(B, M, self.n_heads, self.d_head).transpose(1, 2) # (B, H, M, D_head)

        # 🔴 修復 2：根據絕對位置 positions 應用 RoPE
        cos = self.rope.cos[:, positions, :, :] # (1, M, 1, D_head)
        sin = self.rope.sin[:, positions, :, :] # (1, M, 1, D_head)
        
        k1, k2 = k.chunk(2, dim=-1)
        k_rot = torch.cat((-k2, k1), dim=-1)
        k = k * cos + k_rot * sin
        
        k = k.transpose(1, 2) # (B, H, M, D_head)
        return k, v

class CausalTokenSelector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, hidden_chunk, halt_probs):
        # 🌟 Phase 3 核心：根據思考深度 (1 - halt_prob) 動態加權融合 Super Token
        importance = 1.0 - halt_probs 
        # 🟢 修復 6：將 1e-5 改為 1e-6，提升數值穩定性
        weights = importance / (importance.sum(dim=1, keepdim=True) + 1e-6)
        super_token = (hidden_chunk * weights).sum(dim=1) # (B, D)
        return self.proj(super_token)

# ==========================================
# 3. 核心 Attention (🔥 已整合 Phase 3 緩存機制)
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
        
        # 🎬 Phase 3: 控制外部記憶注入的閘門
        self.cache_gate = nn.Linear(d_model, 1)

    def forward(self, x, external_kv=None):
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
        
        kv_div = (kv_input.float() / safe_df_kv)
        z_div = (z_input.float() / safe_df_z)
        
        kv_div_perm = kv_div.permute(0, 2, 3, 4, 1).contiguous()
        z_div_perm = z_div.permute(0, 2, 3, 1).contiguous()
        
        kv_states = torch.cumsum(kv_div_perm, dim=-1).permute(0, 4, 1, 2, 3) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = torch.cumsum(z_div_perm, dim=-1).permute(0, 3, 1, 2) * torch.exp(cum_log_decay)
        
        out_num = (q_f.unsqueeze(-2) @ kv_states.to(x.dtype)).squeeze(-2) 
        den = torch.clamp((q_f * z_states.to(x.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5) 
        out_local = self.mem_norm((out_num / den).contiguous().view(B, L, D))

        # 🎬 Phase 3: 外部緩存注入 (Flash Attention 加速)
        if external_kv is not None:
            k_ext, v_ext = external_kv # (B, H, M, D_head)
            q_ext = q.transpose(1, 2)  # (B, H, L, D_head)
            
            out_cache = F.scaled_dot_product_attention(q_ext, k_ext, v_ext)
            out_cache = out_cache.transpose(1, 2).contiguous().view(B, L, -1)
            
            gate_c = torch.sigmoid(self.cache_gate(x))
            out_combined = out_local + gate_c * out_cache
        else:
            out_combined = out_local

        gate_val = F.silu(self.out_gate(latent))
        return self.dropout(self.proj(out_combined) * gate_val)

# ==========================================
# 4. 交叉注意力與海馬迴 (保持原樣，因應 Chunk 不受影響)
# ==========================================
class ReasonCrossAttention(nn.Module):
    def __init__(self, d_model, latent_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        self.temp = nn.Parameter(torch.ones(self.n_heads, 1, 1) * math.log(10.0))

    def forward(self, h_query, K_mem, V_mem, external_kv=None):
        B, L, H, D = K_mem.shape
        Q = self.q_proj(h_query).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = K_mem.transpose(1, 2) # (B, H, L, D_head)
        V = V_mem.transpose(1, 2) # (B, H, L, D_head)
        
        # 🟢 修復 4：接收外部 Cache 並在 seq_len 維度拼接到 Cross Attention
        if external_kv is not None:
            k_ext, v_ext = external_kv
            K = torch.cat([k_ext, K], dim=2)
            V = torch.cat([v_ext, V], dim=2)
        
        Q = self.q_norm(Q)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        scale_factor = torch.exp(self.temp * 0.5)
        Q = Q * scale_factor
        K = K * scale_factor
        
        # 注意：若有拼接外部記憶，is_causal 在處理全域 KV 時需視情況調整，但這裡 FlashAttention 的 is_causal 
        # 對於 K, V 長度大於 Q 時，會自動進行對齊，因此可以安全保留。
        out = F.scaled_dot_product_attention(Q, K, V, scale=1.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)

class BrainInspiredHippocampus(nn.Module):
    def __init__(self, latent_dim, expansion_factor=4, n_heads=8, top_k=4):
        super().__init__()
        self.n_heads = n_heads
        self.top_k = top_k  
        self.dg_dim = latent_dim * expansion_factor
        self.d_head = self.dg_dim // n_heads
        self.dg_expand = nn.Linear(latent_dim, self.dg_dim, bias=False)
        self.dg_norm = RMSNorm(self.dg_dim)
        self.q_proj = nn.Linear(self.dg_dim, self.dg_dim, bias=False)
        self.k_proj = nn.Linear(self.dg_dim, self.dg_dim, bias=False)
        self.v_proj = nn.Linear(self.dg_dim, self.dg_dim, bias=False)
        self.beta = nn.Parameter(torch.ones(self.n_heads, 1, 1) * math.log(5.0))
        self.ca1_compress = nn.Linear(self.dg_dim, latent_dim, bias=False)

    def forward(self, h_query):
        B, L, D = h_query.shape
        h_dg = F.silu(self.dg_expand(h_query)) 
        h_curr = self.dg_norm(h_dg)
        mask = ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=h_query.device))

        Q = self.q_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        scores = (Q @ K.transpose(-2, -1)) * torch.exp(self.beta)
        scores = scores.masked_fill(mask, float('-inf'))
        
        if L > self.top_k:
            safe_scores = scores.masked_fill(mask, -1e4)
            topk_vals, _ = torch.topk(safe_scores, self.top_k, dim=-1)
            kth_vals = topk_vals[..., -1].unsqueeze(-1)
            sparse_mask = (scores < kth_vals) & (~mask)
            scores = scores.masked_fill(sparse_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, self.dg_dim)
        h_curr = self.dg_norm(h_curr + out)
            
        return self.ca1_compress(h_curr - h_dg)

# ==========================================
# 5. V22 推理核心：Latent Optimizer
# ==========================================
class ResonanceOptimizerCore(nn.Module):
    def __init__(self, d_model, latent_dim, think_steps=3):
        super().__init__()
        self.steps = think_steps
        self.step_embed = nn.Embedding(think_steps, latent_dim)
        self.init_proj = nn.Linear(d_model, latent_dim)
        self.latent_to_model = nn.Linear(latent_dim, d_model, bias=False)
        self.context_to_kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.k_norm = RMSNorm(d_model // config["n_heads"])
        self.cross_attn = ReasonCrossAttention(d_model, latent_dim, config["n_heads"])
        self.hippocampus = BrainInspiredHippocampus(latent_dim, expansion_factor=4, n_heads=config["n_heads"], top_k=4)
        
        self.router = nn.Linear(latent_dim * 3, latent_dim * 2) 
        self.master_gate = nn.Linear(latent_dim, latent_dim)
        nn.init.constant_(self.master_gate.bias, 1.5)
        self.norm = RMSNorm(latent_dim)
        self.exit_gate = nn.Linear(latent_dim, 1) 
        
        self.register_buffer("avg_diff", torch.zeros(1)) 
        self.register_buffer("avg_halt_prob", torch.zeros(1))

    def forward(self, x, external_kv=None, return_all_steps=True):
        B, L, D = x.shape
        h_latent_init = self.init_proj(x)
        h_latent = h_latent_init
        
        K_mem, V_mem = self.context_to_kv(x).chunk(2, dim=-1)
        K_mem = K_mem.view(B, L, config["n_heads"], -1)
        V_mem = V_mem.view(B, L, config["n_heads"], -1)
        K_mem = self.k_norm(K_mem) 
        
        intermediate_states, diff_norms, halt_logits = [], [], []
        last_intermediate = None
        
        for i in range(self.steps):
            step_ids = torch.full((B,), i, device=x.device, dtype=torch.long)
            h_query = 0.6 * h_latent + 0.3 * h_latent_init + 0.1 * self.step_embed(step_ids).unsqueeze(1)
            
            if i > 0 and self.training:
                K_step = K_mem.detach() + 0.1 * (K_mem - K_mem.detach())
                V_step = V_mem.detach() + 0.1 * (V_mem - V_mem.detach())
            else:
                K_step, V_step = K_mem, V_mem
                
            # 🟢 修復 4：傳遞 external_kv 進入 cross_attn，讓推理核心看見外部記憶
            delta_external = self.cross_attn(h_query, K_step, V_step, external_kv=external_kv)
            delta_internal = self.hippocampus(h_query) 
            
            route_features = torch.cat([h_latent, delta_external, delta_internal], dim=-1)
            route_logits = self.router(route_features) 
            route_gates = torch.sigmoid(route_logits).view(B, L, 2, -1)
            weight_ext, weight_hipp = route_gates.unbind(2) 
            
            master_g = torch.sigmoid(self.master_gate(h_latent))
            delta_total = master_g * (weight_ext * delta_external + weight_hipp * delta_internal)
            delta_total_clamped = torch.clamp(delta_total, min=-4.0, max=4.0)
            
            h_next = self.norm(h_latent + delta_total_clamped)

            raw_diff = torch.norm(delta_total_clamped.detach(), p=2, dim=-1, keepdim=True)
            diff_norm = raw_diff / math.sqrt(config["latent_dim"])
            pred_halt_logit = self.exit_gate(h_next)
            
            current_intermediate = self.latent_to_model(h_next)
            last_intermediate = current_intermediate
            
            if return_all_steps:
                intermediate_states.append(current_intermediate)
                
            diff_norms.append(diff_norm)
            halt_logits.append(pred_halt_logit)
            
            if self.training:
                self.avg_diff = 0.9 * self.avg_diff + 0.1 * diff_norm.mean()
                self.avg_halt_prob = 0.9 * self.avg_halt_prob + 0.1 * torch.sigmoid(pred_halt_logit).detach().mean()
                adaptive_noise_scale = 1e-4 * torch.clamp(self.avg_diff, max=1.0)
                h_next = h_next + torch.randn_like(h_next) * adaptive_noise_scale 
            else:
                if torch.sigmoid(pred_halt_logit).mean() > config["inference_exit_threshold"]:
                    break
                    
            h_latent = h_next

        if return_all_steps:
            return last_intermediate, intermediate_states, diff_norms, halt_logits
        else:
            return last_intermediate, [last_intermediate], diff_norms, halt_logits

# ==========================================
# 5.5 V22 背景記憶核心：SSM 時間序列
# ==========================================
class D2V20SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        
        self.ln = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=4, groups=self.d_inner, padding=3)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        xz = self.in_proj(x_norm)
        x_hidden, z = xz.chunk(2, dim=-1)
        
        x_conv = x_hidden.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_conv = self.act(x_conv)
        
        x_dbl = self.x_proj(x_conv)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) 
        
        A = -torch.exp(self.A_log.float()) 
        log_decay = dt.unsqueeze(-1) * A   
        
        log_decay_perm = log_decay.permute(0, 2, 3, 1).contiguous()
        cum_log_decay = torch.cumsum(log_decay_perm, dim=-1).permute(0, 3, 1, 2)
        
        dt_B_x = dt.unsqueeze(-1) * B_mat.unsqueeze(-2) * x_conv.unsqueeze(-1)
        safe_div = torch.exp(cum_log_decay) + 1e-5
        div_x = (dt_B_x.float() / safe_div)
        div_x_perm = div_x.permute(0, 2, 3, 1).contiguous()
        states = torch.cumsum(div_x_perm, dim=-1).permute(0, 3, 1, 2) * torch.exp(cum_log_decay)
        
        y = (states.to(x.dtype) * C_mat.unsqueeze(-2)).sum(dim=-1)
        y = y + x_conv * self.D
        out = self.act(z) * y
        return x + self.out_proj(out)

# ==========================================
# 6. V22 主模型 (🔥 Phase 3: 全新 Chunk-wise Recurrence)
# ==========================================
class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, config["latent_dim"])
        self.ffn = SwiGLU(d_model)
        
    def forward(self, x, external_kv=None):
        x = x + self.attn(x, external_kv)
        x = x + self.ffn(x)
        return x

class D2V20HybridModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        
        # 🎬 Phase 3: 緩存與選擇器初始化
        self.chunk_size = config["chunk_size"]
        # 🔴 修復：傳入 max_seq_len 給 Cache 用於初始化 RoPE
        self.cache = CausalKeyframeCache(
            config["cache_capacity"], 
            d_model, 
            config["n_heads"], 
            max_seq_len=config["max_seq_len"]
        )
        self.selector = CausalTokenSelector(d_model)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                self.blocks.append(ResonanceOptimizerCore(d_model, config["latent_dim"], config["think_steps"]))
            elif i % 2 == 0:
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                self.blocks.append(D2V20SSMBlock(d_model))
                
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x, return_all_steps=False):
        B, L = x.shape
        chunk_size = self.chunk_size
        
        # 用於收集所有 Chunk 的結果，最後拼接還原 Seq_Len，對 Loss 計算 0 影響！
        final_logits_list = []
        all_diffs_chunked = []
        all_halts_chunked = []
        step_logits_chunked = []
        
        last_opt_idx = max((i for i, b in enumerate(self.blocks) if isinstance(b, ResonanceOptimizerCore)), default=-1)

        # 🎬 Phase 3: 分塊遞推迴圈 (Micro-Chunking)
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            x_chunk = x[:, start:end]
            
            # 🔴 修復：從 Cache 獲取時傳入 current_start_pos，過濾未來的記憶，保證嚴格因果性
            ext_kv = self.cache.get_all_as_kv(B, current_start_pos=start) 
            
            curr_x = self.emb_dropout(self.embedding(x_chunk))
            
            chunk_intermediates = []
            chunk_diffs = []
            chunk_halts = []
            chunk_final_x_base = None

            for idx, block in enumerate(self.blocks):
                if isinstance(block, ResonanceOptimizerCore):
                    x_base = curr_x 
                    # 🟢 修復：將 ext_kv 傳給 OptimizerCore，讓交叉注意力也能看見長時記憶
                    out, intermediates, diffs, halts = block(curr_x, external_kv=ext_kv)
                    curr_x = curr_x + out
                    
                    if idx == last_opt_idx and return_all_steps:
                        chunk_intermediates = intermediates
                        chunk_diffs = diffs
                        chunk_halts = halts
                        chunk_final_x_base = x_base 
                elif isinstance(block, D2V18AttentionBlock):
                    # 將 ext_kv 傳入 Attention 進行相位共振
                    curr_x = block(curr_x, external_kv=ext_kv)
                else:
                    curr_x = block(curr_x)
                    
            chunk_final_logits = self.head(self.out_ln(curr_x))
            final_logits_list.append(chunk_final_logits)
            
            # 🎬 Phase 3: Chunk 結束時進行動態濃縮融合
            if chunk_halts:
                # 拿最後一步的 Halt Probability 作為重要性指標 (1-prob) 
                halt_probs = torch.sigmoid(chunk_halts[-1])
                importance = 1.0 - halt_probs 
                
                # 🟡 修復：增加重要性閾值，只有當 Chunk 平均重要性 > 0.3 時才生成 Super Token
                if importance.mean() > 0.3:
                    # curr_x 是這個 chunk 經過所有 block 後的隱含表示
                    super_token = self.selector(curr_x, halt_probs)
                    # 🔴 修復：傳入該 Chunk 的 end_pos，用於後續的因果過濾與 RoPE 注入
                    self.cache.add(super_token, end_pos=end)

            if return_all_steps:
                c_step_logits = []
                for x_step in chunk_intermediates:
                    step_x = chunk_final_x_base + x_step
                    for i in range(last_opt_idx + 1, len(self.blocks)):
                        block = self.blocks[i]
                        if isinstance(block, D2V18AttentionBlock):
                            step_x = block(step_x, external_kv=ext_kv)
                        else:
                            step_x = block(step_x)
                    c_step_logits.append(self.head(self.out_ln(step_x)))
                    
                step_logits_chunked.append(c_step_logits)
                all_diffs_chunked.append(chunk_diffs)
                all_halts_chunked.append(chunk_halts)
                
        # 迴圈結束，將 Chunk 拼接回原來的維度 (B, L, ...)
        final_logits = torch.cat(final_logits_list, dim=1)
        
        if return_all_steps:
            num_steps = len(step_logits_chunked[0])
            final_step_logits = []
            final_diffs = []
            final_halts = []
            
            for s in range(num_steps):
                final_step_logits.append(torch.cat([chk[s] for chk in step_logits_chunked], dim=1))
                final_diffs.append(torch.cat([chk[s] for chk in all_diffs_chunked], dim=1))
                final_halts.append(torch.cat([chk[s] for chk in all_halts_chunked], dim=1))
                
            return final_logits, final_step_logits, final_diffs, final_halts
            
        return final_logits

def get_lr(it):
    if it < config["warmup_steps"]:
        return config["lr"] * it / config["warmup_steps"]
    if it > config["max_steps"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_steps"]) / (config["max_steps"] - config["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return config["min_lr"] + coeff * (config["lr"] - config["min_lr"])

# ==========================================
# 7. 訓練迴圈
# ==========================================
model = D2V20HybridModel(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0 
smoothed_ce = None

if os.path.exists(config["save_model"]):
    print(f"🔄 找到檢查點 {config['save_model']}，正在載入訓練狀態...")
    checkpoint = torch.load(config["save_model"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint.get('step', 0)
    smoothed_ce = checkpoint.get('smoothed_ce', None)
    print(f"✅ 成功從 Step {global_step} 繼續訓練！")
else:
    print("🆕 未找到既有檢查點，從頭開始訓練。")

pbar = tqdm(initial=global_step, total=config["epochs"], desc="訓練中")

diff_str, halt_str = "N/A", "N/A"

while global_step < config["epochs"]:
    lr = get_lr(global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    step_final_ce = 0
    step_halt_loss = 0 
    
    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        
        # 🎬 嚴格因果性保護：每個新 batch 進入前，務必清空 Cache 記憶
        model.cache.clear() 
        
        with autocast('cuda', dtype=torch.bfloat16):
            final_logits, step_logits, diffs, halts = model(xb, return_all_steps=True)
            target = yb.view(-1)
            final_ce = F.cross_entropy(final_logits.view(-1, vocab_size), target)
            ce_losses = [F.cross_entropy(logits.view(-1, vocab_size), target) for logits in step_logits]
            
            actual_steps = len(ce_losses)
            gamma = 0.8 
            raw_weights = [gamma ** (actual_steps - 1 - i) for i in range(actual_steps)]
            step_weights = [w / sum(raw_weights) for w in raw_weights]
            
            total_loss = 0
            for i in range(actual_steps):
                total_loss += ce_losses[i] * step_weights[i]
                total_loss += (diffs[i] ** 2).mean() * 0.001
            
            halt_loss = 0
            for i in range(actual_steps):
                target_prob = torch.ones_like(halts[i]) if i == actual_steps - 1 else torch.zeros_like(halts[i])
                halt_loss += F.binary_cross_entropy_with_logits(halts[i], target_prob)
            
            total_loss += halt_loss * 0.05
            loss_to_back = total_loss / config["accum_steps"]
            loss_to_back.backward()
            
            step_final_ce += final_ce.item()
            step_halt_loss += halt_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    avg_ce = step_final_ce / config["accum_steps"]
    avg_halt = step_halt_loss / config["accum_steps"]
    if smoothed_ce is None:
        smoothed_ce = avg_ce
    else:
        smoothed_ce = 0.99 * smoothed_ce + 0.01 * avg_ce 
    
    global_step += 1
    pbar.update(1)

    if global_step % 10 == 0:
        diffs_log = [b.avg_diff.item() for b in model.blocks if isinstance(b, ResonanceOptimizerCore)]
        halts_log = [b.avg_halt_prob.item() for b in model.blocks if isinstance(b, ResonanceOptimizerCore)]
        diff_str = f"[{','.join([f'{d:.2f}' for d in diffs_log])}]" if diffs_log else "N/A"
        halt_str = f"[{','.join([f'{h:.2f}' for h in halts_log])}]" if halts_log else "N/A"

    pbar.set_postfix({
        "CE": f"{avg_ce:.3f}", 
        "LR": f"{lr:.2e}", 
        "D": diff_str,      
        "P": halt_str      
    })

    if global_step % 10 == 0:
        with open(config["log_csv"], mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, f"{avg_ce:.4f}", f"{smoothed_ce:.4f}", f"{avg_halt:.4f}", f"{lr:.6f}", diff_str, halt_str])

    if global_step % 1000 == 0:
        ckpt = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'smoothed_ce': smoothed_ce  
        }
        torch.save(ckpt, config["save_model"])
        backup_path = config["save_model"].replace(".pth", f"_step_{global_step}.pth")
        shutil.copy2(config["save_model"], backup_path)