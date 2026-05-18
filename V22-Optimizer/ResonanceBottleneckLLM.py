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
import csv 
import shutil


# ==========================================
# 🌟 神級加速算子：強制連續化與 3D 極速掃描
# ==========================================
class FastContiguousCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        # 1. 強制要求記憶體連續 (解決 Memory Fragmentation 的關鍵)
        x_contig = x.contiguous()
        return torch.cumsum(x_contig, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        grad_output_contig = grad_output.contiguous()
        # 2. Cumsum 的數學反向傳播：將梯度反轉 -> Cumsum -> 再次反轉
        # PyTorch 的 torch.flip 底層極度優化，這比保存計算圖快上幾十倍
        grad_x = torch.flip(torch.cumsum(torch.flip(grad_output_contig, dims=[dim]), dim=dim), dims=[dim])
        return grad_x, None

def fast_cumsum(x, dim=1):
    return FastContiguousCumsum.apply(x, dim)

# ==========================================
# 🎯 V22-Optimizer 實驗配置 (精簡重複項)
# ==========================================
config = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 12,
    "latent_dim": 256,
    "dropout": 0.1,
    "max_seq_len": 512,
    "batch_size": 8,
    "block_size": 256,
    "accum_steps": 4,
    "think_steps": 3,
    "lr": 3e-4,              
    "min_lr": 3e-5,          
    "warmup_steps": 500,     # 統一只保留一個 warmup_steps
    "max_steps": 20000,      
    "epochs": 100000,        # 這是 while 迴圈的終點
    "bin_data": "corpus_v20_twllm.bin", 
    "save_model": "d2_v22_twllm_optimizer.pth", 
    "log_csv": "v22_twllm_optimizer_log.csv",   
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,
    "halt_tau": 0.05,                  
    "inference_exit_threshold": 0.85   
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V22 Latent Optimizer 修復版啟動中 | 設備: {device}")

# ==========================================
# 1. 資料加載與日誌初始化
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
# 3. 核心 Attention (保留原本高精度的自研模塊)
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
        
        # ------ 🌟 高效加速區塊開始 ------
        # 1. 取得除法後的純淨張量 (保持 Float32 以防溢出)
        kv_div = (kv_input.float() / safe_df_kv).contiguous()
        z_div = (z_input.float() / safe_df_z).contiguous()
        
        # 2. 攤平成 3D 張量 (B*H, L, Features)，觸發底層最高速的 Parallel Scan
        B, L, H, D_head = q.shape
        kv_div_flat = kv_div.view(B * H, L, -1)
        z_div_flat = z_div.view(B * H, L, -1)
        
        # 3. 呼叫我們的自定義 Autograd 算子
        kv_states_flat = fast_cumsum(kv_div_flat, dim=1)
        z_states_flat = fast_cumsum(z_div_flat, dim=1)
        
        # 4. 恢復原狀，再乘上衰減矩陣
        kv_states = kv_states_flat.view(B, L, H, D_head, D_head) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = z_states_flat.view(B, L, H, D_head) * torch.exp(cum_log_decay)
        # ------ 🌟 高效加速區塊結束 ------
        out_num = (q_f.unsqueeze(-2) @ kv_states.to(x.dtype)).squeeze(-2) 
        den = torch.clamp((q_f * z_states.to(x.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5) 
        
        out = self.mem_norm((out_num / den).contiguous().view(B, L, D))
        gate_val = F.silu(self.out_gate(latent))
        return self.dropout(self.proj(out) * gate_val)

# ==========================================
# 4. V22 輕量級交叉注意力
# ==========================================
class ReasonCrossAttention(nn.Module): # 加上這行 🌟
    def __init__(self, d_model, latent_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        
        # 🌟 修復: 引入可學習的溫度參數 (取代固定的 math.sqrt(d_head))
        # 初始值設為 log(10)，這樣 exp(temp) 約等於 10，適合 normalized 的 QK 內積
        self.temp = nn.Parameter(torch.ones(self.n_heads, 1, 1) * math.log(10.0))

    def forward(self, h_query, K_mem, V_mem):
        B, L, H, D = K_mem.shape
        Q = self.q_proj(h_query).view(B, L, self.n_heads, self.d_head)
        Q = self.q_norm(Q)
        
        Q = Q.transpose(1, 2) 
        K = K_mem.transpose(1, 2)
        V = V_mem.transpose(1, 2)
        
        # 🌟 修復: QK-Norm (Cosine Attention)，徹底防止 Entropy Collapse
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        # 使用可學習的溫度進行縮放
        scores = (Q @ K.transpose(-2, -1)) * torch.exp(self.temp)
        
        mask = torch.triu(torch.ones(L, L, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)

# ==========================================
# 4.5 仿生海馬迴模組 (DG + CA3)
# ==========================================
class BrainInspiredHippocampus(nn.Module):
    def __init__(self, latent_dim, expansion_factor=4, n_heads=8, top_k=4):
        super().__init__()
        self.n_heads = n_heads
        self.top_k = top_k  # 🌟 側向抑制：只允許最活躍的 K 個神經元存活
        
        # 1. DG (Dentate Gyrus) 齒狀回：模式分離 (Pattern Separation)
        self.dg_dim = latent_dim * expansion_factor
        self.d_head = self.dg_dim // n_heads
        self.dg_expand = nn.Linear(latent_dim, self.dg_dim, bias=False)
        self.dg_norm = RMSNorm(self.dg_dim)
        
        # 2. CA3 自返性側支 (Recurrent Collaterals)
        self.q_proj = nn.Linear(self.dg_dim, self.dg_dim, bias=False)
        self.k_proj = nn.Linear(self.dg_dim, self.dg_dim, bias=False)
        self.v_proj = nn.Linear(self.dg_dim, self.dg_dim, bias=False)
        
        # 高逆溫度 (Beta)，促使吸引子動力學生效
        self.beta = nn.Parameter(torch.ones(self.n_heads, 1, 1) * math.log(5.0))
        
        # 3. CA1 輸出投射：將高維記憶壓縮回皮層維度
        self.ca1_compress = nn.Linear(self.dg_dim, latent_dim, bias=False)

    def forward(self, h_query):
        B, L, D = h_query.shape
        
        # 步驟 1: DG 模式分離
        h_dg = F.silu(self.dg_expand(h_query)) 
        h_curr = self.dg_norm(h_dg)
        
        # ==========================================
        # 步驟 2: CA3 遞迴模式補全 (迭代收斂)
        # ==========================================
        iters = 1
        for _ in range(iters):
            Q = self.q_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            K = self.k_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            V = self.v_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            
            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)
            
            scores = (Q @ K.transpose(-2, -1)) * torch.exp(self.beta)
            
            # 🌟 修復 1：必須【先】套用因果遮罩，確保不偷看未來
            mask = torch.triu(torch.ones(L, L, device=h_query.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            
            # 🌟 修復 2：【再】做側向抑制 (Top-K)
            if L > self.top_k:
                # 找出每行合法範圍內第 K 大的值
                topk_vals, _ = torch.topk(scores, self.top_k, dim=-1)
                kth_vals = topk_vals[..., -1].unsqueeze(-1)
                # 將小於該閾值的分數抹除 (注意：-inf < -inf 在 PyTorch 中為 False，所以不會誤殺合法 token)
                scores = scores.masked_fill(scores < kth_vals, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            out = (attn @ V).transpose(1, 2).contiguous().view(B, L, self.dg_dim)
            
            h_curr = self.dg_norm(h_curr + out)
            
        # 步驟 3: CA1 輸出增量
        memory_delta = self.ca1_compress(h_curr - h_dg) 
        return memory_delta

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
        
        # 🌟 換成全新的仿生海馬迴模組
        self.hippocampus = BrainInspiredHippocampus(
            latent_dim, 
            expansion_factor=4, 
            n_heads=config["n_heads"], 
            top_k=4
        )
        
        self.router = nn.Linear(latent_dim * 3, latent_dim * 2) 
        
        self.master_gate = nn.Linear(latent_dim, latent_dim)
        nn.init.constant_(self.master_gate.bias, 1.5)
        self.norm = RMSNorm(latent_dim)
        self.exit_gate = nn.Linear(latent_dim, 1) 
        
        self.register_buffer("avg_diff", torch.zeros(1)) 
        self.register_buffer("avg_halt_prob", torch.zeros(1))

    def forward(self, x):
        B, L, D = x.shape
        h_latent_init = self.init_proj(x)
        h_latent = h_latent_init
        
        K_mem, V_mem = self.context_to_kv(x).chunk(2, dim=-1)
        K_mem = K_mem.view(B, L, config["n_heads"], -1)
        V_mem = V_mem.view(B, L, config["n_heads"], -1)
        K_mem = self.k_norm(K_mem) 
        
        intermediate_states = []
        diff_norms = []
        halt_logits = []
        
        for i in range(self.steps):
            step_ids = torch.full((B,), i, device=x.device, dtype=torch.long)
            h_query = 0.6 * h_latent + 0.3 * h_latent_init + 0.1 * self.step_embed(step_ids).unsqueeze(1)
            
            if i > 0 and self.training:
                K_step = K_mem.detach() + 0.1 * (K_mem - K_mem.detach())
                V_step = V_mem.detach() + 0.1 * (V_mem - V_mem.detach())
            else:
                K_step, V_step = K_mem, V_mem
                
            # 1. 算出兩個軌道的原始增量
            delta_external = self.cross_attn(h_query, K_step, V_step)
            delta_internal = self.hippocampus(h_query)  # 🌟 呼叫海馬迴提取記憶
            
            # 2. 🌟 獨立協同路由 (Synergistic Gating)
            route_features = torch.cat([h_latent, delta_external, delta_internal], dim=-1)
            route_logits = self.router(route_features) # (B, L, latent_dim * 2)
            
            # 🌟 取代原本的 Softmax，改用 Sigmoid
            route_gates = torch.sigmoid(route_logits).view(B, L, 2, -1)
            weight_ext, weight_hipp = route_gates.unbind(2) # 兩者值域均為 [0, 1]
            
            # 3. 🌟 總量調控 (Master Scaling)
            master_g = torch.sigmoid(self.master_gate(h_latent))
            
            # 最終融合增量
            delta_total = master_g * (weight_ext * delta_external + weight_hipp * delta_internal)
            delta_total_clamped = torch.clamp(delta_total, min=-4.0, max=4.0)
            
            # 更新與標準化
            h_next = self.norm(h_latent + delta_total_clamped)

            raw_diff = torch.norm(delta_total_clamped.detach(), p=2, dim=-1, keepdim=True)
            diff_norm = raw_diff / math.sqrt(config["latent_dim"])
            pred_halt_logit = self.exit_gate(h_next)
            intermediate_states.append(self.latent_to_model(h_next))
            diff_norms.append(diff_norm)
            halt_logits.append(pred_halt_logit)
            
            if self.training:
                self.avg_diff = 0.9 * self.avg_diff + 0.1 * diff_norm.mean()
                self.avg_halt_prob = 0.9 * self.avg_halt_prob + 0.1 * torch.sigmoid(pred_halt_logit).detach().mean()
                h_next = h_next + torch.randn_like(h_next) * 1e-4 
            else:
                if torch.sigmoid(pred_halt_logit).mean() > config["inference_exit_threshold"]:
                    break
                    
            h_latent = h_next

        return intermediate_states[-1], intermediate_states, diff_norms, halt_logits
# ==========================================
# 5.5 V22 背景記憶核心：SSM 時間序列全域掃描
# ==========================================
class D2V20SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        
        self.ln = RMSNorm(d_model)
        
        # 1. 擴展投影與門控
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. 局部時序特徵提取 (因果卷積)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            groups=self.d_inner,
            padding=3 # 配合 kernel_size=4 進行因果 padding
        )
        
        # 3. SSM 參數動態投影層 (提取時間步長 dt, 以及 B, C 矩陣)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # 4. 狀態空間轉移矩陣 (A 與 D)
        # 採用指數衰減初始化，確保長距離記憶穩定
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        
        # 1. 投影與 Gated 控制分支
        xz = self.in_proj(x_norm)
        x_hidden, z = xz.chunk(2, dim=-1)
        
        # 2. 短期局部特徵 (Causal Conv)
        x_conv = x_hidden.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L] # 截斷右側多餘的 padding，保持因果性
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)
        
        # 3. 計算動態 SSM 參數
        x_dbl = self.x_proj(x_conv)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) # (B, L, d_inner) 保證時間步長為正
        
        # 4. 全域時間序列掃描 (Pure PyTorch Parallel Scan)
        # 將連續的 A 矩陣離散化: A_discrete = exp(dt * A_log)
        A = -torch.exp(self.A_log.float()) # 確保 A 為負值，代表衰減 (d_inner, d_state)
        log_decay = dt.unsqueeze(-1) * A   # (B, L, d_inner, d_state)
        
        # 🌟 沿用你的穩定 Cumsum 技巧，完美避開寫自定義 CUDA 的麻煩
        cum_log_decay = torch.cumsum(log_decay, dim=1)
        
        # B_mat: (B, L, 1, d_state) * x_conv: (B, L, d_inner, 1) -> dt_B_x: (B, L, d_inner, d_state)
        dt_B_x = dt.unsqueeze(-1) * B_mat.unsqueeze(-2) * x_conv.unsqueeze(-1)
        
        safe_div = torch.exp(cum_log_decay) + 1e-8
        
        # ------ 🌟 SSM 高效加速區塊開始 ------
        B_ssm, L_ssm, D_inner, D_state = dt_B_x.shape
        
        # 1. 強制連續化與除法
        div_x = (dt_B_x.float() / safe_div).contiguous()
        
        # 2. 攤平成 3D (B*D_inner, L, D_state)
        div_x_flat = div_x.view(B_ssm * D_inner, L_ssm, -1)
        
        # 3. 自定義高速掃描
        states_flat = fast_cumsum(div_x_flat, dim=1)
        
        # 4. 恢復原狀並乘上衰減
        states = states_flat.view(B_ssm, L_ssm, D_inner, D_state) * torch.exp(cum_log_decay)
        # ------ 🌟 SSM 高效加速區塊結束 ------
        
        # 5. 觀測矩陣映射 (Y = C * H + D * X)
        y = (states.to(x.dtype) * C_mat.unsqueeze(-2)).sum(dim=-1)
        y = y + x_conv * self.D
        
        # 6. Gated Output & 殘差連接
        out = self.act(z) * y
        out = self.out_proj(out)
        
        return x + out

# ==========================================
# 6. V22 主模型
# ==========================================
class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, config["latent_dim"])
        self.ffn = SwiGLU(d_model)
        
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class D2V20HybridModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        
        self.blocks = nn.ModuleList()
        # 🌟 核心架構堆疊策略變更：三明治夾心
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                # 階段 3: 動態邏輯工作區
                self.blocks.append(ResonanceOptimizerCore(d_model, config["latent_dim"], config["think_steps"]))
            elif i % 2 == 0:
                # 階段 1: 高精度注意力 (精確檢索)
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                # 階段 2: SSM 時間序列背景記憶 (全域掃描，取代原本的局部卷積)
                self.blocks.append(D2V20SSMBlock(d_model))
                
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x, return_all_steps=False):
        x = self.emb_dropout(self.embedding(x))
        all_step_states = []
        all_diffs = []
        all_halts = []
        final_x_base = None
        
        # 🌟 修復 1: 預先找出最後一個 ResonanceOptimizerCore 的 index
        last_opt_idx = max((i for i, b in enumerate(self.blocks) if isinstance(b, ResonanceOptimizerCore)), default=-1)

        for idx, block in enumerate(self.blocks):
            if isinstance(block, ResonanceOptimizerCore):
                x_base = x  # 🌟 修復 5: 在進入 Optimizer 前保留乾淨的 Backbone Residual
                out, intermediates, diffs, halts = block(x)
                x = x + out
                
                # 判斷是否為最後一個 Optimizer
                if idx == last_opt_idx and return_all_steps:
                    all_step_states = intermediates
                    all_diffs = diffs
                    all_halts = halts
                    final_x_base = x_base # 存下基礎 x_base，供後續 logit 計算使用
            else:
                x = block(x)
                
        final_logits = self.head(self.out_ln(x))
        
        if return_all_steps:
            step_logits = []
            for x_step in all_step_states:
                # 🌟 基礎語意結合：保留 Backbone 在該層的原本特徵
                step_x = final_x_base + x_step
                
                # 🛡️ 動態路由修復：如果 Optimizer 後面還有其他標準層 (Attention/SSM)，
                # 必須讓這些中間態特徵也過完這些層，確保進入 LM Head 前的特徵空間是對齊的！
                for i in range(last_opt_idx + 1, len(self.blocks)):
                    step_x = self.blocks[i](step_x)
                    
                # 通過最後的 LayerNorm 與 LM Head
                step_logits.append(self.head(self.out_ln(step_x)))
                
            return final_logits, step_logits, all_diffs, all_halts
            
        return final_logits


def get_lr(it):
    # 1. Linear Warmup: 在預熱期內線性增加學習率
    if it < config["warmup_steps"]:
        return config["lr"] * it / config["warmup_steps"]
    
    # 2. 超過最大步數後，維持在最小學習率
    if it > config["max_steps"]:
        return config["min_lr"]
    
    # 3. Cosine Decay: 在 Warmup 與 Max_steps 之間進行餘弦衰減
    decay_ratio = (it - config["warmup_steps"]) / (config["max_steps"] - config["warmup_steps"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return config["min_lr"] + coeff * (config["lr"] - config["min_lr"])

# ==========================================
# 7. 訓練迴圈與接續訓練
# ==========================================
model = D2V20HybridModel(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0 
smoothed_ce = None

# 🌟 補上接續訓練 (Resume Training) 邏輯
if os.path.exists(config["save_model"]):
    print(f"🔄 找到檢查點 {config['save_model']}，正在載入訓練狀態...")
    # 若有使用 PyTorch 2.0+，建議加上 weights_only=False 或是預設即可
    checkpoint = torch.load(config["save_model"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint.get('step', 0)
    smoothed_ce = checkpoint.get('smoothed_ce', None)
    print(f"✅ 成功從 Step {global_step} 繼續訓練！")
else:
    print("🆕 未找到既有檢查點，從頭開始訓練。")

# 🌟 注意這裡：將 tqdm 的 initial 設為 global_step，這樣進度條才會接續顯示
pbar = tqdm(initial=global_step, total=config["epochs"], desc="訓練中")

while global_step < config["epochs"]:
    # 1. 更新當前步數的學習率
    lr = get_lr(global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    
    # 2. 梯度累積訓練
    step_final_ce = 0
    step_halt_loss = 0 
    
    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        with autocast('cuda', dtype=torch.bfloat16):
            final_logits, step_logits, diffs, halts = model(xb, return_all_steps=True)
            target = yb.view(-1)
            final_ce = F.cross_entropy(final_logits.view(-1, vocab_size), target)
            ce_losses = [F.cross_entropy(logits.view(-1, vocab_size), target) for logits in step_logits]
            
            # --- 這裡插入你精密的 Loss 計算邏輯 (step_weights, Margin-based 等) ---
            # ==========================================
            # 方案 A：純淨的逐步引導 (返璞歸真版)
            # ==========================================
            actual_steps = len(ce_losses)
            # 讓越深層的思考步驟，佔據越高的 Loss 權重 (例如 3 步就是 1/6, 2/6, 3/6)
            step_weights = [(i + 1) / sum(range(1, actual_steps + 1)) for i in range(actual_steps)]
            
            # 🌟 1. 這裡必須先宣告 total_loss = 0，並把每一步的預測誤差加進去！
            total_loss = 0
            for i in range(actual_steps):
                total_loss += ce_losses[i] * step_weights[i]
                total_loss += (diffs[i] ** 2).mean() * 0.001
            
            # 🌟 2. 計算停機閘門的誤差
            halt_loss = 0
            for i in range(actual_steps):
                target_prob = torch.ones_like(halts[i]) if i == actual_steps - 1 else torch.zeros_like(halts[i])
                halt_loss += F.binary_cross_entropy_with_logits(halts[i], target_prob)
            
            # 🌟 3. 將兩者合併
            total_loss += halt_loss * 0.05
            
            loss_to_back = total_loss / config["accum_steps"]
            loss_to_back.backward()
            
            step_final_ce += final_ce.item()
            step_halt_loss += halt_loss.item()
            # -----------------------------------------------------------



    # 3. 優化器更新與梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    # 4. 統計平滑 Loss
    avg_ce = step_final_ce / config["accum_steps"]
    avg_halt = step_halt_loss / config["accum_steps"]
    if smoothed_ce is None:
        smoothed_ce = avg_ce
    else:
        smoothed_ce = 0.99 * smoothed_ce + 0.01 * avg_ce 
    
    # 5. 更新步數與進度條 (🌟 關鍵：這裡只執行一次)
    global_step += 1
    pbar.update(1)

    # 6. 提取 Log 資訊
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

    # 7. 寫入 CSV 與 存檔
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
