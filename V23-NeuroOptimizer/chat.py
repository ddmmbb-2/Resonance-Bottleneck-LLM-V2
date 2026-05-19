import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# ==========================================
# 🎯 V22-Optimizer 配置 (必須與 ResonanceBottleneckLLM.py 嚴格一致)
# ==========================================
config = {
    "d_model": 512,          
    "n_heads": 8,            
    "n_layers": 12,          
    "latent_dim": 256,       
    "dropout": 0.0,          
    "max_seq_len": 512,      
    "think_steps": 3,        
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,
    "inference_exit_threshold": 0.85, # 推理時的信心門檻
    "save_model": "d2_v22_twllm_optimizer.pth" 
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 基礎組件 (RMSNorm, RoPE, SwiGLU)
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
    def __init__(self, d_model):
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

# ==========================================
# 2. 核心模塊 (Attention, SSM, Optimizer)
# ==========================================
class LatentResonanceAttentionV18(nn.Module):
    def __init__(self, d_model, latent_dim):
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
        q = q.view(B, L, self.n_heads, self.d_head)
        k = k.view(B, L, self.n_heads, self.d_head)
        v = v.view(B, L, self.n_heads, self.d_head)
        q, k = self.rope(self.q_norm(q)), self.rope(self.k_norm(k))
        
        q_f, k_f, v_f = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0, v.float()
        params = self.reso_expand(latent).view(B, L, self.n_heads, 4)
        sem_amp, sem_phase, ctx_amp, ctx_phase = params.unbind(-1)
        
        raw_decay = 0.3 + 0.65 * torch.sigmoid(self.head_decay.view(1, 1, self.n_heads))
        decay_rate = torch.clamp(raw_decay, min=1e-5, max=0.999)
        
        cos_diff = torch.cos(torch.sigmoid(sem_phase)*math.pi - torch.sigmoid(ctx_phase)*math.pi)
        gate = torch.clamp(torch.sigmoid((torch.sigmoid(sem_amp)*torch.sigmoid(ctx_amp)*cos_diff)*self.temperature)*1.2 - 0.1, 0.05, 0.95)
        
        kv_input = (k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)) * gate.unsqueeze(-1).unsqueeze(-1) * (1.0 - decay_rate).unsqueeze(-1).unsqueeze(-1)
        z_input = k_f * (1.0 - decay_rate).unsqueeze(-1)

        log_decay = torch.log(decay_rate).unsqueeze(-1)
        cum_log_decay = torch.cumsum(log_decay.expand(B, L, -1, -1), dim=1)
        
        # 🌟 同步加速：L 維度置底連續化，消除跨步開銷
        kv_div = (kv_input.float() / (torch.exp(cum_log_decay).unsqueeze(-1) + 1e-8))
        z_div = (z_input.float() / (torch.exp(cum_log_decay) + 1e-8))
        
        kv_div_perm = kv_div.permute(0, 2, 3, 4, 1).contiguous()
        z_div_perm = z_div.permute(0, 2, 3, 1).contiguous()
        
        kv_states = torch.cumsum(kv_div_perm, dim=-1).permute(0, 4, 1, 2, 3) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = torch.cumsum(z_div_perm, dim=-1).permute(0, 3, 1, 2) * torch.exp(cum_log_decay)

        out_num = (q_f.unsqueeze(-2) @ kv_states.to(x.dtype)).squeeze(-2)
        den = torch.clamp((q_f * z_states.to(x.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5)
        out = self.mem_norm((out_num / den).contiguous().view(B, L, D))
        return self.proj(out) * F.silu(self.out_gate(latent))

class D2V20SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.ln = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, groups=self.d_inner, padding=3)
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
        x_conv = self.conv1d(x_hidden.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = self.act(x_conv)
        x_dbl = self.x_proj(x_conv)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        log_decay = dt.unsqueeze(-1) * A 
        
        # 🌟 同步加速：SSM 衰減掃描雙軌置底連續化
        log_decay_perm = log_decay.permute(0, 2, 3, 1).contiguous()
        cum_log_decay = torch.cumsum(log_decay_perm, dim=-1).permute(0, 3, 1, 2)
        
        dt_B_x = dt.unsqueeze(-1) * B_mat.unsqueeze(-2) * x_conv.unsqueeze(-1)
        div_x = (dt_B_x.float() / (torch.exp(cum_log_decay) + 1e-8))
        div_x_perm = div_x.permute(0, 2, 3, 1).contiguous()
        
        states = torch.cumsum(div_x_perm, dim=-1).permute(0, 3, 1, 2) * torch.exp(cum_log_decay)
        y = (states.to(x.dtype) * C_mat.unsqueeze(-2)).sum(dim=-1) + x_conv * self.D
        return x + self.out_proj(self.act(z) * y)

class ReasonCrossAttention(nn.Module):
    def __init__(self, d_model, latent_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        self.temp = nn.Parameter(torch.ones(self.n_heads, 1, 1) * math.log(10.0))

    def forward(self, h_query, K_mem, V_mem):
        B, L, H, D = K_mem.shape
        Q = self.q_proj(h_query).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K, V = K_mem.transpose(1, 2), V_mem.transpose(1, 2)
        
        Q = self.q_norm(Q)
        Q, K = F.normalize(Q, p=2, dim=-1), F.normalize(K, p=2, dim=-1)
        
        # 🌟 同步加速：融合溫度直接呼叫 FlashAttention 核心
        scale_factor = torch.exp(self.temp * 0.5)
        Q = Q * scale_factor
        K = K * scale_factor
        
        out = F.scaled_dot_product_attention(Q, K, V, scale=1.0, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, L, -1))

# ==========================================
# 4.5 仿生海馬迴模組 (DG + CA3)
# ==========================================
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

        grid = torch.arange(L, device=h_query.device)
        mask = grid.unsqueeze(0) > grid.unsqueeze(1) 

        iters = 1
        for _ in range(iters):
            Q = self.q_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            K = self.k_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            V = self.v_proj(h_curr).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            
            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)
            
            scores = (Q @ K.transpose(-2, -1)) * torch.exp(self.beta)
            scores = scores.masked_fill(mask, float('-inf'))
            
            if L > self.top_k:
                topk_vals, _ = torch.topk(scores, self.top_k, dim=-1)
                kth_vals = topk_vals[..., -1].unsqueeze(-1)
                scores = scores.masked_fill(scores < kth_vals, float('-inf'))
            
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
        
        # 🌟 同步整合：移植仿生海馬迴與協同路由器
        self.hippocampus = BrainInspiredHippocampus(latent_dim, expansion_factor=4, n_heads=config["n_heads"], top_k=4)
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
        K_mem = self.k_norm(K_mem.view(B, L, config["n_heads"], -1))
        V_mem = V_mem.view(B, L, config["n_heads"], -1)
        
        for i in range(self.steps):
            step_ids = torch.full((B,), i, device=x.device, dtype=torch.long)
            h_query = 0.6 * h_latent + 0.3 * h_latent_init + 0.1 * self.step_embed(step_ids).unsqueeze(1)
            
            delta_external = self.cross_attn(h_query, K_mem, V_mem)
            delta_internal = self.hippocampus(h_query)
            
            # 🌟 協同路由閘門計算
            route_features = torch.cat([h_latent, delta_external, delta_internal], dim=-1)
            route_logits = self.router(route_features)
            route_gates = torch.sigmoid(route_logits).view(B, L, 2, -1)
            weight_ext, weight_hipp = route_gates.unbind(2)
            
            master_g = torch.sigmoid(self.master_gate(h_latent))
            delta_total = master_g * (weight_ext * delta_external + weight_hipp * delta_internal)
            delta_total_clamped = torch.clamp(delta_total, min=-4.0, max=4.0)
            
            h_next = self.norm(h_latent + delta_total_clamped)
            
            # 🌟 修正：只檢查最後一個 Token (dim=1 的 -1) 的停機信心，不要用全局 mean()
            if not self.training:
                if torch.sigmoid(self.exit_gate(h_next[:, -1, :])).mean() > config["inference_exit_threshold"]:
                    h_latent = h_next
                    break
            h_latent = h_next
            
        return self.latent_to_model(h_latent), None, None, None

# ==========================================
# 3. 主模型架構
# ==========================================
class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, config["latent_dim"])
        self.ffn = SwiGLU(d_model)
    def forward(self, x):
        return x + self.ffn(x + self.attn(x))

class D2V22HybridModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: self.blocks.append(ResonanceOptimizerCore(d_model, config["latent_dim"]))
            elif i % 2 == 0: self.blocks.append(D2V18AttentionBlock(d_model))
            else: self.blocks.append(D2V20SSMBlock(d_model))
        self.out_ln = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            if isinstance(block, ResonanceOptimizerCore):
                out, _, _, _ = block(x)
                x = x + out
            else:
                x = block(x)
        return self.head(self.out_ln(x))

# ==========================================
# 4. 推理邏輯
# ==========================================
def chat():
    tokenizer = Tokenizer.from_file(config["vocab_name"])
    model = D2V22HybridModel(tokenizer.get_vocab_size(), config["d_model"], config["n_layers"]).to(device)
    
    if os.path.exists(config["save_model"]):
        ckpt = torch.load(config["save_model"], map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ 已載入權重 (Step: {ckpt.get('step', 'unknown')})")
    model.eval()

    temperature = 0.85         
    top_k = 20                 
    repetition_penalty = 1.3   

    while True:
        prompt = input("\n👤 你: ")
        if prompt.lower() in ['q', 'exit']: break
        
        input_ids = torch.tensor(tokenizer.encode(prompt).ids, device=device).unsqueeze(0)
        print("🤖 AI: ", end="")
        
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(50):
                logits = model(input_ids)[:, -1, :]
                
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty 
                
                logits = logits / temperature
                
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                word = tokenizer.decode([next_token.item()])
                print(word, end="", flush=True)
                
                if next_token.item() == 2: break 
        print()

if __name__ == "__main__":
    chat()