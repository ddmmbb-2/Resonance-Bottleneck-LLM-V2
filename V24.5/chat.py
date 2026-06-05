import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np

# ==================== 配置參數 ====================
config = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 12,
    "latent_dim": 256,
    "dropout": 0.1,
    "max_seq_len": 512,
    "batch_size": 1,           # 推理時固定為1
    "block_size": 256,
    "chunk_size": 64,
    "cache_capacity": 512,
    "think_steps": 5,
    "vocab_name": "bpe_tokenizer_v12.json",   # 請修改為實際路徑
    "vocab_size": 16384,
    "halt_tau": 0.05,
    "inference_exit_threshold": 0.85
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 基礎組件 ====================
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

# ==================== Phase 3 模組 ====================
class CausalKeyframeCache(nn.Module):
    def __init__(self, capacity, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.capacity = capacity
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.rope = RoPE(self.d_head, max_seq_len=max_seq_len)
        self.super_tokens = []

    def clear(self):
        self.super_tokens = []

    def add(self, vector, end_pos):
        self.super_tokens.append((vector.detach(), end_pos))
        if len(self.super_tokens) > self.capacity:
            self.super_tokens.pop(0)

    def get_all_as_kv(self, batch_size, current_start_pos):
        valid_tokens = [(v, p) for v, p in self.super_tokens if p <= current_start_pos]
        if not valid_tokens:
            return None
        vectors = torch.stack([v for v, _ in valid_tokens], dim=1)
        if vectors.shape[0] != batch_size:
            if vectors.shape[0] > batch_size:
                vectors = vectors[:batch_size]
            else:
                padding = vectors.new_zeros(batch_size - vectors.shape[0], *vectors.shape[1:])
                vectors = torch.cat([vectors, padding], dim=0)
        positions = torch.tensor([p for _, p in valid_tokens], device=vectors.device, dtype=torch.long)
        kv = self.kv_proj(vectors)
        k, v = kv.chunk(2, dim=-1)
        B, M, _ = k.shape
        k = k.view(B, M, self.n_heads, self.d_head)
        v = v.view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        cos = self.rope.cos[:, positions, :, :]
        sin = self.rope.sin[:, positions, :, :]
        k1, k2 = k.chunk(2, dim=-1)
        k_rot = torch.cat((-k2, k1), dim=-1)
        k = k * cos + k_rot * sin
        k = k.transpose(1, 2)
        return k, v

class CausalTokenSelector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, hidden_chunk, halt_probs):
        importance = 1.0 - halt_probs
        weights = importance / (importance.sum(dim=1, keepdim=True) + 1e-6)
        super_token = (hidden_chunk * weights).sum(dim=1)
        return self.proj(super_token)

# ==================== 核心 Attention ====================
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
        if external_kv is not None:
            k_ext, v_ext = external_kv
            q_ext = q.transpose(1, 2)
            out_cache = F.scaled_dot_product_attention(q_ext, k_ext, v_ext)
            out_cache = out_cache.transpose(1, 2).contiguous().view(B, L, -1)
            gate_c = torch.sigmoid(self.cache_gate(x))
            out_combined = out_local + gate_c * out_cache
        else:
            out_combined = out_local
        gate_val = F.silu(self.out_gate(latent))
        return self.dropout(self.proj(out_combined) * gate_val)

# ==================== 交叉注意力與海馬迴 ====================
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
        K = K_mem.transpose(1, 2)
        V = V_mem.transpose(1, 2)
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

# ==================== 推理核心 ====================
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

    def forward(self, x, external_kv=None, return_all_steps=False):
        B, L, D = x.shape
        h_latent_init = self.init_proj(x)
        h_latent = h_latent_init
        K_mem, V_mem = self.context_to_kv(x).chunk(2, dim=-1)
        K_mem = K_mem.view(B, L, config["n_heads"], -1)
        V_mem = V_mem.view(B, L, config["n_heads"], -1)
        K_mem = self.k_norm(K_mem)
        for i in range(self.steps):
            step_ids = torch.full((B,), i, device=x.device, dtype=torch.long)
            h_query = 0.6 * h_latent + 0.3 * h_latent_init + 0.1 * self.step_embed(step_ids).unsqueeze(1)
            K_step, V_step = K_mem, V_mem
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
            if not self.training:
                pred_halt_logit = self.exit_gate(h_next)
                if torch.sigmoid(pred_halt_logit).mean() > config["inference_exit_threshold"]:
                    break
            h_latent = h_next
        last_intermediate = self.latent_to_model(h_latent)
        if return_all_steps:
            return last_intermediate, [], [], []
        return last_intermediate

# ==================== SSM 塊 ====================
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

# ==================== Attention 塊 ====================
class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, config["latent_dim"])
        self.ffn = SwiGLU(d_model)
    def forward(self, x, external_kv=None):
        x = x + self.attn(x, external_kv)
        x = x + self.ffn(x)
        return x

# ==================== 主模型 ====================
class D2V20HybridModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        self.chunk_size = config["chunk_size"]
        self.cache = CausalKeyframeCache(config["cache_capacity"], d_model, config["n_heads"], max_seq_len=config["max_seq_len"])
        self.selector = CausalTokenSelector(d_model)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                self.blocks.append(ResonanceOptimizerCore(d_model, config["latent_dim"], config["think_steps"]))
            elif i in [0, 4, 8]:
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                self.blocks.append(D2V20SSMBlock(d_model))
        self.out_ln = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, L = x.shape
        chunk_size = self.chunk_size
        final_logits_list = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            x_chunk = x[:, start:end]
            ext_kv = self.cache.get_all_as_kv(B, current_start_pos=start)
            curr_x = self.emb_dropout(self.embedding(x_chunk))
            for idx, block in enumerate(self.blocks):
                if isinstance(block, ResonanceOptimizerCore):
                    out = block(curr_x, external_kv=ext_kv, return_all_steps=False)
                    curr_x = curr_x + out
                elif isinstance(block, D2V18AttentionBlock):
                    curr_x = block(curr_x, external_kv=ext_kv)
                else:
                    curr_x = block(curr_x)
            final_logits = self.head(self.out_ln(curr_x))
            final_logits_list.append(final_logits)
            # 生成 Super Token（推理時也可選擇不加入，這裡簡化，僅用於長文本記憶）
            # 為避免干擾，推理時暫不自動添加 Super Token
        return torch.cat(final_logits_list, dim=1)

# ==================== 對話生成器 ====================
class DialogueGenerator:
    def __init__(self, model_path, tokenizer_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        config['vocab_size'] = self.tokenizer.get_vocab_size()
        self.model = D2V20HybridModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers']
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ 模型載入完成 | 詞表大小: {config['vocab_size']} | 設備: {self.device}")

    def reset_cache(self):
        self.model.cache.clear()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated = input_ids.copy()
        self.reset_cache()

        for _ in range(max_new_tokens):
            logits = self.model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)

        response = self.tokenizer.decode(generated[len(input_ids):])
        return response

def main():
    # 請修改為您實際的模型路徑和 tokenizer 路徑
    model_path = "d2_v24_samba_latent.pth"
    tokenizer_path = "bpe_tokenizer_v12.json"

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        sys.exit(1)
    if not os.path.exists(tokenizer_path):
        print(f"❌ 找不到 tokenizer 檔案: {tokenizer_path}")
        sys.exit(1)

    generator = DialogueGenerator(model_path, tokenizer_path, device='cuda')
    print("\n🎤 對話測試模式啟動！輸入 'quit' 或 'exit' 結束。")

    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input.strip():
            continue
        print("🤖 模型思考中...")
        # 🚀 修改為這個嚴謹盲測版：
        response = generator.generate(
            user_input, 
            max_new_tokens=150, 
            temperature=0.2,  # 🔴 從 0.7 降到 0.2！大腦極度降溫，讓模型只敢挑機率最高的正統字
            top_k=5,          # 🔴 從 50 縮減到 5！候選字縮到最小，直接封殺那些奇奇怪怪的邊緣垃圾 Token
            top_p=0.85
        )
        print(f"🤖 助手: {response}")

if __name__ == "__main__":
    main()