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

# ==========================================
# 🎯 V20 Phase 1 實驗配置 (Global Workspace 引入)
# ==========================================
config = {
    "d_model": 512,          
    "n_heads": 8,            
    "n_layers": 12,          
    "latent_dim": 256,       
    "workspace_tokens": 8,   # 🌐 新增：全局看板的 Token 數量
    "dropout": 0.1,          
    "max_seq_len": 512,      
    "batch_size": 8,         
    "block_size": 256,       
    "accum_steps": 8,        
    "think_steps": 2,        
    "lr": 3e-4,              
    "epochs": 100000,        
    "warmup_steps": 1000,    
    "bin_data": "corpus_v17_mixed.bin", 
    "save_model": "d2_v20_phase1.pth",   # 🗂️ 升級存檔名稱
    "log_csv": "v20_phase1_log.csv",     # 📊 升級日誌名稱
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,                      
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V20 Phase 1 啟動中 | 設備: {device} | 導入 Global Workspace")

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
        writer.writerow(["Step", "Loss", "LR", "Gate_Values"])

# ==========================================
# 2. 基礎組件 (維持不變)
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
# 3. 核心 Attention (維持不變)
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
        
        kv_states = torch.cumsum(kv_input.float() / safe_df_kv, dim=1) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = torch.cumsum(z_input.float() / safe_df_z, dim=1) * torch.exp(cum_log_decay)

        out_num = (q_f.unsqueeze(-2) @ kv_states.to(x.dtype)).squeeze(-2) 
        den = torch.clamp((q_f * z_states.to(x.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5) 
        
        out = self.mem_norm((out_num / den).contiguous().view(B, L, D))
        gate_val = F.silu(self.out_gate(latent))
        return self.dropout(self.proj(out) * gate_val)

# ==========================================
# 4. V20 推理模塊與主模型 (🔥 Phase 1 重構核心)
# ==========================================
class ResonanceReasoningCoreV20(nn.Module):
    def __init__(self, d_model, latent_dim, think_steps=2):
        super().__init__()
        self.steps = think_steps
        self.latent_dim = latent_dim
        
        # --- 原本的 Reasoning 網路 ---
        self.step_modulator = nn.Embedding(think_steps, latent_dim * 2)
        self.latent_to_model = nn.Linear(latent_dim, d_model, bias=False)
        self.model_to_latent = nn.Linear(d_model, latent_dim, bias=False)
        self.reason_attn = LatentResonanceAttentionV18(d_model, latent_dim)
        self.gate = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = RMSNorm(latent_dim)
        self.init_proj = nn.Linear(d_model, latent_dim)
        self.register_buffer("avg_gate_val", torch.zeros(1))
        
        # ==========================================
        # 🌐 V20 Workspace 核心組件 (高階穩定版)
        # ==========================================
        
        # 🔍 Read Attention Q, K, V
        self.read_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.read_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.read_v = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # ✍️ Write Attention Q, K, V
        self.write_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.write_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.write_v = nn.Linear(latent_dim, latent_dim, bias=False)
        self.write_update = nn.Linear(latent_dim, latent_dim)

        # 💡 升級 2: Read Gate 輸入維度增強 (h, w, h-w)
        self.read_gate = nn.Linear(latent_dim * 3, latent_dim)
        self.write_gate = nn.Linear(latent_dim * 2, latent_dim)
        
        # 降溫初始化
        nn.init.constant_(self.write_gate.bias, -2.0)

        # 💡 升級 1: 可學習的寫入縮放係數 (初始值 0.5)
        self.work_alpha = nn.Parameter(torch.tensor(0.5))

        self.read_temp = nn.Parameter(torch.ones(1))
        self.write_temp = nn.Parameter(torch.ones(1))

        self.work_norm = RMSNorm(latent_dim)

    def _step(self, x, h_latent, workspace, step_idx):
        # ==========================================
        # 📖 1. Attention Read
        # ==========================================
        q_r = self.read_q(h_latent)  
        k_r = self.read_k(workspace) 
        v_r = self.read_v(workspace) 
        
        # 💡 升級 4: Temperature Clamping (0.3 ~ 3.0)
        safe_read_temp = torch.clamp(F.softplus(self.read_temp), 0.3, 3.0)
        scale_r = math.sqrt(self.latent_dim) * safe_read_temp
        read_attn = F.softmax((q_r @ k_r.transpose(-2, -1)) / scale_r, dim=-1)
        work_context = read_attn @ v_r  
        
        # 💡 升級 2: 增強版 Read Gate 輸入
        diff_feat = h_latent - work_context
        r_gate = torch.sigmoid(self.read_gate(torch.cat([h_latent, work_context, diff_feat], dim=-1)))
        h_latent = h_latent + r_gate * work_context
        
        # ==========================================
        # 🧠 2. 核心 Reasoning 邏輯 
        # ==========================================
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

        # ==========================================
        # ✍️ 3. Per-Slot Attention Write 
        # ==========================================
        q_w = self.write_q(workspace) 
        k_w = self.write_k(h_next)    
        v_w = self.write_v(h_next)    
        
        # 💡 升級 4: Temperature Clamping (0.3 ~ 3.0)
        safe_write_temp = torch.clamp(F.softplus(self.write_temp), 0.3, 3.0)
        scale_w = math.sqrt(self.latent_dim) * safe_write_temp
        write_attn = F.softmax((q_w @ k_w.transpose(-2, -1)) / scale_w, dim=-1)
        
        update_candidate = torch.tanh(self.write_update(write_attn @ v_w))
        
        # 💡 升級 3: Write Gate 使用 detach() 隔離歷史梯度
        w_gate = torch.sigmoid(self.write_gate(torch.cat([workspace.detach(), update_candidate], dim=-1)))
        
        # 💡 升級 1: 加入 work_alpha 縮放防爆炸
        workspace_next = workspace + self.work_alpha * w_gate * update_candidate
        
        # 🛡️ Norm 防漂移
        workspace_next = self.work_norm(workspace_next)
        
        # 💡 升級 5: 加入 Workspace 泛化噪聲
        if self.training:
            workspace_next = workspace_next + torch.randn_like(workspace_next) * 0.003
            
        return h_next, workspace_next

    def forward(self, x, workspace):
        h_latent = self.init_proj(x)
        for i in range(self.steps):
            h_latent, workspace = checkpoint(self._step, x, h_latent, workspace, i, use_reentrant=False)
        return self.latent_to_model(self.norm(h_latent)), workspace

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

class D2V20Phase1Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        
        # 🌐 初始化全域學習的 Workspace 基礎權重
        self.workspace_base = nn.Parameter(torch.randn(config["workspace_tokens"], config["latent_dim"]) * 0.02)
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                self.blocks.append(ResonanceReasoningCoreV20(d_model, config["latent_dim"], config["think_steps"]))
            elif i % 2 == 0:
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                self.blocks.append(D2V18ConvBlock(d_model))
                
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 
        
    def forward(self, x):
        B = x.size(0)
        x = self.emb_dropout(self.embedding(x))
        
        # 🌐 將 Workspace 擴展至目前的 Batch Size (B, 8, latent_dim)
        workspace = self.workspace_base.unsqueeze(0).expand(B, -1, -1).clone()
        
        for block in self.blocks:
            if isinstance(block, ResonanceReasoningCoreV20):
                # 推理層：同時傳入 x 與 workspace，並接收更新後的 out_x 與 workspace
                out_x, workspace = checkpoint(block, x, workspace, use_reentrant=False)
                x = x + out_x
            else:
                x = block(x)
                
        return self.head(self.out_ln(x))

# ==========================================
# 5. 訓練與監控迴圈
# ==========================================
model = D2V20Phase1Model(config["vocab_size"], config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)
else:
    # 💡 繼承 V19 權重的小技巧：如果你想拿 V19-Stable 的權重來 Fine-tune V20
    v19_path = "d2_v19_stable.pth"
    if os.path.exists(v19_path):
        print(f"🔄 偵測到 V19-Stable 權重，嘗試無痛繼承參數...")
        ckpt = torch.load(v19_path, map_location=device, weights_only=True)
        # strict=False 允許載入時略過新增加的 Workspace 權重
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("✅ 成功繼承 V19 主幹權重！新增的 Workspace 門控將從頭學習。")

for param_group in optimizer.param_groups:
    param_group['initial_lr'] = config["lr"]
    param_group['lr'] = config["lr"]

def get_lr_multiplier(step):
    if step < config["warmup_steps"]:
        return (step + 1) / config["warmup_steps"]
    decay_steps = config["epochs"] - config["warmup_steps"]
    current_decay_step = step - config["warmup_steps"]
    min_lr_ratio = 0.1 
    cosine_decay = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * current_decay_step / decay_steps))
    return cosine_decay

warmup_scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=global_step)
print(f"🌟 V20 Phase 1 模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V20 Phase 1 訓練中")

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
    warmup_scheduler.step()
    global_step += 1
    
    gate_vals = [b.avg_gate_val.item() for b in model.blocks if isinstance(b, ResonanceReasoningCoreV20)]
    gate_str = f"[{','.join([f'{g:.3f}' for g in gate_vals])}]" if gate_vals else "N/A"
    current_lr = optimizer.param_groups[0]['lr']

    pbar.update(1)
    pbar.set_postfix({
        "Loss": f"{avg_loss:.4f}", 
        "LR": f"{current_lr:.1e}",
        "Gate": gate_str
    })

    if global_step % 10 == 0:
        with open(config["log_csv"], mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, f"{avg_loss:.4f}", f"{current_lr:.6f}", gate_str])

    # 💾 每 2000 步自動存檔權重與備份
    if global_step % 2000 == 0:
        ckpt = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }
        
        # 1️⃣ 儲存最新進度 (覆蓋原本的 d2_v20_phase1.pth，方便腳本自動接續)
        torch.save(ckpt, config["save_model"])  
        
        # 2️⃣ 另存一份帶有步數的獨立備份檔案 (防止模型崩潰時覆蓋掉好權重)
        backup_model_name = config["save_model"].replace(".pth", f"_step_{global_step}.pth")
        torch.save(ckpt, backup_model_name)
        
        print(f"\n🚩 Step {global_step} 存檔成功！已記錄至 {config['log_csv']}。Gate: {gate_str}")
        print(f"   👉 最新進度: {config['save_model']}")
        print(f"   👉 歷史備份: {backup_model_name}")