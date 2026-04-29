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
# 🚀 V18.1: RoPE + Production-Ready 配置
# ==========================================
config = {
    "d_model": 768,          
    "n_heads": 12,           
    "n_layers": 24,          
    "latent_dim": 512,       
    "dropout": 0.1,          
    "max_seq_len": 2048,     # 🌟 RoPE 允許我們設定更長的外推上限
    "batch_size": 8,         
    "block_size": 512,
    "accum_steps": 4,        # ⬅️ 訓練必備參數補回
    "lr": 1e-4,              # ⬅️ 學習率補回
    "epochs": 100000,        # ⬅️ 總步數補回
    "warmup_steps": 2000,    # ⬅️ 暖身步數補回
    "bin_data": "corpus_v15_clean.bin", 
    "save_model": "d2_v18_resonance_pro.pth", 
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,                      
}





import os
import numpy as np
from tokenizers import Tokenizer

# 🌟 1. 定義 Device (解決 NameError 的元兇)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V18.1 啟動中 | 設備: {device}")

# 🌟 2. 資料加載模塊 (解決等一下會發生的 get_batch 找不到的問題)
if not os.path.exists("corpus_v15_clean.bin"):
    raise FileNotFoundError(f"❌ 找不到 corpus_v15_clean.bin！請確認檔案位置。")

tokenizer = Tokenizer.from_file("bpe_tokenizer_v12.json")
vocab_size = tokenizer.get_vocab_size() 

data = np.memmap("corpus_v15_clean.bin", dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([torch.from_numpy(data[i:i+config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+config["block_size"]+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# 🌟 V18 Block 封裝：整合殘差結構與 SwiGLU
class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 注意力層 (內部自帶 RMSNorm)
        self.attn = LatentResonanceAttentionV18(
            d_model, 
            latent_dim=config["latent_dim"], 
            dropout=config["dropout"]
        )
        # 前饋網路層 (內部自帶 RMSNorm)
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])

    def forward(self, x):
        # 殘差連接：x + Sublayer(x)
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class D2V18ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.conv = CausalConv1d(d_model)
        # 卷積層通常也搭配一個 FFN 來提升表達能力
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])

    def forward(self, x):
        # 卷積殘差：注意 CausalConv1d 內部沒有 Norm，這裡手動加上
        x = x + self.conv(self.ln(x))
        x = x + self.ffn(x)
        return x


# 🌟 新增：RoPE 旋轉位置編碼實作
class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len=2048):
        super().__init__()
        # 預計算旋轉頻率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        # 構建 sin/cos 表
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, :, None, :]) # [1, L, 1, D_head]
        self.register_buffer("sin", emb.sin()[None, :, None, :])

    def forward(self, x):
        # x: [B, L, H, D_head]
        L = x.shape[1]
        cos, sin = self.cos[:, :L, :, :], self.sin[:, :L, :, :]
        # 旋轉矩陣運算：[x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rot * sin

# 🌟 RMSNorm (維持 V17.5 優良設計)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_fp32 = x.float() 
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(x.dtype)

# 🌟 CausalConv1d (相對位置感知的輔助)
class CausalConv1d(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., :-2].transpose(1, 2)

# 🌟 SwiGLU
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

# 🌟 V18.4 修正版：修復 Attention 方向 Bug、引入 Hard-Gate 與 Sigmoid Phase
class LatentResonanceAttentionV18(nn.Module):
    def __init__(self, d_model, latent_dim=512, dropout=0.1):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        self.ln = RMSNorm(d_model)
        self.latent_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.qkv_expand = nn.Linear(latent_dim, d_model * 3, bias=False)
        self.reso_expand = nn.Linear(latent_dim, self.n_heads * 4, bias=False) 
        
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        
        # 👇 新增這行：初始化 RoPE
        self.rope = RoPE(self.d_head, max_seq_len=config["max_seq_len"])
        
        self.out_gate = nn.Linear(latent_dim, d_model, bias=False)
        self.head_decay = nn.Parameter(torch.linspace(-3.0, 1.0, self.n_heads))
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x) 
        latent = F.silu(self.latent_compress(x_norm))
        
        q, k, v = self.qkv_expand(latent).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head)
        k = k.view(B, L, self.n_heads, self.d_head)
        v = v.view(B, L, self.n_heads, self.d_head)
        
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        
        q_f = F.elu(q.float()) + 1.0
        k_f = F.elu(k.float()) + 1.0
        v_f = v.float()
        
        params = self.reso_expand(latent).view(B, L, self.n_heads, 4)
        sem_amp, sem_phase, ctx_amp, ctx_phase = params.unbind(-1)
        
        sem_amp = torch.sigmoid(sem_amp)
        ctx_amp = torch.sigmoid(ctx_amp)
        
        # 🎯 修改 1：Phase 改用 Sigmoid，避免 tanh 鎖死，且 [0, pi] 對 cos 已足夠
        sem_phase = torch.sigmoid(sem_phase) * math.pi
        ctx_phase = torch.sigmoid(ctx_phase) * math.pi
        
        raw_decay = 0.3 + 0.65 * torch.sigmoid(self.head_decay.view(1, 1, self.n_heads))
        decay_rate = torch.clamp(raw_decay, min=1e-5, max=0.999)
        
        dt_kv = (1.0 - decay_rate).unsqueeze(-1).unsqueeze(-1)
        dt_z = (1.0 - decay_rate).unsqueeze(-1)
        
        cos_diff = torch.cos(sem_phase - ctx_phase)
        base_gate = torch.sigmoid((sem_amp * ctx_amp * cos_diff) * self.temperature) 
        
        # 🎯 修改 4：引入 Hard-Gate 技巧，強制拉大對比度
        gate = torch.clamp(base_gate * 1.2 - 0.1, min=0.05, max=0.95)
        
        # K(d, 1) @ V(1, d) -> KV(d, d) 外積矩陣
        kv_input = (k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)) * gate.unsqueeze(-1).unsqueeze(-1)
        kv_input = kv_input * dt_kv 
        z_input = k_f * dt_z

        kv_input_f32 = kv_input.float()
        z_input_f32 = z_input.float()
        
        log_decay = torch.log(decay_rate).unsqueeze(-1) 
        cum_log_decay = torch.cumsum(log_decay.expand(B, L, -1, -1), dim=1) 
        
        safe_df_z = torch.exp(cum_log_decay) + 1e-8 
        safe_df_kv = safe_df_z.unsqueeze(-1)        
        
        kv_states = torch.cumsum(kv_input_f32 / safe_df_kv, dim=1) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = torch.cumsum(z_input_f32 / safe_df_z, dim=1) * torch.exp(cum_log_decay)

        kv_states = kv_states.to(x.dtype)
        z_states = z_states.to(x.dtype)
        
        # 🎯 修改 2：修正 Attention 計算方向！ Q @ (KV)
        # q_f.unsqueeze(-2) 變成 [B, L, H, 1, D]
        # kv_states 是 [B, L, H, D, D]
        # 矩陣相乘後變成 [B, L, H, 1, D]，再 squeeze 掉倒數第 2 維度
        out_num = (q_f.unsqueeze(-2) @ kv_states).squeeze(-2) # [B, L, H, D]

        # 🎯 修改 5：保持分母穩定性 (已有)
        den = (q_f * z_states).sum(dim=-1).unsqueeze(-1) # [B, L, H, 1]
        den = torch.clamp(den, min=1e-5) 
        
        out = out_num / den 
        out = out.contiguous().view(B, L, D) 
        
        out = self.mem_norm(out)
        
        gate_val = F.silu(self.out_gate(latent))
        out_proj = self.proj(out)
        return self.dropout(out_proj * gate_val)

# ==========================================
# 🌟 主模型實作更新
# ==========================================
class D2V18Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        
        self.blocks = nn.ModuleList()
        # 利用交替層優勢：Attention 負責全局語意，Conv 負責局部與相對位置
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                self.blocks.append(D2V18ConvBlock(d_model))
                
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 
        
    def forward(self, x):
        x = self.emb_dropout(self.embedding(x))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return self.head(self.out_ln(x))

# ==========================================
# 3. 訓練循環 (V17 極簡純粹版)
# ==========================================
model = D2V18Model(config["vocab_size"], config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)

for param_group in optimizer.param_groups:
    param_group['lr'] = config["lr"]
    param_group['initial_lr'] = config["lr"]

print("-" * 30)
print(f"DEBUG: Config 設定值應該是: {config['lr']}")
print(f"DEBUG: Optimizer 當前實際 LR: {optimizer.param_groups[0]['lr']:.2e}")
print("-" * 30)

restart_step = global_step 
# 🎯 修正：傳入 last_epoch=global_step，讓 Scheduler 知道目前的真實進度
warmup_scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / config["warmup_steps"]), last_epoch=global_step)

print(f"🌟 V17 模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"🚀 已喚醒 V17 模型！層數加深至 {config['n_layers']} 層，準備起飛！")

model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V17 訓練中")
has_decayed_42 = False

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
    
    # 🎯 修正：直接看 global_step，超過 2000 步就不再觸發暖身
    if global_step < config["warmup_steps"]:
        warmup_scheduler.step()
        
    elif avg_loss < 4.2 and not has_decayed_42:
        print(f"\n📉 [階段降速] Loss 跌破 4.2！收尾降速啟動，LR 乘以 0.7")
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.7
        has_decayed_42 = True

    elif global_step % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        # 方案 A：把 7.5 提高到 9.5 或 10.0
        # 方案 B：加上步數限制，例如 global_step > 5000 才啟動
        if avg_loss > 9.5 and current_lr > 1e-5 and global_step > 5000:
            print(f"\n🚨 [守門員] 偵測到 Loss 暴衝！緊急降速")
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5

    global_step += 1
    
    if global_step % 10 == 0:
        log_file = "train_log_v18.csv"
        current_lr = optimizer.param_groups[0]['lr']
        
        file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0
        with open(log_file, "a", encoding="utf-8") as f:
            if not file_exists: 
                f.write("step,loss,lr\n")
            f.write(f"{global_step},{avg_loss:.6f},{current_lr:.2e}\n")

    pbar.update(1)
    pbar.set_postfix({
        "Loss": f"{avg_loss:.4f}", 
        "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
    })

    if global_step % 2000 == 0:
        ckpt = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }
        backup_name = f"d2_v18_step_{global_step}.pth"
        torch.save(ckpt, backup_name)  
        torch.save(ckpt, config["save_model"])  
        print(f"🚩 Step {global_step} 存檔成功！已建立備份：{backup_name}")