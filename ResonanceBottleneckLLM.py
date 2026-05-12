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
# 🎯 V20-Adaptive 實驗配置 (修復 Early Collapse 版本)
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
    "accum_steps": 8,        
    "think_steps": 2,        
    "lr": 3e-4,              
    "epochs": 100000,        
    "warmup_steps": 1000,    
    
    # 👇 1. 改成你剛剛產生的新語料庫檔名
    "bin_data": "corpus_v20_twllm.bin", 
    
    # 👇 2. 換一個新的存檔名稱！(超級重要，避免覆蓋舊的心血，也強迫模型從 0 開始)
    "save_model": "d2_v20_twllm_adaptive.pth", 
    
    # 👇 3. 換一個新的日誌名稱
    "log_csv": "v20_twllm_adaptive_log.csv",   
    
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,
    
    # 🎯 Phase 2 影子門控超參數
    "halt_tau": 0.05,                  
    "halt_weight": 0.5,                
    "inference_exit_threshold": 0.85   
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V20-Adaptive 自適應深度版 (效能獎勵掛載) 啟動中 | 設備: {device}")

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
        writer.writerow(["Step", "CE_Loss", "Halt_Loss", "LR", "Gate_Values"])

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
# 3. 核心 Attention
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
# 4. V20 推理模塊與主模型 (新增效能獎勵機制)
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
        
        self.init_proj = nn.Linear(d_model, latent_dim)
        
        self.gamma = nn.Parameter(torch.ones(latent_dim) * 1e-4)
        self.exit_gate = nn.Linear(latent_dim, 1) 
        
        self.last_halt_losses = []
        self.register_buffer("avg_gate_val", torch.zeros(1))
        self.register_buffer("avg_diff", torch.zeros(1)) 
        self.register_buffer("avg_halt_prob", torch.zeros(1))

    def _step(self, x, h_latent, step_idx):
        step_ids = torch.full((x.size(0),), step_idx, device=x.device, dtype=torch.long)
        mod = self.step_modulator(step_ids).unsqueeze(1) 
        scale, bias = mod.chunk(2, dim=-1)
        
        h_input = h_latent * (1.0 + scale) + bias
        h_query = self.latent_to_model(h_input)
        
        delta_model = self.reason_attn.forward_with_context(context=x, query=h_query)
        delta_latent = self.norm(self.model_to_latent(delta_model))
        
        delta_latent_clamped = torch.clamp(delta_latent, min=-4.0, max=4.0)
        
        gate_val = torch.sigmoid(self.gate(torch.cat([h_latent, delta_latent_clamped], dim=-1)) * 1.2)
        h_next = h_latent + self.gamma * (gate_val * torch.tanh(delta_latent_clamped))

        # 🎯 核心修改：效能獎勵與懲罰機制 (加入維度標準化)
        raw_diff = torch.norm(delta_latent_clamped.detach(), p=2, dim=-1, keepdim=True)
        # 縮放到單一維度的平均變化量 (將原本的 12 變成約 0.75)
        diff_norm = raw_diff / math.sqrt(config["latent_dim"])
        
        if self.training:
            # 1. 懶惰懲罰：低於 0.1 嚴厲懲罰
            lazy_penalty = torch.exp(-diff_norm * 10.0)
            
            # 2. 混亂懲罰：高於 1.5 才懲罰 (模型現在是 0.75，非常安全)
            chaos_penalty = F.relu(diff_norm - 1.5)
            
            # 3. 有效思考獎勵：黃金區間設在 0.7
            reward = -torch.exp(-((diff_norm - 0.7)**2) / 0.1)
            
            # 合併為推理品質 Loss
            quality_loss = (3.0 * lazy_penalty + 1.0 * chaos_penalty + 1.5 * reward).mean()
        else:
            quality_loss = 0.0

        # 改進的 Target Halt：如果 diff_norm 是健康的 0.75，target 大約是 0.08 (繼續思考)
        # 如果模型又想偷懶變成 0.0，target 就會變成 1.0 (強制退出)
        target_halt = torch.exp(-diff_norm / 0.3)
        
        pred_halt_logit = self.exit_gate(h_next)
        pred_halt = torch.sigmoid(pred_halt_logit)
        
        base_halt_loss = F.binary_cross_entropy_with_logits(pred_halt_logit, target_halt)
        
        # 增加熵損失 (Entropy Regularization) 防止預測值極化
        entropy_loss = - (pred_halt * torch.log(pred_halt + 1e-8) + (1 - pred_halt) * torch.log(1 - pred_halt + 1e-8)).mean()
        
        # 總結這一層的附屬 Loss
        total_step_loss = base_halt_loss + quality_loss - 0.02 * entropy_loss
        self.last_halt_losses.append(total_step_loss)
        
        if self.training:
            self.avg_gate_val = 0.9 * self.avg_gate_val + 0.1 * gate_val.detach().mean()
            # 這裡記錄標準化後的 diff_norm，這樣你在畫面上看到的就會是 0.75 左右
            self.avg_diff = 0.9 * self.avg_diff + 0.1 * diff_norm.mean()
            self.avg_halt_prob = 0.9 * self.avg_halt_prob + 0.1 * pred_halt.detach().mean()
            
            # 加入微小噪聲，打破局部最優
            h_next = h_next + torch.randn_like(h_next) * 1e-4 
            
        return h_next, pred_halt

    def forward(self, x):
        h_latent = self.init_proj(x)
        self.last_halt_losses = [] 
        
        for i in range(self.steps):
            if self.training:
                h_latent, s = checkpoint(self._step, x, h_latent, i, use_reentrant=False)
            else:
                h_latent, s = self._step(x, h_latent, i)
                if s.min() > config["inference_exit_threshold"]:
                    break
                    
        return self.latent_to_model(self.norm(h_latent))

class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, latent_dim=config["latent_dim"])
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])
        self.gamma_1 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.gamma_2 = nn.Parameter(torch.ones(d_model) * 1e-4)
        
    def forward(self, x):
        x = x + self.gamma_1 * self.attn(x)
        x = x + self.gamma_2 * self.ffn(x)
        return x

class D2V18ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.conv = CausalConv1d(d_model)
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])
        self.gamma_1 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.gamma_2 = nn.Parameter(torch.ones(d_model) * 1e-4)
        
    def forward(self, x):
        x = x + self.gamma_1 * self.conv(self.ln(x))
        x = x + self.gamma_2 * self.ffn(x)
        return x

class D2V19StableModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                self.blocks.append(ResonanceReasoningCore(d_model, config["latent_dim"], config["think_steps"]))
            elif i % 2 == 0:
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                self.blocks.append(D2V18ConvBlock(d_model))
                
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 

    def get_halt_loss(self):
        total_halt_loss = 0
        count = 0
        for block in self.blocks:
            if isinstance(block, ResonanceReasoningCore):
                if block.last_halt_losses:
                    valid_losses = block.last_halt_losses[:config["think_steps"]]
                    total_halt_loss += torch.stack(valid_losses).mean()
                    count += 1
        return total_halt_loss / count if count > 0 else 0
        
    def forward(self, x):
        x = self.emb_dropout(self.embedding(x))
        for block in self.blocks:
            if isinstance(block, ResonanceReasoningCore):
                x = x + block(x) 
            else:
                x = block(x)
        return self.head(self.out_ln(x))

# ==========================================
# 5. 訓練與監控迴圈
# ==========================================
model = D2V19StableModel(config["vocab_size"], config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # === 👇 把這段「外科手術」註解掉或刪除 👇 ===
    # print("💉 執行外科手術重置：正在打破局部最優，喚醒推理層...")
    # with torch.no_grad():
    #     for i in [3, 7, 11]: 
    #         block = model.blocks[i]
    #         block.exit_gate.bias.fill_(-1.0) 
    #         block.exit_gate.weight.data *= 0.1
    #         block.gamma.fill_(0.02)
    #         block.gate.weight.data += torch.randn_like(block.gate.weight.data) * 0.02
    # print("✅ 喚醒完成！推理層已注射活力藥劑。")
    # === 👆 把這段「外科手術」註解掉或刪除 👆 ===

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)

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
print(f"🌟 模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="訓練中")

smoothed_ce = None

while global_step < config["epochs"]:
    optimizer.zero_grad(set_to_none=True)
    step_ce_loss = 0 
    step_halt_loss = 0

    # 🎯 損失驅動的課程學習 (Loss-Driven Curriculum)
    if smoothed_ce is None:
        current_halt_weight = 0.0
    else:
        loss_upper = 4.5  # 【門檻一】當平滑 CE 降到 4.5 時，門控開始微微甦醒
        loss_lower = 3.0  # 【門檻二】當平滑 CE 降到 3.0 時，門控火力全開 (完全體)
        
        if smoothed_ce >= loss_upper:
            progress = 0.0
        elif smoothed_ce <= loss_lower:
            progress = 1.0
        else:
            # 在 4.5 到 3.0 之間畫一條平滑的溜滑梯
            progress = (loss_upper - smoothed_ce) / (loss_upper - loss_lower)
            
        current_halt_weight = config["halt_weight"] * progress

    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(xb)
            
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            halt_loss = model.get_halt_loss()
            
            combined_loss = ce_loss + (current_halt_weight * halt_loss)
            loss_to_back = combined_loss / config["accum_steps"]
        
        loss_to_back.backward()
        
        step_ce_loss += ce_loss.item()
        step_halt_loss += halt_loss.item() if isinstance(halt_loss, torch.Tensor) else halt_loss

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    # 1. 計算這一步的平均 Loss
    avg_ce = step_ce_loss / config["accum_steps"]
    avg_halt = step_halt_loss / config["accum_steps"]
    
    # 2. 更新排程器與步數
    warmup_scheduler.step()
    global_step += 1
    
    # 3. 計算平滑 Loss (避震器)
    if smoothed_ce is None:
        smoothed_ce = avg_ce
    else:
        smoothed_ce = 0.99 * smoothed_ce + 0.01 * avg_ce 
    
    # 4. 抓取觀測數值
    diffs = [b.avg_diff.item() for b in model.blocks if isinstance(b, ResonanceReasoningCore)]
    halts = [b.avg_halt_prob.item() for b in model.blocks if isinstance(b, ResonanceReasoningCore)]
    
    diff_str = f"[{','.join([f'{d:.2f}' for d in diffs])}]" if diffs else "N/A"
    halt_str = f"[{','.join([f'{h:.2f}' for h in halts])}]" if halts else "N/A"

    # 5. 更新進度條
    pbar.update(1)
    pbar.set_postfix({
        "CE": f"{avg_ce:.3f}", 
        "sCE": f"{smoothed_ce:.3f}", 
        "QL": f"{avg_halt:.3f}", 
        "D": diff_str,      
        "P": halt_str      
    })

    # 6. 存檔與記錄
    if global_step % 10 == 0:
        with open(config["log_csv"], mode='a', newline='') as f:
            writer = csv.writer(f)
            # CSV 也可以順便把 smoothed_ce 存下來，方便以後畫圖分析！
            writer.writerow([global_step, f"{avg_ce:.4f}", f"{smoothed_ce:.4f}", f"{avg_halt:.4f}", f"{optimizer.param_groups[0]['lr']:.6f}", diff_str, halt_str])

    if global_step % 1000 == 0:
        ckpt = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'smoothed_ce': smoothed_ce  # 把平滑 Loss 也存進去，這樣中斷重啟時才不會斷層
        }
        torch.save(ckpt, config["save_model"])
        backup_path = config["save_model"].replace(".pth", f"_step_{global_step}.pth")
        shutil.copy2(config["save_model"], backup_path)
