# 🚀 Resonance-Bottleneck-LLM (V20-Phase 1.1)

> *Beyond attention: Latent resonance, recursive thinking, and a bulletproof global workspace.*

## 🧠 Overview | 概述

**Resonance-Bottleneck-LLM (V20)** 演進至 **"Global Workspace"** 階段。這是一個實驗性的 Transformer 架構，在繼承 **潛空間壓縮（Latent Bottleneck）**、**共振式注意力（Resonance Attention）** 與 **局部遞迴推理（Selective Recurrent Reasoning）** 的基礎上，V20 Phase 1.1 引入了具備工業級防護的 **靜態全域看板（Static Global Workspace）**。

這讓分散在不同層級的「推理大腦」首度擁有了跨層、跨 Token 共享與繼承資訊的能力，真正實現動態的工作記憶。

---

## ✨ Key Features | 核心特色

### 🔹 Static Global Workspace (New in V20!)
A dynamically updated "scratchpad" consisting of 8 tokens that floats alongside the batch. Reasoning Cores can now read from and write to this shared memory, allowing deep logic to be passed across different layers and reasoning steps without gradient explosion.
由 8 個 Token 組成的動態「草稿本」，隨著 Batch 在網路中流動。各層的推理核心（Reasoning Core）可以從這塊共享記憶體中讀寫資訊，讓深層邏輯能跨層傳遞，且不會引發梯度爆炸。

### 🔹 Industrial-Grade Memory Routing
Replaced simple mean-pooling with a highly robust **Attention Read & Write** mechanism. It features:
* **[h, w, h-w] Read Gates** for hyper-sensitive novelty detection.
* **Detached Write Gates** to isolate historical gradients and prevent routing collapse.
* **Temperature Clamping (0.3~3.0)** to stabilize attention focus.
* **Alpha Scaling & Workspace Norm** to eliminate value explosion and drift.

捨棄了粗暴的平均池化，全面升級為極度強健的 **注意力讀寫機制**。包含增強型特徵差異讀取門、隔離歷史梯度的寫入門、動態溫度鉗制，以及防止數值漂移的 Alpha 縮放與歸一化技術。

### 🔹 Selective Recurrent Reasoning
Specific layers are designated as **Reasoning Cores**. These layers use a recurrent loop (`think_steps=2`) to iteratively refine latent representations before passing them to the next block.
特定的層級被指定為 **推理核心（Reasoning Core）**。透過遞迴循環機制，模型能在這些層級進行多次迭代思考，強化複雜邏輯的處理能力。

### 🔹 Latent Bottleneck & Resonance Attention
Compresses information into a latent space, utilizing phase-aware resonance gating (amplitude, phase, and interference) instead of standard softmax.
將資訊壓縮至潛空間，並利用相位干涉與振幅門控進行共振式注意力計算，完全捨棄傳統的 softmax。

---

## 🏗️ Architecture | 模型架構 (V20-Stable Variant)

V20 採用交替結構，將推理層平均分佈，配置如下：

| Layer | Type | Description |
| :--- | :--- | :--- |
| **Layer 0, 1, 2** | D2V18 Attention / Conv | V18.1 Resonance Attention & Causal 1D Conv |
| **Layer 3** | **Reasoning Core V20** ⭐ | **Recurrent Thinking + Workspace I/O** |
| **Layer 4, 5, 6** | D2V18 Attention / Conv | Standard blocks |
| **Layer 7** | **Reasoning Core V20** ⭐ | **Recurrent Thinking + Workspace I/O** |
| **...** | ... | ... |
| **Layer 11** | **Reasoning Core V20** ⭐ | **Recurrent Thinking + Workspace I/O** |

![png](V20.png)

---

## ⚙️ Training Setup | 訓練設定 (V20-Phase 1.1)

* **Model Size**: 512 dim / 12 layers
* **Attention Heads**: 8
* **Latent Dim**: 256
* **Workspace Tokens**: 8
* **Thinking Steps**: 2 (at Layers 3, 7, 11)
* **Context Length**: 512 (RoPE supported)
* **Vocab Size**: 16,384
* **Optimization**: AdamW + Warmup Scheduler + Autocast (BFloat16) + **Dual-Checkpointing Strategy**

---

## 📊 Design Motivation | 設計動機

The V20 update explores **"Stateful Thinking in Latent Space"**:

> **V19 proved that local recursive thinking works. V20 connects those isolated thoughts with a global memory board, complete with the rigorous mathematical safeguards needed to tame dynamic memory networks.**

V20 的設計核心在於「具備狀態的潛空間思考」：如果說 V19 證明了局部遞迴推理是可行的，那麼 V20 則是為這些孤立的思考節點連上了一塊全域記憶板，並配備了馴服動態記憶網路所需的嚴謹數學防線。

---

## 🚧 Status | 開發狀態

* [x] **V19 Core**: Reasoning Core & Latent Modulation.
* [x] **V20 Phase 1.1**: Industrial-Grade Global Workspace Integration.
* [x] **Gate Monitoring**: Real-time tracking of reasoning core activity and memory gating.
* [x] **Time-Machine Backups**: Automated historical checkpointing to prevent training collapse.
* [ ] Phase 2: Shadow Adaptive Depth (Early Exit routing).
* [ ] Phase 3: Span Compression & Phase Recall.

---

## 📜 License

MIT License

---

## ⭐ Support the Project

If you find this "Resonance + Reasoning + Memory" approach interesting, please give us a ⭐!
