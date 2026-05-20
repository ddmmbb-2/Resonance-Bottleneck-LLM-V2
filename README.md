# 🚀 Resonance-Bottleneck-LLM (V24-InfiniteResonance)

> *Not just predicting tokens — but recursively refining latent cognition and dynamically condensing infinite context through resonance.*

## 🧠 Overview | 概述

**Resonance-Bottleneck-LLM (V24-InfiniteResonance)** 代表整體架構從「單次長序列建模器」正式進化為一種 **具備無限記憶潛力的分塊遞推系統 (Chunk-wise Recurrent Reasoning System)**。

V24 延續了 V23 的神經動力學推理核心，並強勢引入了 **Phase 3: Micro-Chunking & Causal Keyframe Cache (微型分塊與因果關鍵影格緩存)** 機制。模型不再受限於傳統的注意力視窗長度，而是將過去的記憶動態濃縮並無縫注入到當前的推理步驟中。

此次版本最大的突破，在於正式引入：

* 🧩 **Micro-Chunking 遞推處理**
* 🌌 **Super Token 動態濃縮與重要性過濾**
* ⏱️ **絕對位置感知 (RoPE-Aware) 外部緩存**
* ⚡ **跨 Chunk 快取注入 (FlashAttention-2 加速)**

模型現在具備了「工作記憶 + 長期聯想 + 內部修正 + **跨時空外部記憶**」的全方位認知特性。

---

# ✨ Key Features | 核心特色

## 🔹 Causal Keyframe Cache & Super Tokens (因果關鍵影格與超級符號)

V24 打破了上下文長度限制，引入了全新的動態記憶濃縮機制：

* **Micro-Chunking**
將長文本切分為微小區塊 (如 `chunk_size=64`) 依序處理，極大化降低 VRAM 消耗。
* **Dynamic Super Token Condensation**
根據 Neuro-Optimizer 的「思考深度 (Halt Probability)」，動態計算 Chunk 內每個 Token 的重要性。只有當整體重要性超過閾值 (如 `> 0.3`) 時，才會將該 Chunk 融合成一個高密度的 **Super Token**。
* **Strict Causality & RoPE Alignment**
外部記憶嚴格記錄 `end_pos` 並結合 RoPE (旋轉位置編碼)，確保模型在檢索歷史 Super Tokens 時擁有完美的絕對位置感知與因果隔離，絕不洩漏未來資訊。

---

## 🔹 Neuro-Sandwich Architecture (神經三明治架構)

V24 採用非對稱混合堆疊架構，並將 **External Cache** 貫穿其中：

* **Resonance Attention + Cache Injection**
精確檢索局部語義，並透過門控機制 (Cache Gate) 動態融合來自歷史 Chunk 的 Super Tokens。
* **SSM Global Scan**
使用平行時間掃描建立當前 Chunk 內的全域背景記憶。
* **Latent Neuro-Optimizer + Cross-Chunk Memory**
在潛空間中反覆修正狀態，其內部的交叉注意力 (Cross Attention) 現在能直接看見並檢索歷史的 Super Tokens。

---

## 🔹 Brain-Inspired Hippocampus (仿生海馬迴模組)

延續 V23 的記憶系統：

### Dentate Gyrus (DG)

先將潛特徵高維展開，形成稀疏記憶表徵，強化模式分離 (Pattern Separation)，降低特徵混疊。

### CA3 Associative Recall

透過 Top-K 聯想注意力建立類似聯想記憶的動態檢索，強化多步推理中的內部一致性。這讓模型具備「回想」而非僅僅「注意」的能力。

---

## 🔹 FlashAttention-2 Reasoning Core

推理核心使用：

### QK-Norm Cosine Attention

透過 RMSNorm、L2 Normalization 與 Learnable Temperature，建立穩定的餘弦相似度推理空間，解決 Entropy Collapse 與 Attention Saturation 問題。

### FlashAttention-2 Dynamic Fusion

內外部記憶的融合直接使用 PyTorch 原生 `scaled_dot_product_attention`，自動觸發 CUDA Kernel Fusion 與 Tensor Core 加速，在無損因果性的前提下達成極高吞吐效率。

---

## 🔹 Latent Neuro-Optimization (潛空間神經優化)

V24 的核心思想：

> 模型不只是 forward 一次，而是在 latent workspace 中進行多輪「自我修正」，且每次修正都能調用跨時空的歷史記憶。

每一步推理都包含：

1. 外部上下文與歷史快取檢索 (External KV Cross-Attention)
2. 海馬迴聯想記憶
3. 動態路由融合
4. Master Gate 控制更新幅度
5. RMSNorm 穩定化

---

## 🔹 Adaptive Halting System (自適應停機機制)

模型會根據 latent change magnitude、halt probability 與 feature convergence，動態決定是否提前停止推理。這個 Halt Probability 在 V24 中被進一步昇華為**記憶濃縮的權重指標 (Importance Score)**，讓「想得越深」的資訊被記憶得越牢。

---

# 🏗️ Architecture | 模型架構 (V24-InfiniteResonance)

| Layer Index | Module Type | Functional Role |
| --- | --- | --- |
| **0, 2, 4, 6, 8, 10** | **Resonance Attention** | 高精度局部語義檢索、共振門控與 **外部快取融合** |
| **1, 5, 9** | **D2V20 SSM Block** | 當前 Chunk 內的背景記憶與時間序列掃描 |
| **3, 7, 11** | **Neuro-Optimizer Core** ⭐ | 多步遞迴推理、記憶聯想與 **跨 Chunk 交叉檢索** |

---

# ⚙️ Training Setup | 訓練設定

* **Model Dimensions**
512 d_model / 256 latent_dim
* **Attention Heads**
8 heads
* **Micro-Chunking & Cache**
`chunk_size: 64`, `cache_capacity: 512` Super Tokens
* **Thinking Steps**
3 recursive optimization steps
* **Optimizer**
AdamW + BFloat16 Autocast
* **Loss Design**
* Multi-Step Cross Entropy
* Diff Regularization
* Halt BCE Supervision


* **Inference Exit Threshold**
`0.85`

---

# 📊 Design Philosophy | 設計哲學

V24 的核心哲學是：

> **「無限的上下文不需要無限的視窗長度，而是需要具備時間感知的動態記憶濃縮。」**

傳統 Transformer：

* 每次計算成本隨序列長度平方暴增 ($O(N^2)$)
* 歷史記憶無法動態篩選與壓縮

而 V24：

* 透過 Micro-Chunking 將計算複雜度降為 $O(C \times N)$
* 根據思考深度 (Halt Probability) 自主篩選重要記憶形成 Super Tokens
* 利用帶有 RoPE 的 Causal Cache 讓潛空間優化器看見無限的過去

使小模型也能在極低 VRAM 消耗下展現超長文本的推理與記憶能力。

---

# 🚧 Status | 開發狀態

* [x] V20 SSM Global Scan
* [x] V21 QK-Norm Stabilization
* [x] V22 Latent Optimizer
* [x] V23 Brain-Inspired Hippocampus
* [x] **V24 Phase 3: Causal Keyframe Cache & Micro-Chunking**
* [x] FlashAttention-2 Dynamic Fusion
* [x] Adaptive Halting System
* [ ] Phase 4: Persistent Cross-Sample Memory
* [ ] Phase 5: Multimodal Resonance Workspace

---

# 📜 License

MIT License

---

# ⭐ Support the Project

如果您對：

* 潛空間遞迴推理
* 動態記憶濃縮 (Super Tokens)
* 無限長度分塊遞推 (Chunk-wise Recurrence)
* 仿生海馬迴記憶

感興趣，歡迎給專案一個 ⭐！
