# 🚀 Resonance-Bottleneck-LLM (V24.5-InfiniteResonance)

> *Not just predicting tokens — but recursively refining latent cognition, aligning thought trajectories, and dynamically condensing infinite context through resonance.*

## 🧠 Overview | 概述

**Resonance-Bottleneck-LLM (V24.5-InfiniteResonance)** 代表整體架構從「單次長序列建模器」正式進化為一種 **具備無限記憶潛力的分塊遞推系統 (Chunk-wise Recurrent Reasoning System)**。

V24 奠定了 Causal Keyframe Cache (因果關鍵影格緩存) 的基礎，而 **V24.5 則是在「訓練穩定度」與「計算效能」上進行了深度的極致優化**。模型不再受限於傳統的注意力視窗長度，而是將過去的記憶動態濃縮並無縫注入到當前的推理步驟中，同時在純潛在空間 (Pure Latent Space) 中完成所有複雜邏輯推演。

此次 V24.5 版本的核心進化包含：

* 🧩 **Micro-Chunking 遞推與彈性吞吐 (解鎖 CPU 瓶頸)**
* 🌌 **Super Token 動態濃縮與重要性過濾**
* ⚡ **Samba-Style 混合層疊架構 (1 Attn : 2 SSM : 1 Opt)**
* 🎯 **Latent Alignment Loss (潛在對齊目標與純潛在優化)**
* 🛡️ **Dynamic Diff-based LR Scaling (自適應動態防禦機制)**

模型現在具備了「工作記憶 + 長期聯想 + 內部修正 + **跨時空外部記憶**」的全方位認知特性，且能在有限的消費級顯卡 (如 RTX 3060 12GB) 上展現驚人的運算能效。

---

# ✨ Key Features | 核心特色

## 🔹 Causal Keyframe Cache & Super Tokens (因果關鍵影格與超級符號)

打破上下文長度限制，引入全新的動態記憶濃縮機制：

* **彈性 Micro-Chunking (空間換取時間)**
支援動態調整區塊大小 (如 `chunk_size=192/384`)，大幅減少 CPU 發布 Kernel 指令的開銷，讓 GPU 算力發揮至 100%。
* **Dynamic Super Token Condensation**
根據 Neuro-Optimizer 的「思考深度 (Halt Probability)」，動態計算 Chunk 內每個 Token 的重要性。只有當整體重要性超過閾值時，才會將該 Chunk 融合成一個高密度的 **Super Token**。
* **Strict Causality & RoPE Alignment**
外部記憶嚴格記錄 `end_pos` 並結合 RoPE (旋轉位置編碼)，確保模型在檢索歷史 Super Tokens 時擁有完美的絕對位置感知與因果隔離，絕不洩漏未來資訊。

## 🔹 Samba-Style Hybrid Architecture (Samba 式混合層疊架構)

V24.5 放棄了對稱式設計，改採更具硬體友善度與語義捕捉力的特定層級排列 (每 4 層為一個大週期)：

* **Resonance Attention (檢索層)**：精確檢索局部語義，並透過門控動態融合歷史 Super Tokens。
* **雙重 SSM Global Scan (背景層)**：連續兩層平行時間掃描，建立強大且線性的當前 Chunk 全域背景記憶。
* **Neuro-Optimizer (推理層)**：在潛空間中反覆修正狀態，整合並收斂前面各層的資訊。

## 🔹 Brain-Inspired Hippocampus (仿生海馬迴模組)

延續並穩定了 V23 的記憶系統：

* **Dentate Gyrus (DG) 高維稀疏化**：先將潛特徵高維展開，形成稀疏記憶表徵，降低特徵混疊。
* **CA3 Associative Recall**：透過 Top-K 聯想注意力建立類似聯想記憶的動態檢索，強化多步推理中的內部一致性。

## 🔹 Latent Neuro-Optimization & Alignment (純潛空間優化與對齊)

V24.5 最大的效能與收斂突破：

* **Pure Latent Workspace (純潛在空間運算)**：模型在 `think_steps` 中**只在 256 維的潛在空間中進行迭代**，只有最後一步才會投射回主模型維度。這極大地節省了 VRAM 與矩陣乘法的開銷。
* **Latent Alignment Loss (潛在對齊損失)**：訓練時引入權重遞增的對齊目標，強迫模型前 $N-1$ 步的思考軌跡逐漸對齊最後一步的潛在表達，確保思考過程不會發散。

## 🔹 Dynamic Defenses (自適應防禦與停機系統)

* **Diff-based LR Scaling (動態學習率防護網)**：內建即時監控神經網路位移 (Diff) 的機制。當模型遇到複雜邏輯導致位移暴衝時，會自動縮放 `lr_scale` 保護梯度；當思考穩定時，則自動恢復滿血學習率。
* **Adaptive Halting System**：讓「想得越深」的資訊被記憶得越牢，賦予模型「知道何時該停止思考」的能力。

---

# 🏗️ Architecture | 模型架構 (V24.5-InfiniteResonance)

| Layer Index | Module Type | Functional Role |
| --- | --- | --- |
| **0, 4, 8** | **Resonance Attention** | 高精度局部語義檢索、共振門控與 **外部快取融合** |
| **1, 2, 5, 6, 9, 10** | **D2V20 SSM Block** | 當前 Chunk 內的背景記憶與時間序列平行掃描 |
| **3, 7, 11** | **Neuro-Optimizer Core** ⭐ | 多步遞迴推理、海馬迴聯想與 **純潛在空間對齊** |

---

# ⚙️ Training Setup | 訓練設定

* **Model Dimensions**
512 d_model / 256 latent_dim
* **Attention Heads**
8 heads
* **Micro-Chunking & Cache**
`chunk_size: 64~384 (Adaptive)`, `cache_capacity: 512` Super Tokens
* **Thinking Steps**
3 to 6 recursive latent optimization steps (可依硬體彈性配置)
* **Loss Design**
* Multi-Step Cross Entropy
* **Latent Alignment Loss (MSE with dynamic weighting)** 🚀 *[V24.5 New]*
* Diff Regularization
* Halt BCE Supervision


* **Optimizer & Stability**
AdamW + BFloat16 Autocast + **Auto-LR Scaling via Max Diff** 🚀 *[V24.5 New]*

---

# 📊 Design Philosophy | 設計哲學

V24.5 的核心哲學是：

> **「無限的上下文不需要無限的視窗長度；深度的思考不需要昂貴的全域算圖展開。」**

傳統 Transformer：

* 每次計算成本隨序列長度平方暴增 ($O(N^2)$)
* 多步推理（如 CoT）需要消耗大量實體 Token 長度

而 V24.5：

* 透過 Micro-Chunking 將記憶複雜度降為線性 ($O(C \times N)$)
* 透過 **Latent Optimizer** 將推理過程隱含於潛在空間中，並透過 **Latent Alignment** 確保思考的一致性。
* 只有最終結果才會投射回輸出層，使小模型也能在極低硬體負載下，展現超越其參數規模的邏輯推演與跨時空聯想力。

---

# 🚧 Status | 開發狀態

* [x] V20 SSM Global Scan
* [x] V21 QK-Norm Stabilization
* [x] V22 Latent Optimizer
* [x] V23 Brain-Inspired Hippocampus
* [x] V24 Causal Keyframe Cache & Micro-Chunking
* [x] **V24.5 Samba-Style Hybrid Layer Reshaping**
* [x] **V24.5 Pure Latent Projection & Alignment Loss**
* [x] **V24.5 Dynamic Diff-based LR Defense System**
* [ ] Phase 4: Persistent Cross-Sample Memory
* [ ] Phase 5: Multimodal Resonance Workspace

---

# 📜 License

MIT License

---

# ⭐ Support the Project

如果您對：

* 潛空間遞迴推理 (Latent Recursive Reasoning)
* 動態記憶濃縮 (Super Tokens)
* 防禦型訓練優化 (Dynamic LR Defense)
* 仿生海馬迴記憶

感興趣，歡迎給專案一個 ⭐！
