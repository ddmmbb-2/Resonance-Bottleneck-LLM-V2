# 🚀 Resonance-Bottleneck-LLM (V24.5-InfiniteResonance)

> *Not just predicting tokens — but recursively refining latent cognition, aligning thought trajectories, and dynamically condensing infinite context through resonance.*

[![GitHub](https://img.shields.io/badge/GitHub-V24.5-blue?logo=github)](https://github.com/ddmmbb-2/Resonance-Bottleneck-LLM-V2/tree/main/V24.5)
[![Data & Checkpoints](https://img.shields.io/badge/Drive-Model_&_Corpus-orange?logo=googledrive)](https://drive.google.com/drive/folders/1wKT4AunavJBZCtAAqhwT0w3oLrizms2D?usp=drive_link)

---

## 🧠 概述 | Overview

**Resonance-Bottleneck-LLM (V24.5-InfiniteResonance)** 代表整體架構從「單次長序列建模器」正式進化為一種**具備無限記憶潛力的分塊遞推系統 (Chunk-wise Recurrent Reasoning System)**。

V24 奠定了 Causal Keyframe Cache (因果關鍵影格緩存) 的基礎，而 **V24.5 則是在「訓練穩定度」與「計算效能」上進行了深度的極致優化**。模型不再受限於傳統的注意力視窗長度，而是將過去的記憶動態濃縮並無縫注入到當前的推理步驟中，同時在純潛在空間 (Pure Latent Space) 中完成所有複雜邏輯推演。

此次 V24.5 版本的核心進化包含：

* 🧩 **Micro-Chunking 遞推與彈性吞吐 (解鎖 CPU 瓶頸)**
* 🌌 **Super Token 動態濃縮與重要性過濾**
* ⚡ **Samba-Style 混合層疊架構 (1 Attn : 2 SSM : 1 Opt)**
* 🎯 **Latent Alignment Loss (潛在對齊目標與純潛在優化)**
* 🛡️ **Dynamic Diff-based LR Scaling (自適應動態防禦機制)**

模型現在具備了「工作記憶 + 長期聯想 + 內部修正 + **跨時空外部記憶**」的全方位認知特性，且能在有限的消費級顯卡 (如 RTX 3060 12GB) 上展現驚人的運算能效。

---

## 📖 理論基礎 | Theoretical Foundation

### 1. 從固定視窗到無限記憶：潛在空間共振瓶頸

傳統 Transformer 的自注意力計算複雜度為 $O(N^2)$，長序列生成時記憶體與計算量迅速膨脹。V24.5 引入**共振瓶頸 (Resonance Bottleneck)** 理論：

- 將輸入序列分割為多個 `chunk`（$C$ 個片段，每段長度 $M$），在每個 chunk 內進行局部注意力與 SSM 掃描。
- 跨 chunk 的資訊流透過**因果關鍵影格緩存 (Causal Keyframe Cache)** 實現：每個 chunk 推理結束後，根據思考重要性動態生成 **Super Token**，並以 RoPE 位置編碼記錄其時間戳。後續 chunk 可無縫檢索這些濃縮記憶，實現理論上**無限長上下文**的遞推推理，總計算複雜度降至 $O(C \cdot M^2)$，當 $M$ 固定時為線性。

### 2. 純潛在空間神經優化器 (Latent Neuro-Optimizer)

傳統 Chain-of-Thought 需要消耗大量文字 token 來進行邏輯推演。V24.5 將推理過程**壓縮至 256 維的純潛在空間**：

- 在每個 `ResonanceOptimizerCore` 層，輸入表徵 $x \in \mathbb{R}^{B \times L \times d_{model}}$ 首先被線性投影至潛在狀態 $h_{latent} \in \mathbb{R}^{B \times L \times d_{latent}}$。
- 在 `think_steps` 步內，模型反覆執行：
  \[
  h_{t+1} = \text{Norm}\left(h_t + \text{CrossAttn}(h_t, K_{ctx}, V_{ctx}) + \text{Hippocampus}(h_t) \right)
  \]
  其中 $\text{CrossAttn}$ 允許與當前 chunk 的上下文表徵及外部 Super Token 快取進行交互；$\text{Hippocampus}$ 模擬海馬迴聯想記憶，進行內部一致性修正。
- 僅在最後一步，將收斂的 $h_{final}$ 線性投影回 $d_{model}$ 空間，**大幅節省了深層算圖所需的 VRAM 與計算量**，同時讓推理過程本身脫離 tokenizer 的離散瓶頸。

### 3. 動態對齊損失 (Latent Alignment Loss)

為保證多步推理軌跡的收斂性，V24.5 引入潛在狀態對齊：
\[
\mathcal{L}_{align} = \sum_{i=1}^{T-1} \frac{i}{T-1} \cdot \text{MSE}(h_i, h_T^{\text{detach}})
\]
該損失強制模型前 $T-1$ 步的潛在表達逐步向最終步靠攏，有效抑制推理發散，並使中間步驟的潛在表徵也具備語義一致性。

### 4. 仿生海馬迴記憶 (Brain-Inspired Hippocampus)

受到大腦 DG-CA3 迴路啟發：

- **DG 高維稀疏化**：將潛在向量膨脹至 $4\times d_{latent}$，模擬齒狀回對輸入模式的分離。
- **CA3 自聯想檢索**：使用 Top-K 稀疏注意力，僅保留最強的若干聯想權重，實現類似聯想記憶的動態檢索，強化多步推理中的長期依賴。

---

## 🧪 推論機制 | Inference Mechanism

推理時，模型以**分塊迴圈 (Chunk-wise Recurrence)** 生成文本，流程如下：

1. **初始提示處理**  
   將輸入提示（或先前生成的序列）分割為 `chunk_size` 長度的片段。首個 chunk 直接進行嵌入並通過所有層。

2. **外部記憶檢索**  
   對於第 $k$ 個 chunk（$k>0$），從 `CausalKeyframeCache` 中取出所有 `end_pos` ≤ 當前起始位置 $S_k$ 的 Super Tokens，作為外部鍵值對 $K_{cache}, V_{cache}$ 輸入注意力層與推理核心。

3. **混合層疊處理**  
   - **共振注意力層 (Layer 0, 4, 8)**：執行局部注意力，並透過門控融合外部 Super Token 的長程記憶。  
   - **SSM 時間掃描層 (Layer 1,2,5,6,9,10)**：對當前 chunk 建立雙重線性時間記憶，捕捉局部與全域順序特徵。  
   - **神經優化器推理層 (Layer 3,7,11)**：在純潛在空間中執行 3～6 步遞迴推理，動態融合外部記憶與海馬迴聯想，直至思考收斂或達到退出閾值 (`inference_exit_threshold`)。

4. **Super Token 濃縮與快取更新**  
   推理層的最終 `halt_prob` 被用來計算每個 token 的重要性（$importance = 1 - halt\_prob$）。若當前 chunk 的平均重要性高於閾值（例如 0.3），則透過 `CausalTokenSelector` 將整個 chunk 濃縮為一個 Super Token，連同其 `end_pos` 存入快取；否則捨棄，避免記憶體中堆積冗餘資訊。

5. **輸出投影與取樣**  
   最後一步的輸出表徵通過輸出層投影至詞表維度，進行下一個 token 的取樣。生成的 token 被拼接到序列尾部，當累積長度達到 `chunk_size` 時，觸發下一個 chunk 推理迴圈。

6. **無限長生成**  
   由於每個 chunk 僅需 $O(M^2)$ 計算（$M$ 為 chunk 大小），且外部記憶的檢索成本與記憶數量 $K$ 呈線性關係，模型可以持續生成任意長度的文本，而不會出現傳統 Transformer 的記憶體爆增問題。理論上，只要 `cache_capacity` 足夠，上下文長度可達**數十萬 token**。

---

## ✨ 核心特色 | Key Features

### 🔹 Causal Keyframe Cache & Super Tokens (因果關鍵影格與超級符號)

打破上下文長度限制，引入全新的動態記憶濃縮機制：

* **彈性 Micro-Chunking (空間換取時間)**  
  支援動態調整區塊大小 (如 `chunk_size=192/384`)，大幅減少 CPU 發布 Kernel 指令的開銷，讓 GPU 算力發揮至 100%。
* **Dynamic Super Token Condensation**  
  根據 Neuro-Optimizer 的「思考深度 (Halt Probability)」，動態計算 Chunk 內每個 Token 的重要性。只有當整體重要性超過閾值時，才會將該 Chunk 融合成一個高密度的**Super Token**。
* **Strict Causality & RoPE Alignment**  
  外部記憶嚴格記錄 `end_pos` 並結合 RoPE (旋轉位置編碼)，確保模型在檢索歷史 Super Tokens 時擁有完美的絕對位置感知與因果隔離，絕不洩漏未來資訊。

### 🔹 Samba-Style Hybrid Architecture (Samba 式混合層疊架構)

V24.5 放棄了對稱式設計，改採更具硬體友善度與語義捕捉力的特定層級排列 (每 4 層為一個大週期)：

* **Resonance Attention (檢索層)**：精確檢索局部語義，並透過門控動態融合歷史 Super Tokens。
* **雙重 SSM Global Scan (背景層)**：連續兩層平行時間掃描，建立強大且線性的當前 Chunk 全域背景記憶。
* **Neuro-Optimizer (推理層)**：在潛空間中反覆修正狀態，整合並收斂前面各層的資訊。

### 🔹 Brain-Inspired Hippocampus (仿生海馬迴模組)

延續並穩定了 V23 的記憶系統：

* **Dentate Gyrus (DG) 高維稀疏化**：先將潛特徵高維展開，形成稀疏記憶表徵，降低特徵混疊。
* **CA3 Associative Recall**：透過 Top-K 聯想注意力建立類似聯想記憶的動態檢索，強化多步推理中的內部一致性。

### 🔹 Latent Neuro-Optimization & Alignment (純潛空間優化與對齊)

V24.5 最大的效能與收斂突破：

* **Pure Latent Workspace (純潛在空間運算)**：模型在 `think_steps` 中**只在 256 維的潛在空間中進行迭代**，只有最後一步才會投射回主模型維度。這極大地節省了 VRAM 與矩陣乘法的開銷。
* **Latent Alignment Loss (潛在對齊損失)**：訓練時引入權重遞增的對齊目標，強迫模型前 $N-1$ 步的思考軌跡逐漸對齊最後一步的潛在表達，確保思考過程不會發散。

### 🔹 Dynamic Defenses (自適應防禦與停機系統)

* **Diff-based LR Scaling (動態學習率防護網)**：內建即時監控神經網路位移 (Diff) 的機制。當模型遇到複雜邏輯導致位移暴衝時，會自動縮放 `lr_scale` 保護梯度；當思考穩定時，則自動恢復滿血學習率。
* **Adaptive Halting System**：讓「想得越深」的資訊被記憶得越牢，賦予模型「知道何時該停止思考」的能力。

---

## 🏗️ 架構 | Architecture (V24.5-InfiniteResonance)

| Layer Index | Module Type | Functional Role |
| :--- | :--- | :--- |
| **0, 4, 8** | **Resonance Attention** | 高精度局部語義檢索、共振門控與**外部快取融合** |
| **1, 2, 5, 6, 9, 10** | **D2V20 SSM Block** | 當前 Chunk 內的背景記憶與時間序列平行掃描 |
| **3, 7, 11** | **Neuro-Optimizer Core** ⭐ | 多步遞迴推理、海馬迴聯想與**純潛在空間對齊** |

---

## ⚙️ 訓練設定 | Training Setup

* **模型維度**  
  `d_model: 512`, `latent_dim: 256`
* **注意力頭數**  
  `8 heads`
* **Micro-Chunking 與快取**  
  `chunk_size: 64~384 (可調)`, `cache_capacity: 512` Super Tokens
* **思考步數**  
  3～6 步遞迴潛在優化 (可依硬體配置)
* **損失設計**  
  * 多步交叉熵 (CE)  
  * **潛在對齊損失 (Latent Alignment Loss, 動態權重 MSE)** 🚀 *[V24.5 新增]*  
  * 位移正則 (Diff Regularization)  
  * 停止機率二元交叉熵監督 (Halt BCE)
* **優化器與穩定性**  
  AdamW + BFloat16 自動混合精度 + **基於最大位移的自動學習率縮放** 🚀 *[V24.5 新增]*
* **資料與預訓練模型**  
  * 預訓練語料庫 (`corpus_v20_twllm.bin`) 與檢查點 (`d2_v24_samba_latent.pth`)  
    請至 [Google Drive](https://drive.google.com/drive/folders/1wKT4AunavJBZCtAAqhwT0w3oLrizms2D?usp=drive_link) 下載並置於專案根目錄。

---

## 📊 設計哲學 | Design Philosophy

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

## 🚧 開發狀態 | Development Status

- [x] V20 SSM Global Scan
- [x] V21 QK-Norm Stabilization
- [x] V22 Latent Optimizer
- [x] V23 Brain-Inspired Hippocampus
- [x] V24 Causal Keyframe Cache & Micro-Chunking
- [x] **V24.5 Samba-Style Hybrid Layer Reshaping**
- [x] **V24.5 Pure Latent Projection & Alignment Loss**
- [x] **V24.5 Dynamic Diff-based LR Defense System**
- [ ] Phase 4: Persistent Cross-Sample Memory
- [ ] Phase 5: Multimodal Resonance Workspace

---

## 📜 授權 | License

MIT License

---

## ⭐ 支持 | Support the Project

如果您對：

* 潛空間遞迴推理 (Latent Recursive Reasoning)
* 動態記憶濃縮 (Super Tokens)
* 防禦型訓練優化 (Dynamic LR Defense)
* 仿生海馬迴記憶

感興趣，歡迎給專案一個 ⭐！

完整源碼請見 [GitHub V24.5](https://github.com/ddmmbb-2/Resonance-Bottleneck-LLM-V2/tree/main/V24.5)。  
模型權重及語料庫請從 [Google Drive](https://drive.google.com/drive/folders/1wKT4AunavJBZCtAAqhwT0w3oLrizms2D?usp=drive_link) 下載。
