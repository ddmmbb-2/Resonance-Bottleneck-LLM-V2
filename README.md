

# 🚀 Resonance-Bottleneck-LLM (V22-Optimizer)

> *Beyond attention: The model is no longer just a predictor, but an iterative optimizer of its own latent logic.*

## 🧠 Overview | 概述

**Resonance-Bottleneck-LLM (V22-Optimizer)** 標誌著架構設計的重大轉向。我們不再僅僅依賴增加層數來獲取智能，而是將模型視為一個**遞迴優化系統**。在 V22 中，我們正式引入了「三明治夾心架構 (Sandwich Architecture)」，融合了高精度注意力、SSM 全域背景掃描，以及最核心的 **Latent Optimizer Core**。

透過 V22 的**信用分配機制 (Credit Assignment)** 與 **QK-Norm 交叉注意力**，模型現在不僅「知道何時停止」，更學會了「如何高效地修正自己的錯誤」。

---

## ✨ Key Features | 核心特色

### 🔹 Hybrid Sandwich Architecture (三明治夾心架構)

V22 徹底重構了層級堆疊策略，將三種不同性質的運算模組交織在一起：

* **Precision Retrieval (Attention):** 負責精確的短程特徵檢索。
* **Global Context (SSM):** 透過平行掃描 (Parallel Scan) 捕捉全域背景記憶，取代了過時的局部卷積。
* **Logic Optimization (Optimizer Core):** 作為動態工作區，對潛空間特徵進行深度邏輯提煉。

### 🔹 SSM Global Scan (時間序列全域掃描)

引入了基於 `A_log` 離散化參數的 SSM 模組。這讓模型具備了類似 Mamba 的長距離依賴處理能力，同時透過我們自研的穩定 Cumsum 技巧，在不依賴自定義 CUDA Kernel 的情況下實現了高效的全域掃描。

### 🔹 Latent Optimizer Core (潛空間優化核心)

推理核心現在運作起來更像是一個最小化的 Adam 優化器：

* **QK-Norm Cross Attention:** 引入餘弦相似度注意力與**可學習溫度參數**，徹底杜絕熵崩潰 (Entropy Collapse)。
* **Gradient Contamination Fix:** 透過梯度縮放（僅保留 10% 梯度傳回 KV），防止深層思考過度干擾基礎表徵空間。
* **Direct Latent Normalization:** 在每一步思考後直接進行 RMSNorm，防止潛特徵在多次迭代中產生漂移。

### 🔹 Step-wise Credit Assignment (階梯式信用分配機制)

我們捨棄了固定的懲罰，改用動態的表現評估：

* **Monotonicity Constraint (單調遞減約束):** 強迫模型下一步的表現必須優於上一步。
* **Margin-based Penalty:** 如果第 N 步思考導致 Loss 上升，模型會受到嚴厲懲罰，迫使優化器學會撤回無效的推理軌跡。
* **Incremental Weighting:** 給予後續思考步驟更高的權重 (0.2 -> 0.3 -> 0.5)，確保深度推理產出的 Logits 具有最高品質。

---

## 🏗️ Architecture | 模型架構 (V22-Optimizer Variant)

V22 採用非對稱堆疊，邏輯分佈如下：

| Layer Index | Module Type | Functional Role |
| --- | --- | --- |
| **0, 2, 4, 6, 8, 10** | **Resonance Attention** | 高精度特徵檢索與共振門控 |
| **1, 5, 9** | **D2V20 SSM Block** | 全域時間序列背景掃描 (Global Context) |
| **3, 7, 11** | **Optimizer Core** ⭐ | **遞迴優化工作區 (Think Steps: 3)** |

---

## ⚙️ Training Setup | 訓練設定 (V22)

* **Model Dimensions**: 512 d_model / 256 latent_dim.
* **Thinking Steps**: 3 (於第 3, 7, 11 層觸發)。
* **Optimization**: AdamW + Autocast (BFloat16)。
* **Loss Mechanics**: 結合了 Step-wise CE, Margin Loss, 以及 Halt Binary Cross Entropy。
* **Inference Exit**: `threshold = 0.85` (信心達標即停)。

---

## 📊 Design Motivation | 設計動機

V22-Optimizer 的核心哲學是 **「特徵的演化而非堆疊」**：

> **傳統 LLM 像是一條直線流水線，每一層只能處理一次資訊；V22 則像是一個實驗室，在 Optimizer Core 中，模型被允許針對當前語境進行三次「自我修正」。透過信用分配機制，我們確保了每一次的修正都是有意義的，讓 12 層的模型展現出遠超其參數規模的邏輯深度。**

---

## 🚧 Status | 開發狀態

* [x] **V20 SSM**: 全域掃描取代局部卷積。
* [x] **V21 QK-Norm**: 解決交叉注意力的穩定性問題。
* [x] **V22 Optimizer Core**: 實裝信用分配與梯度隔離。
* [x] **Adaptive Exit**: 基於特徵變化量的動態停機邏輯。
* [ ] Phase 4: 多模態共振與跨樣本記憶槽。

---

## 📜 License

MIT License

---

## ⭐ Support the Project

如果您對「潛空間優化器」與「三明治 SSM 架構」感興趣，請給我們一個 ⭐！
