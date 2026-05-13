

# 🚀 Resonance-Bottleneck-LLM (V20-Adaptive)

> *Beyond attention: Latent resonance, recursive thinking, and self-aware adaptive depth.*

## 🧠 Overview | 概述

**Resonance-Bottleneck-LLM (V20-Adaptive)** 正式邁入 **「自適應深度 (Adaptive Depth)」** 與 **「效能獎勵掛載」** 的全新階段（修復 Early Collapse 版本）。在繼承潛空間壓縮與共振式注意力的基礎上，V20-Adaptive 將開發重心轉向了賦予模型「知道何時該停止思考」的能力。

透過引入精密的**效能獎勵機制 (Performance Reward Mechanism)** 與**損失驅動的課程學習 (Loss-Driven Curriculum)**，模型現在能夠在局部遞迴推理中自我評估思考品質，動態決定是否提早退出 (Early Exit)，從而在推理效能與運算成本之間取得完美的平衡。

---

## ✨ Key Features | 核心特色

### 🔹 Adaptive Depth & Early Exit (Phase 2!)

Introduces a shadow gating mechanism that allows the Reasoning Cores to dynamically halt their recursive thinking.
引入影子門控機制，讓推理核心能夠動態中斷遞迴思考。

* **Target Halt:** 動態計算退出機率，當潛在特徵變化量偏離健康區間時強制觸發退出。
* **Inference Exit Threshold:** 推理時的信心閾值（設為 `0.85`），達到即提前結束思考步驟，大幅節省算力。

### 🔹 Latent Thinking Quality Control (效能獎勵機制)

Replaced static loops with a dimension-normalized quality assessment of the latent delta.
針對模型在潛空間中的「思考軌跡」進行嚴格的品質控管：

* **Lazy Penalty (懶惰懲罰):** 嚴厲懲罰微小且無效的特徵變化（低於 0.1）。
* **Chaos Penalty (混亂懲罰):** 壓制過度劇烈的特徵震盪（高於 1.5），確保穩定收斂。
* **Effective Thinking Reward (有效思考獎勵):** 鼓勵模型將特徵變化量維持在「黃金區間」（約 0.7 左右），最大化推理效益。
* **Entropy Regularization (熵損失):** 防止預測值極化，保持門控的靈活性與彈性。

### 🔹 Loss-Driven Curriculum (損失驅動的課程學習)

A smart curriculum that smoothly activates the halting mechanism based on the model's fundamental understanding.
利用平滑交叉熵（Smoothed CE Loss）作為避震器，動態調整門控權重 (`halt_weight`)：

* 當平滑 Loss 降至 `4.5` 時，門控機制開始微微甦醒。
* 當平滑 Loss 降至 `3.0` 時，門控火力全開進入完全體。
* 完美避免模型在尚未學會基礎語意前，就因提早退出機制而導致的早期崩潰 (Early Collapse)。

### 🔹 Latent Bottleneck & Selective Recurrent Reasoning

Compresses information into a latent space using phase-aware resonance gating. Specific layers (Reasoning Cores) use a recurrent loop (`think_steps=2`) to iteratively refine latent representations before passing them forward.
將資訊壓縮至潛空間並進行共振式計算。特定層級被指定為推理核心，透過遞迴循環機制進行多次迭代思考。

---

## 🏗️ Architecture | 模型架構 (V20-Adaptive Variant)

V20 採用交替結構，將自適應推理層平均分佈，配置如下：

| Layer | Type | Description |
| --- | --- | --- |
| **Layer 0, 1, 2** | D2V18 Attention / Conv | V18.1 Resonance Attention & Causal 1D Conv |
| **Layer 3** | **Reasoning Core V20** ⭐ | **Recurrent Thinking + Adaptive Early Exit** |
| **Layer 4, 5, 6** | D2V18 Attention / Conv | Standard blocks |
| **Layer 7** | **Reasoning Core V20** ⭐ | **Recurrent Thinking + Adaptive Early Exit** |
| **...** | ... | ... |
| **Layer 11** | **Reasoning Core V20** ⭐ | **Recurrent Thinking + Adaptive Early Exit** |

---

## ⚙️ Training Setup | 訓練設定 (V20-Adaptive)

* **Model Size**: 512 dim / 12 layers
* **Attention Heads**: 8
* **Latent Dim**: 256
* **Thinking Steps**: 2 (at Layers 3, 7, 11)
* **Adaptive Gating**: `halt_tau = 0.05`, `halt_weight = 0.5`
* **Optimization**: AdamW + Cosine Decay Warmup + Autocast (BFloat16)
* **Stability**: Dual-Checkpointing & Smoothed CE Tracking (Loss 避震器)

---

## 📊 Design Motivation | 設計動機

The V20-Adaptive update explores **"Self-Awareness in Latent Thinking"**:

> **If V19 proved that local recursive thinking works, V20-Adaptive teaches the model *when to stop*. By observing the dimension-normalized trajectory of its own thoughts, the model is rewarded for productive logic and penalized for lazy or chaotic updates, all guarded by a smooth loss-driven curriculum.**

V20-Adaptive 的設計核心在於「潛空間思考的自我認知」：讓模型不僅會思考，還知道何時該見好就收。透過觀測標準化後的潛特徵變化軌跡，模型在產出有效邏輯時獲得獎勵，在偷懶或混亂時受到懲罰，並由平滑的損失函數逐步驅動其學習進程，徹底解決了動態深度模型難以訓練的痛點。

---

## 🚧 Status | 開發狀態

* [x] **V19 Core**: Reasoning Core & Latent Modulation.
* [x] **Phase 2 (V20-Adaptive)**: Shadow Adaptive Depth & Early Exit routing.
* [x] **Quality Control**: Reward/Penalty mechanics for latent delta trajectory.
* [x] **Curriculum Learning**: Smoothed CE-driven dynamic halt weighting.
* [ ] Phase 3: Span Compression & Phase Recall.

---

## 📜 License

MIT License

---

## ⭐ Support the Project

If you find this "Resonance + Adaptive Reasoning" approach interesting, please give us a ⭐!
