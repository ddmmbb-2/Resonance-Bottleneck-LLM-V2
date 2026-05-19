# 🚀 Resonance-Bottleneck-LLM (V23-NeuroOptimizer)

> *Not just predicting tokens — but recursively refining latent cognition through resonance, memory, and self-optimization.*

## 🧠 Overview | 概述

**Resonance-Bottleneck-LLM (V23-NeuroOptimizer)** 代表整體架構正式從「序列建模器」進化為一種 **神經動力學推理系統 (Neural Dynamical Reasoning System)**。

V23 不再把 Transformer 視為單純的深層堆疊，而是將模型拆解為三種互補的認知功能：

* **Resonance Attention**：高精度局部特徵共振。
* **SSM Global Memory**：長距離時間序列背景掃描。
* **Latent Neuro-Optimizer**：在潛空間中進行遞迴推理與自我修正。

此次版本最大的突破，在於正式引入：

* 🧠 **仿生海馬迴記憶模組 (DG + CA3)**
* ⚡ **FlashAttention-2 加速交叉推理**
* 🔄 **多步潛空間自我優化**
* 🚦 **自適應停止推理機制**

模型開始具備類似「工作記憶 + 長期聯想 + 內部修正」的混合推理特性，而不再只是單向傳遞資訊。

---

# ✨ Key Features | 核心特色

## 🔹 Neuro-Sandwich Architecture (神經三明治架構)

V23 採用非對稱混合堆疊架構，將不同認知功能交錯排列：

* **Resonance Attention**
  精確檢索局部語義與短程依賴。

* **SSM Global Scan**
  使用平行時間掃描建立全域背景記憶。

* **Latent Neuro-Optimizer**
  在潛空間中反覆修正與強化語意狀態。

這種設計使模型同時具備：

* 局部精度
* 全域記憶
* 深度推理能力

而不需要極端增加層數。

---

## 🔹 Brain-Inspired Hippocampus (仿生海馬迴模組)

V23 新增了靈感來自生物海馬迴的記憶系統：

### Dentate Gyrus (DG)

先將潛特徵高維展開，形成稀疏記憶表徵：

* 強化模式分離 (Pattern Separation)
* 降低特徵混疊
* 提升長程推理穩定性

### CA3 Associative Recall

透過 Top-K 聯想注意力：

* 建立類似聯想記憶的動態檢索
* 模擬內容尋址記憶 (Content Addressable Memory)
* 強化多步推理中的內部一致性

這讓模型開始具備「回想」而非僅僅「注意」的能力。

---

## 🔹 FlashAttention-2 Reasoning Core

V23 的推理核心改用：

### QK-Norm Cosine Attention

透過：

* RMSNorm
* L2 Normalization
* Learnable Temperature

建立穩定的餘弦相似度推理空間。

有效解決：

* Entropy Collapse
* Attention Saturation
* 深層推理發散

問題。

### FlashAttention-2 Dynamic Fusion

直接使用 PyTorch 原生：

`scaled_dot_product_attention`

自動觸發：

* FlashAttention-2
* CUDA Kernel Fusion
* Tensor Core 加速

在不依賴自定義 CUDA 的情況下，仍能達成極高吞吐效率。

---

## 🔹 Latent Neuro-Optimization (潛空間神經優化)

V23 的核心思想：

> 模型不只是 forward 一次，而是在 latent workspace 中進行多輪「自我修正」。

每一步推理都包含：

1. 外部上下文檢索
2. 海馬迴聯想記憶
3. 動態路由融合
4. Master Gate 控制更新幅度
5. RMSNorm 穩定化

形成類似：

* iterative refinement
* recurrent reasoning
* latent optimization

的混合神經動力學。

---

## 🔹 Adaptive Halting System (自適應停機機制)

模型不再固定思考步數。

V23 會根據：

* latent change magnitude
* halt probability
* feature convergence

動態決定是否提前停止推理。

這讓模型具備：

* 更高推理效率
* 更低無效計算
* 更穩定的深度思考

特性。

---

## 🔹 Stable Global SSM Scan (全域穩定掃描)

V23 延續並強化 V20 的 SSM 設計：

* 離散化 `A_log` 動態衰減
* 平行 Prefix Scan
* 高速 Cumsum 重排優化
* 完全 GPU Friendly

同時避免：

* kernel explosion
* custom CUDA dependency
* unstable recurrent accumulation

問題。

---

# 🏗️ Architecture | 模型架構 (V23-NeuroOptimizer)

| Layer Index           | Module Type                | Functional Role |
| --------------------- | -------------------------- | --------------- |
| **0, 2, 4, 6, 8, 10** | **Resonance Attention**    | 高精度局部語義檢索與共振門控  |
| **1, 5, 9**           | **D2V20 SSM Block**        | 長距離背景記憶與時間序列掃描  |
| **3, 7, 11**          | **Neuro-Optimizer Core** ⭐ | 潛空間多步遞迴推理與記憶聯想  |

---

# ⚙️ Training Setup | 訓練設定

* **Model Dimensions**
  512 d_model / 256 latent_dim

* **Attention Heads**
  8 heads

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

* **Training Strategy**

  * Gradient Isolation
  * Step-wise Latent Refinement
  * Adaptive Recursive Reasoning

---

# 📊 Design Philosophy | 設計哲學

V23 的核心哲學是：

> **「智能不是來自更深的堆疊，而是來自可反覆修正的潛空間動力學。」**

傳統 Transformer：

* 每層只處理一次資訊
* 推理路徑固定
* 無法真正反思

而 V23：

* 允許 latent state 多次修正
* 透過海馬迴進行聯想回憶
* 透過動態路由重新組織特徵
* 透過 halt gate 自主停止思考

使小模型也能展現超越參數規模的推理深度。

---

# 🚧 Status | 開發狀態

* [x] V20 SSM Global Scan
* [x] V21 QK-Norm Stabilization
* [x] V22 Latent Optimizer
* [x] V23 Brain-Inspired Hippocampus
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
* 神經動力學架構
* 仿生海馬迴記憶
* SSM + Attention 混合模型

感興趣，歡迎給專案一個 ⭐！
