---

# 🚀 Resonance-Bottleneck-LLM (V19-Mini)

> *Beyond attention: Latent resonance with recursive thinking.*

---

## 🧠 Overview | 概述

**Resonance-Bottleneck-LLM (V19)** 演進至 **"Mini-Reasoning"** 階段。這是一個實驗性的 Transformer 架構，除了延續 **潛空間壓縮（Latent Bottleneck）** 與 **共振式注意力（Resonance Attention）** 外，V19 引入了 **Selective Recurrent Reasoning (局部遞迴推理)** 機制，允許模型在特定層級進行「多次思考（Think Steps）」，以深化語意理解。

---

## ✨ Key Features | 核心特色

### 🔹 Selective Recurrent Reasoning (New!)
In V19, specific layers (Block 2 & 4) are upgraded to **Reasoning Cores**. These layers use a recurrent loop (`think_steps=2`) to iteratively refine latent representations before passing them to the next block.
在 V19 中，特定的層級（第 2 與第 4 層）被升級為 **推理核心（Reasoning Core）**。透過遞迴循環機制，模型能在這些層級進行多次迭代思考，強化複雜邏輯的處理能力。

### 🔹 Latent Bottleneck & Resonance Attention
Compresses information into a 128-dim latent space, utilizing phase-aware resonance gating (amplitude, phase, and interference) instead of standard softmax.
將資訊壓縮至 128 維的潛空間，並利用相位干涉與振幅門控進行共振式注意力計算，完全捨棄傳統的 softmax。

### 🔹 Triple-Hybrid Architecture
A balanced mix of **Attention**, **Causal Convolution**, and **Reasoning Cores**.
結合 **注意力層**、**因果卷積層** 與 **推理核心** 的三重複合架構，兼顧全域關聯、局部結構與深度邏輯。

---

## 🏗️ Architecture | 模型架構 (V19-Mini)

V19-Mini 採用 6 層交替結構，配置如下：

| Layer | Type | Description |
| :--- | :--- | :--- |
| **Layer 0** | Attention Block | V18.1 Resonance Attention |
| **Layer 1** | Conv Block | Causal 1D Convolution |
| **Layer 2** | **Reasoning Core** ⭐ | **Recurrent Thinking (2 steps)** |
| **Layer 3** | Conv Block | Causal 1D Convolution |
| **Layer 4** | **Reasoning Core** ⭐ | **Recurrent Thinking (2 steps)** |
| **Layer 5** | Conv Block | Causal 1D Convolution |

👉 *Detailed diagram: See the generated `V19-Mini_Architecture.png`*

![png](Gemini_Generated_Image_cjbpm7cjbpm7cjbp.png)

---

## ⚙️ Training Setup | 訓練設定 (V19-Mini)

* **Model Size**: 256 dim / 6 layers
* **Attention Heads**: 4
* **Latent Dim**: 128
* **Thinking Steps**: 2 (at Layer 2 & 4)
* **Context Length**: 512 (RoPE supported)
* **Vocab Size**: 16,384
* **Optimization**: AdamW + Warmup Scheduler + Autocast (BFloat16)

---

## 📊 Design Motivation | 設計動機

The V19 update explores **"Thinking in Latent Space"**:

> **Standard LLMs process tokens linearly. Resonance V19 processes tokens through resonance, then pauses to "think" via recurrent latent updates.**

V19 的設計核心在於「潛空間中的思考」：傳統 LLM 是線性地處理 Token，而 Resonance V19 則是透過共振捕捉關聯後，在特定的推理層停留並進行遞迴運算，實現更深層的特徵演化。

---

## 🚧 Status | 開發狀態

* [x] **V19 Core Implementation**: Reasoning Core & Latent Modulation.
* [x] **Mini-Experiment (3060 Friendly)**: Optimized for 12GB VRAM.
* [x] **Gate Monitoring**: Real-time tracking of reasoning core activity.
* [ ] Scaling to V19-Large (7B+ with more reasoning blocks).
* [ ] Benchmarking on logical reasoning tasks.

---

## 📜 License

MIT License

---

## ⭐ Support the Project

If you find this "Resonance + Reasoning" approach interesting, please give us a ⭐!
