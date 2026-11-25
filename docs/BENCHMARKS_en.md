# Benchmark Configurations & Experimental Protocols

This document details the hardware/software environments, training hyperparameters, architectural alignment protocols, and evaluation metric definitions required to reproduce the experiments in the **MirrorNetv1** paper. *(For specific quantitative results and empirical tables, please refer to the main manuscript.)*

> **Unified Hardware & Software Environment**
> All model training, architectural probing, and ablation studies in this repository are implemented in **PyTorch** and trained/evaluated from scratch on a single **NVIDIA GeForce RTX 5060 Laptop GPU**.

---

## 1. Training Hyperparameters Overview

| Section | Benchmark / Task | Dataset | Backbone | Optimizer | LR Schedule | Batch Size | Epochs | Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Sec. III-C** | Stream-Wise Norm. Probe | CIFAR-10 | 5-Layer CNN (Twin, ReLU) | SGD (Mom. 0.9) | Cosine ($0.1 \to 10^{-4}$) | 128 | 50 | 10 |
| **Sec. III-D** | Complex Object Detection | VisDrone-DET | YOLOv8n / YOLOv8s (Backbone+Neck) | AdamW (`lr0=0.001`, `pretrained=False`) | Ultralytics Linear Decay (`close_mosaic=20`) | 8 | 100 (`patience=30`) | 3 |
| **Sec. IV-B** | Fast Topology Probe (540-Variant Rectifier Search) | CIFAR-10 | 3-Layer CNN | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 50 | 3 (Fixed: `1, 42, 2026`) |
| **Sec. IV-D** | Macro-Module Ablation | CIFAR-10 | MirrorNetv1 | SGD (Mom. 0.9) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 10 |
| **Sec. V-A** | Multi-Width Pareto Scaling & MirrorAct Transfer | CIFAR-10 | MirrorNetv1, MobileNetV2, ShuffleNetV2, SimpleCNN (3-Layer CNN) | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 10 |
| **Sec. V-B** | Person Detection & $20\text{K}$ Capacity Saturation Probe | VWW ($96\times 96$) | MirrorNetv1, MobileNetV2, SimpleCNN (3-Layer CNN) | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 5 |
| **Sec. V-C** | Ultra-Low Budget ($<20\text{K}$) Micro-Operator Drop-In | VWW ($96\times 96$) | MirrorNetv1 | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 5 |
| **Sec. V-D** | Operator Drop-In & Classifier Head Upward Ablation | VWW ($96\times 96$) | MirrorNetv1 | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 5 |
| **Sec. V-E** | Cross-Resolution Stability & Unassisted INT8 QAT | Wake Vision ($48\times 48$ / $96\times 96$) | MirrorNetv1 | AdamW (WD $1\times 10^{-4}$) / SGD (Mom. 0.9, WD $1\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) / Constant ($10^{-5}$) | 128 | 30 / 10 | 3 |

---

## 2. Dataset Splits & Evaluation Protocols

Based on the official split structures of each benchmark dataset, the protocols for training, validation, and final metric reporting are defined as follows:

| Dataset | Training Split | Validation / Evaluation Split Usage | Metric Selection Protocol |
| :--- | :--- | :--- | :--- |
| **CIFAR-10** | Official `train` split ($50{,}000$ images) | Official `test` split ($10{,}000$ images; used directly as the validation set `val`) | Reports the best validation accuracy achieved during training (`best-val` Top-1 Accuracy). |
| **VisDrone-DET** | Official `train` split ($6{,}471$ images) | Official `val` split ($548$ images; used directly as the evaluation set `test`, as test-set annotations are private) | Reports the best validation performance achieved during training (`best-val`). |
| **VWW** | `train2014`-derived split ($82{,}783$ images) | `val2014`-derived split ($40{,}504$ images; used directly as the evaluation set `test`, as COCO 2014 lacks public test annotations) | Reports the best validation accuracy achieved during training (`best-val` Top-1 Accuracy). |
| **Wake Vision** | Official `train` split ($1{,}196{,}221$ images) | Official `val` split ($18{,}582$ images) for in-training validation; official `test` split ($55{,}763$ images) for final evaluation | Uses `val` to select and save the best model checkpoints (`best-val`) during training; loads the saved checkpoints to evaluate on the held-out `test` split ($55{,}763$ images) and reports **Top-1 Test Accuracy** in the paper. |

---

## 3. Two-Tier Alignment Protocol

To ensure rigorous fairness and prevent conflating the intrinsic mathematical representation efficiency of micro-operators with macro-architectural engineering gains (such as multi-layer Batch Normalization, multiple non-linear activations, and channel expansion), all custom operator-replacement experiments are strictly partitioned into **two evaluation tiers**:

### 3.1 Tier 1: Micro-Operator Drop-In Alignment

- **Applicable Sections:**
  - **Section III-C** (Internal Operator Normalization and Quantization Resilience Probe)
  - **Section V-C** (Micro-Operator Drop-In Benchmark under Ultra-Tight $<20\text{K}$ Parameter Budget, VWW)
  - **Section V-D** (Upward Ablation of Operator Drop-In and Classifier Head Expansion, VWW)
- **Core Alignment Logic (Unified Single External Non-Linearity Socket):**
  In this tier, we perform pure operator-level module replacement. Every candidate operator $\mathcal{O}(\cdot)$ is stripped of its internal Batch Normalization (BN) and non-linear activation functions (Act), and is uniformly encapsulated within an identical external non-linear socket: $\text{Input} \to \mathcal{O}(\cdot) \to \text{Norm} \to \text{Act}$. This ensures all operators are compared fairly under a **single external non-linear transformation**:
  - **`StdConv` (Standard Convolution):** Standard dense $k \times k$ convolution with stride $s$.
  - **`DSC` (Linearized Pure-Operator Variant):** Cascaded `DW -> PW` ($k \times k$ depthwise convolution with stride $s$ followed by $1 \times 1$ pointwise projection) with **intermediate BN and ReLU removed**, functioning as a pure linear spatial-channel factorized operator.
  - **`GhostModule` (Linearized Pure-Operator Variant):** Retains dual-path generation ($r = 2$, primary convolution for $\lceil C_{\text{out}}/2 \rceil$ channels plus cheap depthwise convolution), but **strips internal BN and activations from both branches**, passing through a single unified external `Norm -> Act`.
  - **`PConv + PW` (Plug-and-Play Adapted Variant):** Applies $k \times k$ convolution only to the first $1/4$ channels ($C_p = \max(1, \lfloor C_{\text{in}}/4 \rfloor)$) while identity-mapping the remaining $3/4$ channels (using $2 \times 2$ average pooling for spatial alignment when $s > 1$); followed by a linear $1 \times 1$ pointwise projection (`PConv + PW`) to prevent degeneration into a standard convolution under fixed parameter budgets and to support arbitrary channel mappings ($C_{\text{in}} \to C_{\text{out}}$).

### 3.2 Tier 2: Macro-Architecture Full-Strength Benchmark

- **Applicable Sections:**
  - **Section III-D** (Downstream Complex Object Detection Verification, YOLOv8n / YOLOv8s on VisDrone-DET)
- **Core Alignment Logic (Full-Strength Official Counterparts):**
  In this tier, we evaluate complete engineered building blocks at the system level (Architecture vs. Architecture). All baselines retain their **full-strength engineering designs** from their respective original papers or official repositories (including multi-branch topologies, channel expansion ratios, and multiple internal non-linearities) without artificial stripping:
  - **`DSC` Series (Full-Strength):** Employs the complete `DW -> BN -> ReLU -> PW -> BN -> ReLU` dual-non-linearity block (MobileNetV2 also preserves its $t = 6$ channel expansion and inverted residual structure, with non-destructive stride and classifier head adjustments for low-resolution datasets).
  - **`Ghost` (Official Full-Strength):** Directly invokes the official Ultralytics / original `GhostConv` and `C2f` modules, preserving independent `BN + SiLU` activations after both the primary and cheap convolutions.
  - **`PConv + PW` (Core-Concept Surrogate):** Because the full `FasterNet Block` (with two-layer inverted residual MLP expansion) is structurally incompatible with block-level drop-in replacement in this backbone, **`PConv + PW`** is uniformly adopted across both Tier 1 and Tier 2 as the core-concept surrogate ($1/4$-channel spatial convolution concatenated and fused via $1 \times 1$ pointwise convolution).

---

## 4. Section III-C: Internal Operator Normalization & Quantization Resilience Probe (CIFAR-10)

### 4.1 Training & Architectural Comparison Scope

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch).
- **Dataset & Preprocessing:** CIFAR-10 ($50{,}000$ training images and $10{,}000$ test images) at $32 \times 32 \times 3$ input resolution; training applies standard random cropping to $32 \times 32$ (with $4$-pixel padding, $\text{padding} = 4$) and random horizontal flipping.
- **Normalization Parameters:** Normalized using standard CIFAR-10 per-channel statistics ($\text{mean} = [0.4914, 0.4822, 0.4465]$, $\text{std} = [0.2023, 0.1994, 0.2010]$).
- **Optimization & Training Setup:**
  - **Optimizer & Learning Rate:** `SGD` optimizer ($\text{momentum} = 0.9$) with Cosine Annealing LR schedule ($0.1 \to 10^{-4}$).
  - **Batch Size & Convergence:** Batch size set to $128$, trained for $50$ epochs across $10$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the official `train` split ($50{,}000$ images) and evaluated on the `test` split ($10{,}000$ images) treated as validation (`val`), reporting the best validation accuracy (`best-val` Top-1 Accuracy).
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.
- **Normalization Variants Evaluated:** Based on a 5-Layer CNN probing backbone (Twin mode, ReLU activation), three normalization configurations are evaluated between `Mirror Fusion` and the spatial decoder:
  1. `NONE`: No normalization layer inserted.
  2. `BN`: Standard per-channel Batch Normalization.
  3. `GN`: Stream-wise Group Normalization ($G = 2$), strictly grouped along the $Pos$ and $Neg$ stream boundaries.

### 4.2 Quantization-Oriented Metric Definitions

On clean test batches ($N = 128$), two metrics are computed over the pre-activation `Mirror Fusion` tensors ($L_1$–$L_5$ layers):

1. **Quantization Information Entropy $H(Q)$:** Clean fused tensors $F_{\text{clean}}$ are linearly mapped into $K = 256$ discrete bins to simulate uniform INT8 quantization, computing the Shannon entropy (theoretical upper bound of $8.0\text{ bits}$):

   $$
   H(Q) = -\sum_{k: p_k > 0} p_k \log_2(p_k)
   $$

2. **Statistical Noise Sensitivity $S_{\text{Frob}}$:** Computed over $M = 10$ Monte Carlo trials by injecting isotropic Gaussian noise ($\sigma = 0.05$) and measuring the sample-wise relative Frobenius norm deviation ($\epsilon_0 = 10^{-8}$):

   $$
   S_{\text{Frob}} = \mathbb{E}\left[\frac{\Vert{}F_{\text{noisy}} - F_{\text{clean}}\Vert{}_F}{\Vert{}F_{\text{clean}}\Vert{}_F + \epsilon_0}\right]
   $$

---

## 5. Section III-D: Downstream Object Detection Verification (VisDrone)

### 5.1 Training & Module Replacement Scope

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch / Ultralytics).
- **Dataset & Preprocessing:** VisDrone-DET ($10$ classes, `nc=10`, comprising $6{,}471$ training images and $548$ validation images) at $640 \times 640$ input resolution; follows standard Ultralytics detection augmentations with Mosaic disabled during the final $20$ epochs (`close_mosaic=20`) to focus learning on true-scale small objects.
- **Optimization & Training Setup:**
  - **Initialization & Precision:** All models are trained from scratch (`pretrained=False`) with Automatic Mixed Precision enabled (`amp=True`).
  - **Optimizer & Learning Rate:** Strictly standardized to `AdamW` (`lr0=0.001`) with the Ultralytics linear learning rate decay schedule.
  - **Batch Size & Convergence:** Batch size set to $8$ (`workers=4`), trained for up to $100$ epochs (early stopping `patience=30`), averaged over $3$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the official `train` split ($6{,}471$ images) and evaluated on the `val` split ($548$ images) treated as the test set (`test`), reporting the best validation performance (`best-val`).
- **Complexity Profiling:** Measured under `nc=10` and `imgsz=640` via `DetectionModel` (prioritizing `thop` / `get_flops`, with automatic fallback to PyTorch's native `FlopCounterMode`).
- **Replaced Modules:** Replaces all convolutional blocks (`C2f` and `SPPF` modules) across the **Backbone** and **Neck** of `YOLOv8n` and `YOLOv8s`.
- **Preserved Modules:** The input **Stem** layer and **Detect Head** remain identical across all variants.

---

## 6. Section IV-B: Fast Topology Probe & Rectifier Search (CIFAR-10)

### 6.1 Training & Search Space Scope

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch).
- **Dataset & Preprocessing:** CIFAR-10 ($50{,}000$ training images and $10{,}000$ test images) at $32 \times 32 \times 3$ input resolution; training applies random horizontal flipping and random cropping to $32 \times 32$ ($\text{padding} = 4$).
- **Normalization Parameters:** Normalized using standard CIFAR-10 per-channel statistics ($\text{mean} = [0.4914, 0.4822, 0.4465]$, $\text{std} = [0.2023, 0.1994, 0.2010]$).
- **Optimization & Training Setup:**
  - **Optimizer & Learning Rate:** `SGD` optimizer ($\text{momentum} = 0.9$, $\text{weight decay} = 5 \times 10^{-4}$) with Cosine Annealing LR schedule ($0.1 \to 10^{-4}$).
  - **Batch Size & Convergence:** Batch size set to $128$, trained for $50$ epochs across $3$ fixed random seeds (`1`, `42`, `2026`).
- **Evaluation & Metric Selection:** Trained on the official `train` split ($50{,}000$ images) and evaluated on the `test` split ($10{,}000$ images) treated as validation (`val`), reporting the best validation accuracy (`best-val` Top-1 Accuracy).
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.

---

## 7. Section IV-D: Macro-Architecture Module Ablation (CIFAR-10)

### 7.1 Training & Ablation Scope

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch).
- **Dataset & Preprocessing:** CIFAR-10 ($50{,}000$ training images and $10{,}000$ test images) at $32 \times 32 \times 3$ input resolution; training applies random horizontal flipping and random cropping to $32 \times 32$ ($\text{padding} = 4$).
- **Normalization Parameters:** Normalized using standard CIFAR-10 per-channel statistics ($\text{mean} = [0.4914, 0.4822, 0.4465]$, $\text{std} = [0.2023, 0.1994, 0.2010]$).
- **Optimization & Training Setup:**
  - **Optimizer & Learning Rate:** `SGD` optimizer ($\text{momentum} = 0.9$) with Cosine Annealing LR schedule ($0.1 \to 10^{-4}$).
  - **Batch Size & Convergence:** Batch size set to $128$, extended to $100$ epochs to ensure full convergence, evaluated across $10$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the official `train` split ($50{,}000$ images) and evaluated on the `test` split ($10{,}000$ images) treated as validation (`val`), reporting the best validation accuracy (`best-val` Top-1 Accuracy).
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.

---

## 8. Section V-A: Multi-Width Pareto Scaling & MirrorAct Portability (CIFAR-10)

### 8.1 Training, Baseline Alignment & Evaluation Scope

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch / ONNX Runtime).
- **Dataset & Preprocessing:** CIFAR-10 ($50{,}000$ training images and $10{,}000$ test images) at $32 \times 32 \times 3$ input resolution; training applies random horizontal flipping and random cropping to $32 \times 32$ ($\text{padding} = 4$).
- **Normalization Parameters:** Normalized using standard CIFAR-10 per-channel statistics ($\text{mean} = [0.4914, 0.4822, 0.4465]$, $\text{std} = [0.2023, 0.1994, 0.2010]$).
- **Optimization & Training Setup:**
  - **Initialization:** All configurations are trained from scratch.
  - **Optimizer & Learning Rate:** `SGD` optimizer ($\text{momentum} = 0.9$, $\text{weight decay} = 5 \times 10^{-4}$) with Cosine Annealing LR schedule ($0.1 \to 10^{-4}$).
  - **Batch Size & Convergence:** Batch size set to $128$, trained for $100$ epochs across $10$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the official `train` split ($50{,}000$ images) and evaluated on the `test` split ($10{,}000$ images) treated as validation (`val`), reporting the best validation accuracy (`best-val` Top-1 Accuracy).
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.
- **Low-Resolution Structural Alignment (Unified to $4 \times 4$ Pre-GAP Feature Map Resolution):**
  - **`MirrorNetv1` Scaling Sequence:** Evaluates width multipliers $\alpha \in \{\texttt{x0.5}, \texttt{x1.0}, \texttt{x2.0}, \texttt{x4.0}, \texttt{x10.0}\}$, spanning ultra-compact to high-capacity regimes.
  - **`ShuffleNetV2` Structural Alignment:** Reduces the initial convolution stride to $1$ and removes the subsequent `MaxPool` layer while preserving the three spatial downsampling stages (Stages 2–4), scaling channels across $\alpha \in \{0.1, 0.25, 0.5, 1.0\}$.
  - **`MobileNetV2` Structural Alignment (Eliminating the "Fat Head" Bottleneck):** Sets early-layer strides to $1$ to prevent spatial resolution collapse and replaces the fixed $1280$-channel final $1 \times 1$ convolution ("Fat Head") with proportional scaling governed by width multiplier $\alpha$.
  - **`SimpleCNN` (3-Layer CNN) Portability Verification:** Compares standard `ReLU` against `MirrorReLU` within a generic 3-layer CNN backbone to verify cross-architecture portability independent of specific macro-topologies.

---

## 9. Section V-B: Person Detection Benchmark & $20\text{K}$ Capacity Saturation Probe (VWW)

### 9.1 Dataset Construction, Training & Probing Scope

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch).
- **Dataset & Preprocessing:** Visual Wake Words (VWW) at $96 \times 96 \times 3$ input resolution. Custom-extracted from COCO 2014 following the standard protocol (images are labeled positive only if a person bounding box occupies $> 0.5\%$ of the total image area, yielding $82{,}783$ training images and $40{,}504$ test images); preprocessing applies only random horizontal flipping, pre-packed into compressed binary arrays (`.npz`) to eliminate training I/O bottlenecks.
- **Normalization Parameters:** Normalized using standard ImageNet per-channel statistics ($\text{mean} = [0.485, 0.456, 0.406]$, $\text{std} = [0.229, 0.224, 0.225]$).
- **Optimization & Training Setup:**
  - **Initialization:** All models are trained from scratch.
  - **Optimizer & Learning Rate:** `SGD` optimizer ($\text{momentum} = 0.9$, $\text{weight decay} = 5 \times 10^{-4}$) with Cosine Annealing LR schedule ($0.1 \to 10^{-4}$).
  - **Batch Size & Convergence:** Batch size set to $128$, trained for $100$ epochs across $5$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the `train2014`-derived split ($82{,}783$ images) and evaluated on the `val2014`-derived split ($40{,}504$ images) treated as the test set (`test`), reporting the best validation accuracy (`best-val` Top-1 Accuracy).
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.

---

## 10. Section V-C: Ultra-Low Budget ($<20\text{K}$) Micro-Operator Drop-In Benchmark (VWW)

### 10.1 Training & Operator Replacement Scope (Tier 1 Alignment)

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch).
- **Dataset & Preprocessing:** Visual Wake Words (VWW) at $96 \times 96 \times 3$ input resolution. Custom-extracted from COCO 2014 ($> 0.5\%$ person bounding-box area threshold, yielding $82{,}783$ training images and $40{,}504$ test images); preprocessing applies only random horizontal flipping, pre-packed into compressed binary arrays (`.npz`) to eliminate training I/O bottlenecks.
- **Normalization Parameters:** Normalized using standard ImageNet per-channel statistics ($\text{mean} = [0.485, 0.456, 0.406]$, $\text{std} = [0.229, 0.224, 0.225]$).
- **Optimization & Training Setup:**
  - **Optimizer & Learning Rate:** `SGD` optimizer ($\text{momentum} = 0.9$, $\text{weight decay} = 5 \times 10^{-4}$) with Cosine Annealing LR schedule ($0.1 \to 10^{-4}$).
  - **Batch Size & Convergence:** Batch size set to $128$, trained for $100$ epochs across $5$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the `train2014`-derived split ($82{,}783$ images) and evaluated on the `val2014`-derived split ($40{,}504$ images) treated as the test set (`test`), reporting the best validation accuracy (`best-val` Top-1 Accuracy).
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.

---

## 11. Section V-D: Operator Drop-In & Classifier Head Upward Ablation (VWW)

### 11.1 Training & Upward Ablation Scope

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch).
- **Dataset & Preprocessing:** Visual Wake Words (VWW) at $96 \times 96 \times 3$ input resolution. Custom-extracted from COCO 2014 ($> 0.5\%$ person bounding-box area threshold, yielding $82{,}783$ training images and $40{,}504$ test images); preprocessing applies only random horizontal flipping, pre-packed into compressed binary arrays (`.npz`) to eliminate training I/O bottlenecks.
- **Normalization Parameters:** Normalized using standard ImageNet per-channel statistics ($\text{mean} = [0.485, 0.456, 0.406]$, $\text{std} = [0.229, 0.224, 0.225]$).
- **Optimization & Training Setup:**
  - **Optimizer & Learning Rate:** `SGD` optimizer ($\text{momentum} = 0.9$, $\text{weight decay} = 5 \times 10^{-4}$) with Cosine Annealing LR schedule ($0.1 \to 10^{-4}$).
  - **Batch Size & Convergence:** Batch size set to $128$, trained for $100$ epochs across $5$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the `train2014`-derived split ($82{,}783$ images) and evaluated on the `val2014`-derived split ($40{,}504$ images) treated as the test set (`test`), reporting the best validation accuracy (`best-val` Top-1 Accuracy).
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.

---

## 12. Section V-E: Cross-Resolution Stability & Unassisted INT8 QAT (Wake Vision)

### 12.1 Cross-Resolution Data Processing, Two-Stage Quantization & Evaluation Scope (Tier 2 Alignment)

- **Hardware Environment:** Single NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch).
- **Dataset & Preprocessing:** Wake Vision ($1{,}196{,}221$ training images, $18{,}582$ validation images, and $55{,}763$ held-out test images), processed into both high-compression **$48 \times 48$** and high-fidelity **$96 \times 96$** input resolutions. Both resolutions share an identical augmentation pipeline consisting of random horizontal flipping ($p = 0.5$), random rotation (up to $15^\circ$), and color jittering (brightness, contrast, and saturation bounded within $0.2$).
- **Normalization Parameters:** Pixel values are linearly normalized to the $[-1, 1]$ dynamic range ($\text{mean} = [0.5, 0.5, 0.5]$, $\text{std} = [0.5, 0.5, 0.5]$).
- **Optimization & Training Setup:**
  - **Stage 1 (FP32 Pre-Training):** Trained without knowledge distillation (KD) or teacher model assistance, using `AdamW` ($\text{weight decay} = 1 \times 10^{-4}$) with a Cosine Annealing LR schedule ($0.1 \to 10^{-4}$) for $30$ epochs.
  - **Stage 2 (INT8 Quantization-Aware Training, QAT):** Switched to `SGD` ($\text{momentum} = 0.9$, $\text{weight decay} = 1 \times 10^{-4}$) with a fixed Constant LR ($10^{-5}$) for $10$ epochs of quantization-aware fine-tuning.
  - **Batch Size & Convergence:** Batch size set to $128$, evaluated across $3$ independent random seeds.
- **Evaluation & Metric Selection:** Trained on the official `train` split ($1{,}196{,}221$ images) while using the `val` split ($18{,}582$ images) for in-training validation and checkpoint selection (saving `best-val` checkpoints independently for FP32 and INT8 QAT); final paper metrics are obtained by loading the saved checkpoints and evaluating them on the held-out `test` split ($55{,}763$ images) to report **Top-1 Test Accuracy**.
- **Complexity Profiling:** Parameter counts and FLOPs are profiled using `fvcore`.