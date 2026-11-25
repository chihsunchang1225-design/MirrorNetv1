# 基準測試設定與實驗規範 (Benchmark Configurations & Experimental Protocols)

本文件詳細說明重現 **MirrorNetv1** 論文實驗所需的軟硬體環境、訓練超參數、結構對齊規則以及評估指標定義。*（具體的量化實驗數據與實證結果表格，請參閱論文主文。）*

> **統一硬體與軟體執行環境 (Hardware & Software Environment)**
> 本專案中的所有模型訓練、架構探測與消融實驗皆基於 **PyTorch** 框架實作，並統一於單張 **NVIDIA GeForce RTX 5060 Laptop GPU** 上從頭完成訓練與評估。

---

## 1. 訓練超參數總覽

| 論文章節 | 基準測試 / 任務 | 資料集 | 骨幹網路 | 優化器 | 學習率排程 (LR Schedule) | 批次大小 (Batch Size) | 訓練週期 (Epochs) | 隨機種子數 (Seeds) |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Sec. III-C** | 流向正規化探測 (Stream-Wise Norm. Probe) | CIFAR-10 | 5-Layer CNN (Twin, ReLU) | SGD (Mom. 0.9) | Cosine ($0.1 \to 10^{-4}$) | 128 | 50 | 10 |
| **Sec. III-D** | 複雜物件偵測 (Complex Object Detection) | VisDrone-DET | YOLOv8n / YOLOv8s (Backbone+Neck) | AdamW (`lr0=0.001`, `pretrained=False`) | Ultralytics 線性衰減 (`close_mosaic=20`) | 8 | 100 (`patience=30`) | 3 |
| **Sec. IV-B** | 快速拓撲探測 (540-Variant Rectifier Search) | CIFAR-10 | 3-Layer CNN | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 50 | 3（固定：`1, 42, 2026`） |
| **Sec. IV-D** | 巨觀模組消融實驗 (Macro-Module Ablation) | CIFAR-10 | MirrorNetv1 | SGD (Mom. 0.9) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 10 |
| **Sec. V-A** | 多寬度 Pareto 縮放與 MirrorAct 移植 | CIFAR-10 | MirrorNetv1、MobileNetV2、ShuffleNetV2、SimpleCNN(3-Layer CNN) | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 10 |
| **Sec. V-B** | 人物偵測與 $20\text{K}$ 容量飽和探測 | VWW ($96\times 96$) | MirrorNetv1、MobileNetV2、SimpleCNN(3-Layer CNN) | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 5 |
| **Sec. V-C** | 極限預算（$<20\text{K}$）微觀算子替換 | VWW ($96\times 96$) | MirrorNetv1 | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 5 |
| **Sec. V-D** | 算子替換與分類頭更換向上消融 | VWW ($96\times 96$) | MirrorNetv1 | SGD (Mom. 0.9, WD $5\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) | 128 | 100 | 5 |
| **Sec. V-E** | 跨解析度穩定性與無輔助 INT8 QAT | Wake Vision ($48\times 48$ / $96\times 96$) | MirrorNetv1 | AdamW (WD $1\times 10^{-4}$) / SGD (Mom. 0.9, WD $1\times 10^{-4}$) | Cosine ($0.1 \to 10^{-4}$) / Constant ($10^{-5}$) | 128 | 30 / 10 | 3 |

---

## 2. 資料集切分與準確率評估規範 (Dataset Splits & Evaluation Protocols)

針對各基準資料集官方公開的切分結構，本專案對訓練、驗證與最終指標採計定義如下：

| 資料集 | 訓練集 (Train Split) | 驗證 / 測試集切分用途 | 指標採計規範 (Metric Selection) |
| :--- | :--- | :--- | :--- |
| **CIFAR-10** | 官方 `train` 集（$50{,}000$ 張） | 官方 `test` 集（$10{,}000$ 張，直接作為驗證集 `val` 使用） | 採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。 |
| **VisDrone-DET** | 官方 `train` 集（$6{,}471$ 張） | 官方 `val` 集（$548$ 張，因官方未公開測試集標註，直接作為測試集 `test` 使用） | 採計訓練週期內之最佳驗證集表現（`best-val`）。 |
| **VWW** | `train2014` 提取集（$82{,}783$ 張） | `val2014` 提取集（$40{,}504$ 張，因 COCO 2014 未公開測試集標註，直接作為測試集 `test` 使用） | 採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。 |
| **Wake Vision** | 官方 `train` 集（$1{,}196{,}221$ 張） | 官方 `val` 集（$18{,}582$ 張）用於訓練階段驗證；官方 `test` 集（$55{,}763$ 張）用於最終評估 | 訓練階段依 `val` 篩選並保存最佳模型權重（`best-val`）；論文最終數據統一載入最佳權重於獨立 `test` 測試集（$55{,}763$ 張）評估並匯報 **Top-1 Test Accuracy**。 |

---

## 3. 雙層基準測試對齊規範 (Two-Tier Alignment Protocol)

為確保實驗公平性，避免將「微觀算子本身的數學表徵效率」與「巨觀網路的工程堆疊紅利（如多層 BN、多重非線性激勵、通道擴張）」混為一談，本專案自建算子替換的所有實驗嚴格切分為**兩大評估層級**，各自採用截然不同的模組對齊標準：

### 3.1 第一層級：算子對算子（Micro-Operator Drop-In Alignment）

- **適用實驗範圍：** 
  - **Section III-C**（算子內部正規化與量化韌性探測）
  - **Section V-C**（極限 $<20\text{K}$ 參數預算下的微觀算子替換基準，VWW）
  - **Section V-D**（結合算子替換與分類頭更換之向上消融實驗，VWW）
- **核心對齊邏輯（統一單層非線性插座）：**
  在此層級中，我們進行的是純粹的「算子級別模組替換」。所有候選算子 $\mathcal{O}(\cdot)$ 皆剝除其內部自帶的批次正規化（BN）與非線性激勵函數（Act），統一封裝進完全相同的外部非線性插座 $\text{Input} \to \mathcal{O}(\cdot) \to \text{Norm} \to \text{Act}$ 中，確保所有算子都在**單一外部非線性變換**下進行公平比較：
  - **`StdConv`（標準卷積）：** 步長為 $s$ 的標準 $k \times k$ 密集卷積。
  - **`DSC`（線性化純算子版）：** 採用 `DW -> PW` 串接（步長 $s$ 的 $k \times k$ 深度卷積緊接 $1 \times 1$ 逐點投影），**拿掉中間的 BN 與 ReLU**，使其實質上成為等效於單層卷積的線性空間-通道分解算子。
  - **`GhostModule`（線性化純算子版）：** 保留雙路徑生成（$r = 2$，主卷積 $\lceil C_{\text{out}}/2 \rceil$ 通道 + 廉價深度卷積），但**移除兩條分支內部的 BN 與激勵函數**，僅在外部經過一次統一的 `Norm -> Act`。
  - **`PConv + PW`（即插即用適配版）：** 僅對前 $1/4$ 通道（$C_p = \max(1, \lfloor C_{\text{in}}/4 \rfloor)$）執行 $k \times k$ 卷積，其餘 $3/4$ 通道直接恆等映射（若 $s > 1$ 則以 $2 \times 2$ 平均池化對齊空間尺寸）；拼接後補上一層線性 $1 \times 1$ 投影（`PConv + PW`）以防止在固定參數預算下退化為標準卷積，並滿足任意通道數轉換（$C_{\text{in}} \to C_{\text{out}}$）。

### 3.2 第二層級：巨觀架構系統比較（Macro-Architecture Full-Strength Benchmark）

- **適用實驗範圍：** 
  - **Section III-D**（下游複雜物件偵測驗證，YOLOv8n / YOLOv8s on VisDrone-DET）
- **核心對齊邏輯（官方滿血版對決）：**
  在此層級中，我們評估的是完整工程區塊在真實任務上的「系統級表現（Architecture vs. Architecture）」。所有對照組皆保留其**原論文或官方開源庫的「滿血工程設計」**（包含多分支拓撲、通道擴張率與多重內部非線性），不進行任何人為剝除：
  - **`DSC` 系列（滿血版）：** 採用完整的 `DW -> BN -> ReLU -> PW -> BN -> ReLU` 雙重非線性區塊（MobileNetV2 亦保留 $t = 6$ 通道擴張與倒殘差結構，並針對低解析度資料集進行非破壞性步長與分類頭對齊）。
  - **`Ghost`（官方滿血版）：** 直接調用 Ultralytics / 原論文的 `GhostConv` 與 `C2f` 模組，保留主卷積與廉價卷積後方各自獨立的 `BN + SiLU` 雙重激勵紅利。
  - **`PConv + PW`（核心概念替代版）：** 由於 FasterNet 原論文的完整 `FasterNet Block`（包含雙層倒殘差 MLP 擴張）不適用於本任務的模組替換架構，因此在 Tier 1 與 Tier 2 中皆統一採用 **`PConv + PW`** 作為其核心設計概念的替代實作（即 $1/4$ 通道空間卷積拼接後接 $1\times 1$ 點卷積融合）。

---

## 4. Section III-C：算子內部正規化與量化韌性探測 (CIFAR-10)

### 4.1 訓練與架構對照範圍

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch）。
- **資料集與前處理：** CIFAR-10（共計 $50{,}000$ 張訓練影像與 $10{,}000$ 張測試影像），輸入解析度為 $32 \times 32 \times 3$；訓練階段套用標準隨機裁切至 $32 \times 32$（邊緣填充 $4$ 像素，Padding $= 4$）與隨機水平翻轉進行資料增強。
- **正規化參數：** 採用 CIFAR-10 標準通道均值與標準差進行正規化（$\text{mean} = [0.4914, 0.4822, 0.4465]$，$\text{std} = [0.2023, 0.1994, 0.2010]$）。
- **優化與訓練設定：**
  - **優化器與學習率：** 採用 `SGD` 優化器（動量 Momentum $0.9$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$）。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，訓練週期為 $50$ 個 Epochs，於 $10$ 組獨立隨機種子下進行評估。
- **評估與指標採計：** 使用官方 `train` 集（$50{,}000$ 張）進行訓練，並以 `test` 集（$10{,}000$ 張）作為驗證集（Val），最終採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。
- **評估與正規化變體對照：** 基於 5-Layer CNN（Twin 模式，ReLU 激勵函數）探測骨幹，在 `Mirror Fusion` 與空間解碼器之間評估以下三種正規化變體：
  1. `NONE`：不加入任何正規化層。
  2. `BN`：標準逐通道批次正規化（Per-channel Batch Normalization）。
  3. `GN`：流向分組正規化（Stream-wise Group Normalization，$G = 2$，嚴格沿著 $Pos$ 與 $Neg$ 流形邊界分組）。

### 4.2 量化導向評估指標定義

於乾淨測試批次（$N = 128$）中，針對激勵函數前的 `Mirror Fusion` 張量（$L_1$–$L_5$ 層）計算以下兩項指標：

1. **量化資訊熵 $H(Q)$（Quantization Information Entropy）：** 將乾淨的融合張量 $F_{\text{clean}}$ 線性映射至 $K = 256$ 個離散區間以模擬均勻 INT8 量化，並計算香農熵（Shannon Entropy，理論上限為 $8.0\text{ bits}$）：

   $$
   H(Q) = -\sum_{k: p_k > 0} p_k \log_2(p_k)
   $$

2. **統計雜訊敏感度 $S_{\text{Frob}}$（Statistical Noise Sensitivity）：** 進行 $M = 10$ 次蒙地卡羅試驗，每次注入各向同性高斯雜訊（$\sigma = 0.05$），並計算逐樣本相對 Frobenius 範數偏移量（$\epsilon_0 = 10^{-8}$）：

   $$
   S_{\text{Frob}} = \mathbb{E}\left[\frac{\Vert{}F_{\text{noisy}} - F_{\text{clean}}\Vert{}_F}{\Vert{}F_{\text{clean}}\Vert{}_F + \epsilon_0}\right]
   $$

---

## 5. Section III-D：下游物件偵測任務驗證 (VisDrone)

### 5.1 訓練與模組替換範圍

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch / Ultralytics）。
- **資料集與前處理：** VisDrone-DET（$10$ 類別，`nc=10`，共計 $6{,}471$ 張訓練影像與 $548$ 張驗證影像），輸入解析度為 $640 \times 640$；沿用 Ultralytics 預設偵測資料增強，並設定於最後 $20$ 個 Epochs 關閉 Mosaic 增強（`close_mosaic=20`），讓模型專注於學習真實尺寸的小型目標。
- **優化與訓練設定：**
  - **初始化與精度：** 所有模型皆從頭隨機初始化訓練（`pretrained=False`），並啟用自動混合精度訓練（`amp=True`）。
  - **優化器與學習率：** 統一強制採用 `AdamW` 優化器（初始學習率 `lr0=0.001`）搭配 Ultralytics 線性學習率衰減排程。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $8$（`workers=4`），最大訓練週期為 $100$ 個 Epochs（早停機制 `patience=30`），最終結果取 $3$ 組獨立隨機種子之平均值。
- **評估與指標採計：** 使用官方 `train` 集（$6{,}471$ 張）進行訓練，並以 `val` 集（$548$ 張）作為測試集（Test），最終採計訓練週期內之最佳驗證集表現（`best-val`）。
- **複雜度統計規範：** 於 `nc=10`、`imgsz=640` 設定下，透過 `DetectionModel` 統計實際參數量與 GFLOPs（優先採用 `thop` / `get_flops`，若無則自動切換至 PyTorch 原生 `FlopCounterMode`）。
- **替換模組範圍：** 替換 `YOLOv8n` 與 `YOLOv8s` 的 **Backbone** 與 **Neck** 中所有卷積區塊（`C2f` 與 `SPPF` 模組）。
- **保留未改動模組：** 所有變體的輸入 **Stem** 層與 **Detect Head** 偵測頭皆保持完全一致。

---

## 6. Section IV-B：快速拓撲探測與 Rectifier 搜尋 (CIFAR-10)

### 6.1 訓練與搜尋空間範圍

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch）。
- **資料集與前處理：** CIFAR-10（共計 $50{,}000$ 張訓練影像與 $10{,}000$ 張測試影像），輸入解析度為 $32 \times 32 \times 3$；訓練階段套用隨機水平翻轉與隨機裁切至 $32 \times 32$（邊緣填充 $4$ 像素，Padding $= 4$）。
- **正規化參數：** 採用 CIFAR-10 標準通道均值與標準差進行正規化（$\text{mean} = [0.4914, 0.4822, 0.4465]$，$\text{std} = [0.2023, 0.1994, 0.2010]$）。
- **優化與訓練設定：**
  - **優化器與學習率：** 採用 `SGD` 優化器（動量 Momentum $0.9$、權重衰減 Weight Decay $5 \times 10^{-4}$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$）。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，訓練週期為 $50$ 個 Epochs，於 $3$ 組固定隨機種子（`1`、`42`、`2026`）下進行評估。
- **評估與指標採計：** 使用官方 `train` 集（$50{,}000$ 張）進行訓練，並以 `test` 集（$10{,}000$ 張）作為驗證集（Val），最終採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。

---

## 7. Section IV-D：巨觀架構模組消融實驗 (CIFAR-10)

### 7.1 訓練與消融驗證範圍

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch）。
- **資料集與前處理：** CIFAR-10（共計 $50{,}000$ 張訓練影像與 $10{,}000$ 張測試影像），輸入解析度為 $32 \times 32 \times 3$；訓練階段套用隨機水平翻轉與隨機裁切至 $32 \times 32$（邊緣填充 $4$ 像素，Padding $= 4$）。
- **正規化參數：** 採用 CIFAR-10 標準通道均值與標準差進行正規化（$\text{mean} = [0.4914, 0.4822, 0.4465]$，$\text{std} = [0.2023, 0.1994, 0.2010]$）。
- **優化與訓練設定：**
  - **優化器與學習率：** 採用 `SGD` 優化器（動量 Momentum $0.9$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$）。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，訓練週期延長至 $100$ 個 Epochs 以確保模型完全收斂，於 $10$ 組獨立隨機種子下進行評估。
- **評估與指標採計：** 使用官方 `train` 集（$50{,}000$ 張）進行訓練，並以 `test` 集（$10{,}000$ 張）作為驗證集（Val），最終採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。

---

## 8. Section V-A：多寬度 Pareto 縮放與 MirrorAct 可移植性驗證 (CIFAR-10)

### 8.1 訓練、基準結構對齊與評估範圍

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch / ONNX Runtime）。
- **資料集與前處理：** CIFAR-10（共計 $50{,}000$ 張訓練影像與 $10{,}000$ 張測試影像），輸入解析度為 $32 \times 32 \times 3$；訓練階段套用隨機水平翻轉與隨機裁切至 $32 \times 32$（邊緣填充 $4$ 像素，Padding $= 4$）。
- **正規化參數：** 採用 CIFAR-10 標準通道均值與標準差進行正規化（$\text{mean} = [0.4914, 0.4822, 0.4465]$，$\text{std} = [0.2023, 0.1994, 0.2010]$）。
- **優化與訓練設定：**
  - **初始化設定：** 所有配置皆從頭隨機初始化訓練。
  - **優化器與學習率：** 採用 `SGD` 優化器（動量 Momentum $0.9$、權重衰減 Weight Decay $5 \times 10^{-4}$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$）。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，訓練週期為 $100$ 個 Epochs，於 $10$ 組獨立隨機種子下進行評估。
- **評估與指標採計：** 使用官方 `train` 集（$50{,}000$ 張）進行訓練，並以 `test` 集（$10{,}000$ 張）作為驗證集（Val），最終採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。
- **評估與低解析度結構對齊範圍（統一對齊至 GAP 前 $4 \times 4$ 解析度）：**
  - **`MirrorNetv1` 縮放序列：** 測試通道寬度乘數 $\alpha \in \{\texttt{x0.5}, \texttt{x1.0}, \texttt{x2.0}, \texttt{x4.0}, \texttt{x10.0}\}$，涵蓋極限微型至高容量區間。
  - **`ShuffleNetV2` 結構對齊：** 將初始卷積步長降為 $1$ 並移除隨後的 `MaxPool`，完整保留 Stage 2–4 的三次空間降採樣，通道數隨 $\alpha \in \{0.1, 0.25, 0.5, 1.0\}$ 動態縮放。
  - **`MobileNetV2` 結構對齊（消除「Fat Head」瓶頸）：** 將早期層步長改為 $1$ 以防止解析度崩潰，並移除末端 $1\times 1$ 卷積中固定的 $1280$ 通道瓶頸（Fat Head），改為隨寬度乘數 $\alpha$ 動態等比例縮放。
  - **`SimpleCNN`（3-Layer CNN）算子移植驗證：** 在通用 3 層 CNN 骨幹中對比標準 `ReLU` 與 `MirrorReLU`，驗證 `MirrorAct` 脫離特定巨觀拓撲後的跨架構可移植性。

---

## 9. Section V-B：人物偵測基準與 $20\text{K}$ 容量飽和探測 (VWW)

### 9.1 資料集建構、訓練與容量探測範圍

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch）。
- **資料集與前處理：** Visual Wake Words（VWW），輸入解析度為 $96 \times 96 \times 3$。由 COCO 2014 資料集客製化提取（僅當人物邊界框面積超過總影像面積的 $0.5\%$ 時標記為正樣本，共計 $82{,}783$ 張訓練影像與 $40{,}504$ 張測試影像）；前處理僅套用隨機水平翻轉，預先封裝為壓縮二進位陣列（`.npz`）以消除訓練 I/O 瓶頸。
- **正規化參數：** 採用 ImageNet 標準通道均值與標準差進行正規化（$\text{mean} = [0.485, 0.456, 0.406]$，$\text{std} = [0.229, 0.224, 0.225]$）。
- **優化與訓練設定：**
  - **初始化設定：** 所有模型皆從頭隨機初始化訓練。
  - **優化器與學習率：** 採用 `SGD` 優化器（動量 Momentum $0.9$、權重衰減 Weight Decay $5 \times 10^{-4}$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$）。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，訓練週期為 $100$ 個 Epochs，於 $5$ 組獨立隨機種子下進行評估。
- **評估與指標採計：** 使用 `train2014` 提取集（$82{,}783$ 張）進行訓練，並以 `val2014` 提取集（$40{,}504$ 張）作為測試集（Test），最終採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。

---

## 10. Section V-C：極限預算（$<20\text{K}$）微觀算子替換基準 (VWW)

### 10.1 訓練與算子替換範圍（Tier 1 對齊）

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch）。
- **資料集與前處理：** Visual Wake Words（VWW），輸入解析度為 $96 \times 96 \times 3$。由 COCO 2014 資料集客製化提取（僅當人物邊界框面積超過總影像面積的 $0.5\%$ 時標記為正樣本，共計 $82{,}783$ 張訓練影像與 $40{,}504$ 張測試影像）；前處理僅套用隨機水平翻轉，預先封裝為壓縮二進位陣列（`.npz`）以消除訓練 I/O 瓶頸。
- **正規化參數：** 採用 ImageNet 標準通道均值與標準差進行正規化（$\text{mean} = [0.485, 0.456, 0.406]$，$\text{std} = [0.229, 0.224, 0.225]$）。
- **優化與訓練設定：**
  - **優化器與學習率：** 採用 `SGD` 優化器（動量 Momentum $0.9$、權重衰減 Weight Decay $5 \times 10^{-4}$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$）。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，訓練週期為 $100$ 個 Epochs，於 $5$ 組獨立隨機種子下進行評估。
- **評估與指標採計：** 使用 `train2014` 提取集（$82{,}783$ 張）進行訓練，並以 `val2014` 提取集（$40{,}504$ 張）作為測試集（Test），最終採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。

---

## 11. Section V-D：算子替換與分類頭更換向上消融實驗 (VWW)

### 11.1 訓練與向上消融範圍

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch）。
- **資料集與前處理：** Visual Wake Words（VWW），輸入解析度為 $96 \times 96 \times 3$。由 COCO 2014 資料集客製化提取（僅當人物邊界框面積超過總影像面積的 $0.5\%$ 時標記為正樣本，共計 $82{,}783$ 張訓練影像與 $40{,}504$ 張測試影像）；前處理僅套用隨機水平翻轉，預先封裝為壓縮二進位陣列（`.npz`）以消除訓練 I/O 瓶頸。
- **正規化參數：** 採用 ImageNet 標準通道均值與標準差進行正規化（$\text{mean} = [0.485, 0.456, 0.406]$，$\text{std} = [0.229, 0.224, 0.225]$）。
- **優化與訓練設定：**
  - **優化器與學習率：** 採用 `SGD` 優化器（動量 Momentum $0.9$、權重衰減 Weight Decay $5 \times 10^{-4}$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$）。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，訓練週期為 $100$ 個 Epochs，於 $5$ 組獨立隨機種子下進行評估。
- **評估與指標採計：** 使用 `train2014` 提取集（$82{,}783$ 張）進行訓練，並以 `val2014` 提取集（$40{,}504$ 張）作為測試集（Test），最終採計訓練週期內之最佳驗證準確率（`best-val` Top-1 Accuracy）。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。

---

## 12. Section V-E：跨解析度穩定性與無輔助 INT8 QAT 驗證 (Wake Vision)

### 12.1 跨解析度資料處理、兩階段量化訓練與評估範圍（Tier 2 對齊）

- **硬體環境：** 單張 NVIDIA GeForce RTX 5060 Laptop GPU（PyTorch）。
- **資料集與前處理：** Wake Vision（共計 $1{,}196{,}221$ 張訓練影像、$18{,}582$ 張驗證影像與 $55{,}763$ 張獨立測試影像），分別處理為高壓縮 **$48 \times 48$** 與高保真 **$96 \times 96$** 兩種輸入解析度；兩組解析度共用相同資料增強管線，包含隨機水平翻轉（$p = 0.5$）、隨機旋轉（最大角度 $15^\circ$）、色彩抖動（亮度、對比度與飽和度變化限制於 $0.2$ 以內）。
- **正規化參數：** 將像素值線性正規化至 $[-1, 1]$ 動態範圍（$\text{mean} = [0.5, 0.5, 0.5]$，$\text{std} = [0.5, 0.5, 0.5]$）。
- **優化與訓練設定：**
  - **第一階段（FP32 預訓練）：** 不使用知識蒸餾（KD）或教師模型輔助，採用 `AdamW` 優化器（權重衰減 Weight Decay $1 \times 10^{-4}$）搭配餘弦退火學習率排程（Cosine Annealing LR，$0.1 \to 10^{-4}$），訓練 $30$ 個 Epochs。
  - **第二階段（INT8 量化感知訓練 QAT）：** 切換為 `SGD` 優化器（動量 Momentum $0.9$、權重衰減 Weight Decay $1 \times 10^{-4}$），並直接採用固定學習率（Constant LR，$10^{-5}$），進行 $10$ 個 Epochs 的量化感知微調。
  - **批次與收斂控制：** 批次大小（Batch Size）設為 $128$，於 $3$ 組獨立隨機種子下進行評估。
- **評估與指標採計：** 使用官方 `train` 集（$1{,}196{,}221$ 張）進行訓練，並以 `val` 集（$18{,}582$ 張）進行訓練過程驗證與最佳模型權重篩選（FP32 與 INT8 QAT 階段各自保存 `best-val` 權重）；論文最終數據則統一載入最佳權重於獨立的 `test` 測試集（$55{,}763$ 張）上執行推論，匯報最終 **Top-1 Test Accuracy**。
- **複雜度統計規範：** 透過 `fvcore` 統計模型參數量與 FLOPs。