# Intro2AI Project - Comic Colorization Tool

> Team 61 Final Project

- 本專案旨在開發一個使用深度學習技術為黑白漫畫圖像自動上色的系統。系統基於 U-Net 架構（可作為獨立模型或 GAN 的生成器），透過學習大量成對的黑白與彩色漫畫圖像，實現對新的黑白漫畫進行上色。

## 專案特色與近期改進

在最近的開發迭代中，我們進行了以下主要的更新與優化：

1.  **模型架構 (`model.py`, `baseline.py`)**:
    *   `model.py` 包含 `unet_advanced_prelu` (U-Net 生成器), `build_discriminator` (PatchGAN 判別器), 和 `define_gan` (組合 GAN 模型) 的定義。
    *   `baseline.py` 包含其他 U-Net 生成器模型如 `unet_vgg16` 和 `unet_relu_leaky` (也被稱為 `best_version`)。
    *   所有 U-Net 生成器輸入圖像尺寸統一為 **512x512**。
    *   調整了 `embed_input` (形狀 1000) 的融合邏輯，以適應 512x512 的輸入。
    *   U-Net 模型編譯時使用 `Adam` 優化器，損失函數可在 MSE (`MeanSquaredError`) 和 MAE (`MeanAbsoluteError` aka L1 Loss) 之間選擇。
    *   GAN 訓練流程已在 `train.py` 中實現，包含判別器和生成器的交替訓練，以及對抗性損失與 L1 重建損失的組合。

2.  **資料預處理與轉換 (`file_converter.py`, `train.py`)**:
    *   `file_converter.py`: 提供將原始彩色圖像轉換為模型所需的黑白 (L 通道) 輸入圖像的功能，並可 resize 圖像至 512x512。使用 `tqdm` 顯示進度。
    *   `train.py` 中的 `load_and_preprocess_data`:
        *   載入黑白 L 通道圖像 (輸入) 和原始彩色圖像 (目標)。
        *   所有圖像 resize 到 512x512。
        *   L 通道標準化到 `[0, 1]`。
        *   彩色圖像轉為 Lab 色彩空間，ab 通道作為目標，標準化到 `[-1, 1]`。
        *   `embed_input` 目前設定為從標準正態分佈採樣的**隨機向量** (長度 1000)，旨在增加輸出顏色的多樣性。
        *   使用 **多進程 (`multiprocessing`)** 並行處理圖像載入和預處理，顯著提升資料準備效率，並使用 `tqdm` 顯示進度。

3.  **模型訓練 (`train.py`)**:
    *   使用 `argparse` 接收命令列參數，包括資料夾路徑、模型類型、訓練輪次 (epochs)、批次大小 (batch size)、學習率、損失函數類型 (`mse` 或 `mae`) 以及模型儲存路徑。
    *   根據使用者選擇的模型名稱和損失函數類型載入並編譯模型。
    *   使用自訂的 `TQDMProgressBar` Keras 回呼函式顯示每個 epoch 的訓練進度，並在 epoch 結束時打印損失值。
    *   訓練完成後，模型會儲存到指定的完整路徑。
    *   新增**損失曲線繪製**功能：訓練結束後會自動繪製訓練損失隨 epoch 變化的曲線圖，並將其儲存為 PNG 檔案，與模型檔案放在同一目錄下，檔名為 `[模型檔名]_loss_curve.png`。

4.  **模型評估與視覺化 (`utils.py`)**:
    *   `evaluate_model` 函數作為評估流程的核心。
    *   使用 `keras.models.load_model` 載入模型，並在 `custom_objects` 中加入了 `LeakyReLU` 和 `PReLU` 以確保包含這些自訂激活層的模型能正確載入。
    *   `load_test_image_pairs`: 載入測試用的 L 通道圖和原始彩色圖，進行與訓練時類似的預處理。
    *   `predict_ab_channels`: 使用載入的模型進行預測 ab 通道。
    *   `reconstruct_rgb_from_lab`: 將預測的 ab 通道與原始 L 通道合併，轉換回 RGB。
    *   `calculate_metrics`: 計算預測 RGB 圖與原始 RGB 圖之間的 **PSNR** 和 **SSIM**。
    *   `plot_results`:
        *   為每個測試圖片產生對比圖 (輸入L、預測RGB、原始RGB) 並標註 PSNR/SSIM。
        *   產生 PSNR 和 SSIM 的匯總長條圖。
        *   所有結果儲存到指定資料夾。
    *   評估時的 `embed_input` 也使用隨機向量，與訓練時保持一致。
    *   使用 `argparse` 接收命令列參數，包括模型路徑、測試資料夾路徑、結果儲存路徑。

5.  **主流程控制 (`main.py`)**:
    *   整合訓練和評估流程的腳本。
    *   使用 `argparse` 接收訓練、評估和流程控制的參數，包括選擇損失函數類型。
    *   可以選擇執行完整流程 (訓練後評估) 或跳過訓練直接評估已有的模型。
    *   在腳本開頭加入了 TensorFlow GPU 檢測程式碼，以確認 GPU 是否可用並設定 memory growth。
    *   模型儲存檔名模板中加入了損失函數類型，例如 `model_[model_type]_loss[loss_type]_lr[lr].h5`。

## 環境設定

1.  **克隆專案** (如果您是從版本控制系統獲取)：
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **安裝必要的 Python 套件**：
    建議使用虛擬環境 (如 `venv` 或 `conda`)。
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    ```
    主要的依賴套件包括：
    *   `tensorflow` (建議 >= 2.x，本專案開發時使用 TensorFlow 2.19.0)
    *   `Pillow` (PIL)
    *   `scikit-image`
    *   `numpy`
    *   `matplotlib`
    *   `tqdm`

    您可以透過 `requirements.txt` (如果提供) 或手動安裝：
    ```bash
    pip install tensorflow Pillow scikit-image numpy matplotlib tqdm
    ```
    **注意**: 如果您有 NVIDIA GPU 並希望使用 GPU 進行訓練，請確保已正確安裝 NVIDIA 驅動程式、CUDA Toolkit 和 cuDNN，且版本與您安裝的 TensorFlow 版本兼容。

## 使用說明

### 1. 資料準備

*   **原始彩色圖像**: 準備一個包含原始彩色漫畫圖像的資料夾 (例如 `1000img-paul/` 或 `data/color_images/`)。
*   **黑白 L 通道圖像**: 使用 `file_converter.py` 將您的彩色圖像轉換為模型所需的單通道灰階圖像。

    執行 `file_converter.py`:
    ```bash
    python file_converter.py --input_folder path/to/your/color_images --output_folder path/to/your/bw_images --size 512
    ```
    *   `--input_folder`: 包含原始彩色圖像的資料夾。
    *   `--output_folder`: 轉換後的黑白圖像將儲存於此 (預設 `bw_images_512`)。
    *   `--size`: 圖像將被 resize 到的尺寸的寬度 (預設 512，輸出為 512x512)。

### 2. 模型訓練

使用 `train.py` 腳本進行模型訓練。

**基本執行範例 (使用 `best_version` 模型和 MAE 損失)：**
```bash
python train.py \
    --bw_dir path/to/your/bw_images \
    --color_dir path/to/your/color_images \
    --model best_version \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.0001 \
    --loss_type mae \
    --save_path trained_models/model_best_version_mae_lr0p0001_epochs50.h5
```

**`train.py` 命令列參數說明:**

*   `--bw_dir` (必需): 包含由 `file_converter.py` 生成的黑白 (L 通道) 圖像的資料夾。
*   `--color_dir` (必需): 包含對應的原始彩色圖像的資料夾。
*   `--model` (可選): 要訓練的模型名稱。可選: `unet_vgg16`, `best_version`, `unet_advanced_prelu` (預設: `best_version`)。
*   `--epochs` (可選): 訓練的 epoch 數量 (預設: 50)。
*   `--batch_size` (可選): 訓練的 batch_size (預設: 16，對於 512x512 圖像，您可能需要根據 GPU 記憶體調整此值，例如 2 或 4)。
*   `--lr` (可選): 學習率 (預設: 0.0001)。
*   `--loss_type` (可選): 要使用的損失函數類型。可選: `mse`, `mae`, `l1` (預設: `mse`)。
*   `--save_path` (必需): 儲存訓練後模型的完整檔案路徑 (例如：`trained_models/my_model.h5`)。損失曲線圖將儲存在相同位置，檔名為 `[save_path]_loss_curve.png`。

訓練完成後，模型檔案和損失曲線圖將儲存到您指定的路徑。

### 3. 模型評估

使用 `utils.py` 腳本評估已訓練模型的性能 (通常是 U-Net 生成器或 GAN 的生成器部分)。

**基本執行範例:**
```bash
python utils.py \
    --model_path path/to/your/trained_generator_model.h5 \
    --l_dir path/to/your/test_bw_images \
    --color_dir path/to/your/test_color_images \
    --results_dir evaluation_results/run1
```

**`utils.py` 命令列參數說明:**

*   `--model_path` (必需): 已訓練模型的 `.h5` 檔案路徑。
*   `--l_dir` (必需): 包含測試用的黑白 (L 通道) 圖像的資料夾。
*   `--color_dir` (必需): 包含測試用的原始彩色對照圖像的資料夾。
*   `--results_dir` (可選): 儲存評估結果 (包含對比圖、PSNR/SSIM 長條圖) 的資料夾 (預設: `evaluation_results`)。

評估腳本會輸出平均 PSNR 和 SSIM，並在指定的結果資料夾中生成對比圖像和指標匯總圖。

### 4. 完整流程 (`main.py`)

使用 `main.py` 腳本可以執行從訓練到評估的完整流程，支援 U-Net 單獨訓練和 GAN 訓練模式。

**`main.py` 主要命令列參數 (部分)：**

*   `--train_mode`: 選擇訓練模式 (`unet` 或 `gan`)。
*   **基礎訓練參數**: 
    *   `--train_bw_dir`, `--train_color_dir`: 訓練圖像資料夾。
    *   `--model_type`: U-Net 模型類型 (例如 `best_version`, `unet_advanced_prelu`)，在 GAN 模式下作為生成器。
    *   `--epochs`, `--batch_size`, `--lr`, `--loss_type` (U-Net 的 L1/MAE 部分)。
    *   `--model_output_template`: 模型儲存路徑模板。
*   **GAN 特定參數** (當 `--train_mode gan`時):
    *   `--lambda_l1`: L1 損失的權重。
    *   `--lr_discriminator`: 判別器的學習率。
    *   `--lr_generator_gan`: 生成器在 GAN 對抗訓練時的學習率。
*   **評估參數**: 
    *   `--test_l_dir`, `--test_color_dir`: 測試圖像資料夾。
    *   `--eval_results_dir`: 評估結果儲存目錄。
*   **控制流程參數**: 
    *   `--skip_training`, `--skip_evaluation`.
    *   `--trained_model_path`: 若跳過訓練，指定已訓練模型的路徑。

**執行範例 (U-Net 訓練並評估)：**
```bash
python main.py --train_mode unet \
    --train_bw_dir path/to/train_bw --train_color_dir path/to/train_color \
    --model_type best_version --epochs 50 --batch_size 2 --lr 0.0001 --loss_type mae \
    --test_l_dir path/to/test_bw --test_color_dir path/to/test_color \
    --eval_results_dir evaluation_output_unet
```

**執行範例 (GAN 訓練並評估生成器)：**
```bash
python main.py --train_mode gan \
    --train_bw_dir path/to/train_bw --train_color_dir path/to/train_color \
    --model_type unet_advanced_prelu --epochs 100 --batch_size 1 --lr 0.0002 --loss_type mae \
    --lambda_l1 100 --lr_discriminator 0.0002 --lr_generator_gan 0.0002 \
    --test_l_dir path/to/test_bw --test_color_dir path/to/test_color \
    --eval_results_dir evaluation_output_gan
```

**執行範例 (僅評估已訓練的 GAN 生成器模型)：**
```bash
python main.py --skip_training --skip_evaluation=False \
    --trained_model_path trained_models/model_gan_unet_advanced_prelu_lossmae_lr0p0002_final_generator.h5 \
    --test_l_dir path/to/test_bw --test_color_dir path/to/test_color \
    --eval_results_dir evaluation_output_gan_eval_only
```

### 5. 專案結構 (概覽)

```
.
├── model.py                  # 主要模型定義 (unet_advanced_prelu, discriminator, GAN)
├── baseline.py               # 其他基準模型定義 (unet_vgg16, unet_relu_leaky)
├── predict.py                # 使用已訓練模型對圖像資料夾進行著色的腳本
├── file_converter.py         # 彩色圖像轉黑白 L 通道工具
├── train.py                  # 模型訓練腳本
├── utils.py                  # 模型評估與結果視覺化工具
├── main.py                   # 整合訓練與評估的主流程腳本
├── trained_models/           # (建議) 儲存訓練好的模型 (.h5) 和損失曲線圖 (.png)
├── data/                     # (建議) 存放原始資料集
│   ├── color_images/         #   例如：原始彩色圖像
│   └── bw_images_512/        #   例如：轉換後的 512x512 黑白圖像
├── evaluation_results/       # (建議) 儲存模型評估結果的資料夾
└── README.md                 # 本文件
```

## 未來可能的改進方向

*   **引入驗證集 (Validation Set)**: 在訓練過程中監控驗證損失，以更好地判斷模型泛化能力和實現早停 (Early Stopping)。
*   **更複雜的損失函數**: 探索感知損失 (Perceptual Loss) 或 GAN 損失，以期生成更逼真、細節更豐富的顏色。
*   **資料增強 (Data Augmentation)**: 應用更豐富的資料增強技術，特別是顏色相關的增強，以提升模型的魯棒性和顏色多樣性。
*   **注意力機制 (Attention Mechanisms)**: 在 U-Net 中加入注意力模塊，使模型能更關注重要的圖像區域。
*   **超參數調優**: 系統地對學習率、batch size、網絡結構等超參數進行調優。
*   **使用者介面 (UI)**: 開發一個簡單的圖形化使用者介面，方便使用者上傳黑白圖片並獲得上色結果。
