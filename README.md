# Pix2Pix 漫畫上色專案

## 專案結構

```
pix2pix-mine/
│
├── data/                      # 數據目錄
│   ├── color/                # 原始彩色圖片
│   ├── train/               # 訓練數據
│   │   ├── gray/           # 訓練用灰階圖片
│   │   └── color/          # 訓練用彩色圖片
│   └── test/                # 測試數據
│       ├── gray/           # 測試用灰階圖片
│       └── color/          # 測試用彩色圖片
│
├── model.py                  # 模型架構定義
├── dataset.py               # 數據集處理
├── prepare_data.py          # 數據預處理腳本
├── train.py                 # 訓練腳本
├── predict.py               # 預測/推理腳本
│
├── trained_model/           # 訓練好的模型
│   └── generator_model.keras
│
├── output/                  # 訓練過程輸出
└── predictions/             # 預測結果輸出
```

這是一個使用 Pix2Pix 生成對抗網絡（GAN）實現漫畫自動上色的專案。該專案可以將黑白漫畫圖片轉換為彩色版本。

## 環境需求

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Pillow
- NumPy
- Matplotlib
- tqdm

### 安裝環境

```bash
pip install tensorflow opencv-python pillow numpy matplotlib tqdm
```

## 使用流程

### 1. 數據準備
將您的原始彩色圖片放入 `data/color` 目錄：
```powershell
# 複製圖片到data/color目錄
Copy-Item "<你的圖片目錄>\*" -Destination "data\color" -Recurse
```

### 2. 數據預處理
運行數據預處理腳本，將數據分割為訓練集和測試集：
```bash
python prepare_data.py
```
此步驟會：
- 自動創建必要的目錄結構
- 將圖片分割為訓練集和測試集
- 生成對應的灰階版本
- 將數據整理到對應的訓練和測試目錄中

### 3. 訓練模型
```bash
python train.py
```
訓練過程中：
- 每個 epoch 結束後會保存示例圖片到 `output` 目錄
- 訓練完成後，模型會被保存到 `trained_model/generator_model.keras`

### 4. 生成預測結果和相似度數據
```bash
python predict.py --input <輸入路徑> --compare_ssim
```
此命令會：
- 生成彩色預測結果
- 計算與原始圖片的相似度
- 將結果保存在 `predictions` 目錄

### 5. 保存結果（可選）
如果需要保存結果到特定目錄：
```powershell
# 移動預測結果到指定目錄
Move-Item -Path "predictions\*" -Destination "<目標路徑>" -Force
```

### 6. 清理數據（準備下一批測試）
清空所有數據目錄，準備處理下一批圖片：
```powershell
Remove-Item "data\color\*" -Force -Recurse
Remove-Item "data\test\color\*" -Force -Recurse
Remove-Item "data\test\gray\*" -Force -Recurse
Remove-Item "data\train\color\*" -Force -Recurse
Remove-Item "data\train\gray\*" -Force -Recurse
```

## 批次處理流程

1. 放入數據：將圖片複製到 `data/color`
2. 預處理：運行 `prepare_data.py`
3. 訓練：運行 `train.py`
4. 預測和評估：運行 `predict.py`
5. 保存結果：移動 `predictions` 內容到目標目錄
6. 清理數據：清空所有數據目錄
7. 重複以上步驟處理下一批數據

