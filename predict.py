#!/usr/bin/env python
# predict.py
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import argparse
import matplotlib.pyplot as plt
import logging
import warnings
import sys
import io
import cv2 as cv

# 徹底抑制TensorFlow日誌
# 設置環境變量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 關閉oneDNN自定義操作警告

# 禁用所有TensorFlow日誌
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if hasattr(tf, 'get_logger'):
    tf.get_logger().setLevel('ERROR')

# 禁用Python警告
warnings.filterwarnings('ignore')

# 替換stderr以捕獲TensorFlow C++日誌
class NullWriter(io.IOBase):
    def write(self, *args, **kwargs):
        pass
    
    def flush(self, *args, **kwargs):
        pass

# 保存原始stderr
original_stderr = sys.stderr
# 在某些操作前暫時替換stderr
sys.stderr = NullWriter()

def load_and_preprocess_image(image_path, img_size=512):
    """載入並預處理圖像"""
    img = Image.open(image_path)
    
    # 調整大小
    img = img.resize((img_size, img_size))
    
    # 轉為numpy數組並標準化
    img = np.array(img) / 127.5 - 1.0
    
    # 確保為灰階圖像 (如果是彩色的，轉為灰階)
    if len(img.shape) == 3 and img.shape[2] == 3:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        img = gray
    
    # 轉為3維
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    
    # 增加批次維度
    img = np.expand_dims(img, axis=0)
    
    return img, image_path

def get_ssim(image1_path, image2_path):
    """計算兩張圖片的SSIM值"""
    try:
        # 讀取並轉換圖片
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        # 確保兩張圖片大小相同
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # 轉換為numpy數組
        img1 = np.array(img1, dtype=np.float64)
        img2 = np.array(img2, dtype=np.float64)
        
        # 計算均值
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # 計算方差和協方差
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM常數
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # 計算SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = numerator / denominator
        
        return ssim
        
    except Exception as e:
        print(f"錯誤：計算SSIM時發生錯誤 - {str(e)}")
        return None

def save_result(input_img, generated_img, output_path, compare_ssim=False):
    """保存結果到輸出路徑，分別保存灰階、生成的彩色圖片和原始彩色圖片"""
    # 處理輸入圖像和生成圖像
    if tf.is_tensor(input_img):
        input_img = input_img.numpy()
    if tf.is_tensor(generated_img):
        generated_img = generated_img.numpy()
    
    # 轉換為uint8格式
    input_img = ((input_img[0] + 1) * 127.5).astype(np.uint8)
    generated_img = ((generated_img[0] + 1) * 127.5).astype(np.uint8)
    
    # 獲取基本檔名和目錄
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    if base_name.endswith('_colored'):
        base_name = base_name[:-8]  # 移除_colored後綴
    
    # 保存灰階圖片
    gray_path = os.path.join(base_dir, f"{base_name}_gray.jpg")
    cv.imwrite(gray_path, input_img)
    
    # 保存生成的彩色圖片
    colored_path = os.path.join(base_dir, f"{base_name}_colored.jpg")
    # OpenCV使用BGR格式，需要轉換
    generated_img_bgr = cv.cvtColor(generated_img, cv.COLOR_RGB2BGR)
    cv.imwrite(colored_path, generated_img_bgr)
    
    # 讀取並處理原始彩色圖片
    # 嘗試多個可能的位置和格式
    possible_paths = [
        os.path.join('data', 'test', 'color', f"{base_name}.jpg"),
        os.path.join('data', 'test', 'color', f"{base_name}.png"),
        os.path.join('data', 'train', 'color', f"{base_name}.jpg"),
        os.path.join('data', 'train', 'color', f"{base_name}.png"),
        os.path.join('data', 'color', f"{base_name}.jpg"),
        os.path.join('data', 'color', f"{base_name}.png")
    ]
    
    original_color_path = None
    for path in possible_paths:
        if os.path.exists(path):
            original_color_path = path
            break
    
    if original_color_path:
        original_img = cv.imread(original_color_path)
        if original_img is not None:
            # 調整大小到512x512
            original_img = cv.resize(original_img, (512, 512))
            # 保存調整大小後的原始彩色圖片
            target_original_path = os.path.join(base_dir, f"{base_name}_original.jpg")
            cv.imwrite(target_original_path, original_img)
            print(f"已保存原始彩色圖片: {target_original_path}")
            
            # 如果需要進行相似度比較
            if compare_ssim and os.path.exists(colored_path) and os.path.exists(target_original_path):
                ssim = get_ssim(colored_path, target_original_path)
                if ssim is not None:
                    print(f"SSIM相似度: {ssim:.4f}")
        else:
            print(f"警告：無法讀取原始彩色圖片: {original_color_path}")
    else:
        print("警告：找不到對應的原始彩色圖片")
    
    print(f"已保存灰階圖片: {gray_path}")
    print(f"已保存生成的彩色圖片: {colored_path}")

def main():
    # 恢復原始stderr以顯示用戶信息
    sys.stderr = original_stderr
    
    # 設置中文字體
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 設置中文字體
        plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
        print("已設置中文字體: Microsoft YaHei")
    except:
        print("無法設置中文字體，將使用系統默認字體")
    
    # 命令行參數
    parser = argparse.ArgumentParser(description='使用pix2pix模型為漫畫上色')
    parser.add_argument('--input', type=str, required=True, help='輸入灰階圖像或目錄的路徑')
    parser.add_argument('--output_dir', type=str, default='predictions', help='輸出目錄路徑')
    parser.add_argument('--model_path', type=str, default='trained_model/generator_model.keras', 
                        help='已訓練模型的路徑 (.keras或.h5文件)')
    parser.add_argument('--img_size', type=int, default=512, help='輸入圖片大小')
    parser.add_argument('--compare_ssim', action='store_true', 
                        help='是否在生成後立即計算與原始圖片的SSIM相似度')
    
    args = parser.parse_args()
    
    # 檢查模型路徑
    if not os.path.exists(args.model_path):
        print(f"錯誤：找不到指定的模型文件 '{args.model_path}'")
        print(f"請確保模型已經被正確訓練和保存。")
        return
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 禁用stderr以抑制TensorFlow消息
    sys.stderr = NullWriter()
    
    # 載入模型
    try:
        model = tf.keras.models.load_model(args.model_path)
        # 恢復stderr以顯示用戶信息
        sys.stderr = original_stderr
        print(f"已成功載入模型: {args.model_path}")
    except Exception as e:
        # 恢復stderr以顯示錯誤
        sys.stderr = original_stderr
        print(f"載入模型時出錯: {str(e)}")
        print(f"請確保模型格式正確。")
        return
    
    # 處理輸入路徑
    if os.path.isdir(args.input):
        # 如果輸入是一個目錄，處理目錄中的所有圖像
        image_paths = glob.glob(os.path.join(args.input, "*.jpg")) + glob.glob(os.path.join(args.input, "*.png"))
        if not image_paths:
            print(f"在 '{args.input}' 目錄中未找到任何圖像文件。")
            return
    elif os.path.isfile(args.input) and args.input.endswith((".jpg", ".jpeg", ".png")):
        # 如果輸入是一個文件，直接處理該文件
        image_paths = [args.input]
    else:
        print(f"無效的輸入路徑: '{args.input}'")
        return
    
    # 處理圖像
    for i, img_path in enumerate(image_paths):
        print(f"處理圖像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # 禁用stderr以抑制TensorFlow消息
        sys.stderr = NullWriter()
        
        # 載入並預處理圖像
        input_img, _ = load_and_preprocess_image(img_path, args.img_size)
        
        # 生成彩色圖像
        generated_img = model(input_img, training=False)
        
        # 恢復stderr以顯示用戶信息
        sys.stderr = original_stderr
        
        # 保存結果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_colored.jpg")
        save_result(input_img, generated_img, output_path, args.compare_ssim)
    
    print(f"\n所有圖像處理完成。結果保存在 '{args.output_dir}' 目錄中。")

if __name__ == "__main__":
    main() 