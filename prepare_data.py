#!/usr/bin/env python
# prepare_data.py
import os
import glob
import shutil
import random
from PIL import Image
import argparse
from tqdm import tqdm

def setup_directories(data_root="data"):
    """建立必要的目錄結構"""
    directories = [
        os.path.join(data_root, "train", "gray"),
        os.path.join(data_root, "train", "color"),
        os.path.join(data_root, "test", "gray"),
        os.path.join(data_root, "test", "color"),
        os.path.join(data_root, "color")  # 原始彩色圖存放處
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("目錄結構已建立:")
    for directory in directories:
        print(f"- {directory}")

def convert_to_grayscale(img_path, output_path):
    """將彩色圖像轉換為灰階圖像（保持3通道）"""
    try:
        # 打開彩色圖片
        color_img = Image.open(img_path)
        
        # 轉換為灰階
        gray_img = color_img.convert('L')
        
        # 將單通道灰階圖擴展為3通道
        gray_img_3channel = Image.merge('RGB', (gray_img, gray_img, gray_img))
        
        # 保存灰階圖片
        gray_img_3channel.save(output_path)
        
        return True
    except Exception as e:
        print(f"處理 {img_path} 時出錯: {e}")
        return False

def split_dataset(color_dir, train_ratio=0.8, data_root="data"):
    """分割數據集為訓練集和測試集"""
    # 獲取所有彩色圖片
    color_images = glob.glob(os.path.join(color_dir, "*.jpg")) + glob.glob(os.path.join(color_dir, "*.png"))
    
    if not color_images:
        print(f"在 {color_dir} 中未找到圖片文件")
        return
    
    # 隨機打亂
    random.shuffle(color_images)
    
    # 分割為訓練集和測試集
    split_idx = int(len(color_images) * train_ratio)
    train_images = color_images[:split_idx]
    test_images = color_images[split_idx:]
    
    print(f"總圖片數: {len(color_images)}")
    print(f"訓練集: {len(train_images)} 張圖片")
    print(f"測試集: {len(test_images)} 張圖片")
    
    # 處理訓練集
    process_image_set(train_images, "train", data_root)
    
    # 處理測試集
    process_image_set(test_images, "test", data_root)

def process_image_set(image_paths, set_type, data_root):
    """處理一組圖片（訓練集或測試集）"""
    color_dir = os.path.join(data_root, set_type, "color")
    gray_dir = os.path.join(data_root, set_type, "gray")
    
    # 確保目錄存在
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(gray_dir, exist_ok=True)
    
    print(f"處理{set_type}集...")
    
    for img_path in tqdm(image_paths):
        try:
            filename = os.path.basename(img_path)
            
            # 複製彩色圖片到對應的color目錄
            color_output_path = os.path.join(color_dir, filename)
            shutil.copy2(img_path, color_output_path)
            
            # 生成灰階版本
            gray_output_path = os.path.join(gray_dir, filename)
            convert_to_grayscale(img_path, gray_output_path)
            
        except Exception as e:
            print(f"處理 {img_path} 時出錯: {e}")

def main():
    parser = argparse.ArgumentParser(description="準備漫畫圖片數據集，分割為訓練集和測試集")
    parser.add_argument("--color_dir", default="data/color", help="彩色漫畫圖片所在的目錄")
    parser.add_argument("--data_root", default="data", help="數據根目錄")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="訓練集所占比例 (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    
    args = parser.parse_args()
    
    # 設置隨機種子
    random.seed(args.seed)
    
    # 建立目錄結構
    setup_directories(args.data_root)
    
    # 分割數據集
    split_dataset(args.color_dir, args.train_ratio, args.data_root)
    
    print("數據處理完成！")

if __name__ == "__main__":
    main() 