#!/usr/bin/env python
import os
from PIL import Image
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim

def get_ssim(image1_path, image2_path):
    """計算兩張圖片的SSIM值"""
    try:
        # 讀取並轉換圖片
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        # 強制調整兩張圖片為 256x256，使用 LANCZOS 重採樣
        target_size = (256, 256)
        img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
        img2 = img2.resize(target_size, Image.Resampling.LANCZOS)
        print(f"已將兩張圖片調整為 {target_size} 大小（使用 LANCZOS 重採樣）")
        
        # 轉換為numpy數組
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        # 使用 skimage 的 ssim 函數計算相似度
        ssim_value = ssim(img1, img2, channel_axis=2, data_range=255)
        
        return ssim_value
        
    except Exception as e:
        print(f"錯誤：計算SSIM時發生錯誤 - {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='計算兩張圖片之間的SSIM相似度')
    parser.add_argument('image1', type=str,
                        help='第一張圖片的路徑（可以是相對或絕對路徑）')
    parser.add_argument('image2', type=str,
                        help='第二張圖片的路徑（可以是相對或絕對路徑）')
    
    args = parser.parse_args()
    
    # 檢查文件是否存在
    if not os.path.exists(args.image1):
        print(f"錯誤：找不到第一張圖片 '{args.image1}'")
        return
    
    if not os.path.exists(args.image2):
        print(f"錯誤：找不到第二張圖片 '{args.image2}'")
        return
    
    # 計算SSIM
    ssim_value = get_ssim(args.image1, args.image2)
    
    if ssim_value is not None:
        print(f"\n圖片1: {args.image1}")
        print(f"圖片2: {args.image2}")
        print(f"SSIM相似度: {ssim_value:.4f}")
        print(f"\n相似度範圍: [0, 1]，值越接近1表示兩張圖片越相似")

if __name__ == "__main__":
    main() 