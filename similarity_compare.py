#!/usr/bin/env python
import os
from PIL import Image
import numpy as np
import argparse

def get_ssim(image1_path, image2_path):
    """計算兩張圖片的SSIM值"""
    try:
        # 讀取並轉換圖片
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        # 確保兩張圖片大小相同
        if img1.size != img2.size:
            print(f"注意：圖片大小不同，正在將第二張圖片調整為與第一張圖片相同大小 ({img1.size})")
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
    ssim = get_ssim(args.image1, args.image2)
    
    if ssim is not None:
        print(f"\n圖片1: {args.image1}")
        print(f"圖片2: {args.image2}")
        print(f"SSIM相似度: {ssim:.4f}")
        print(f"\n相似度範圍: [-1, 1]，值越接近1表示兩張圖片越相似")

if __name__ == "__main__":
    main() 