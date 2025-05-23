# dataset.py
import tensorflow as tf
import os
from PIL import Image
import glob
import numpy as np

IMG_SIZE = 512  # 輸入圖片會被縮放到這個尺寸
AUTOTUNE = tf.data.experimental.AUTOTUNE

# 更新目錄結構
DATA_ROOT = "data"
TRAIN_GRAY_DIR = os.path.join(DATA_ROOT, "train", "gray")
TRAIN_COLOR_DIR = os.path.join(DATA_ROOT, "train", "color")
TEST_GRAY_DIR = os.path.join(DATA_ROOT, "test", "gray")
TEST_COLOR_DIR = os.path.join(DATA_ROOT, "test", "color")

def normalize(input_image, target_image):
    input_image = (input_image / 255.0) * 2 - 1
    target_image = (target_image / 255.0) * 2 - 1
    return input_image, target_image

def resize(input_image, target_image, height, width):
    input_image = tf.image.resize(input_image, [height, width])
    target_image = tf.image.resize(target_image, [height, width])
    return input_image, target_image

def random_jitter(input_image, target_image):
    # Resize + Random crop
    input_image, target_image = resize(input_image, target_image, 572, 572)
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_SIZE, IMG_SIZE, 3])
    input_image, target_image = cropped_image[0], cropped_image[1]

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image

def process_path(file_path_inp, file_path_tar, is_train=True):
    """處理文件路徑並加載圖像"""
    # 讀取文件
    input_image = tf.io.read_file(file_path_inp)
    target_image = tf.io.read_file(file_path_tar)
    
    # 解碼圖像
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    target_image = tf.image.decode_jpeg(target_image, channels=3)
    
    # 轉換類型
    input_image = tf.cast(input_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)
    
    # 調整大小
    input_image, target_image = resize(input_image, target_image, IMG_SIZE, IMG_SIZE)
    
    # 應用數據增強
    if is_train:
        input_image, target_image = random_jitter(input_image, target_image)
    
    # 標準化
    input_image, target_image = normalize(input_image, target_image)
    
    return input_image, target_image

def load_dataset(is_train=True, batch_size=1):
    """
    載入訓練或測試數據集
    
    參數:
        is_train: 是否載入訓練集
        batch_size: 批次大小
    """
    # 確保目錄存在
    if is_train:
        os.makedirs(TRAIN_GRAY_DIR, exist_ok=True)
        os.makedirs(TRAIN_COLOR_DIR, exist_ok=True)
        input_dir = TRAIN_GRAY_DIR
        target_dir = TRAIN_COLOR_DIR
    else:
        os.makedirs(TEST_GRAY_DIR, exist_ok=True)
        os.makedirs(TEST_COLOR_DIR, exist_ok=True)
        input_dir = TEST_GRAY_DIR
        target_dir = TEST_COLOR_DIR
    
    # 獲取所有圖片文件路徑
    input_paths = []
    target_paths = []
    
    for ext in ['*.jpg', '*.png']:
        files = glob.glob(os.path.join(input_dir, ext))
        for input_path in files:
            filename = os.path.basename(input_path)
            target_path = os.path.join(target_dir, filename)
            # 確認目標文件存在
            if os.path.exists(target_path):
                input_paths.append(input_path)
                target_paths.append(target_path)
    
    if len(input_paths) == 0:
        raise ValueError(f"未在 {input_dir} 找到任何匹配的圖片文件")
    
    print(f"找到 {len(input_paths)} 對{'訓練' if is_train else '測試'}圖片")
    
    # 創建數據集
    input_paths_ds = tf.data.Dataset.from_tensor_slices(input_paths)
    target_paths_ds = tf.data.Dataset.from_tensor_slices(target_paths)
    path_dataset = tf.data.Dataset.zip((input_paths_ds, target_paths_ds))
    
    # 處理圖像
    image_dataset = path_dataset.map(
        lambda inp, tar: process_path(inp, tar, is_train),
        num_parallel_calls=AUTOTUNE
    )
    
    # 配置數據集
    if is_train:
        image_dataset = image_dataset.shuffle(1000)
    
    image_dataset = image_dataset.batch(batch_size).prefetch(AUTOTUNE)
    
    return image_dataset

# 用於確保目錄結構正確的函數
def setup_directories():
    """創建必要的目錄結構"""
    directories = [TRAIN_GRAY_DIR, TRAIN_COLOR_DIR, TEST_GRAY_DIR, TEST_COLOR_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("目錄結構已創建:")
    print(f"訓練集灰階圖片: {TRAIN_GRAY_DIR}")
    print(f"訓練集彩色圖片: {TRAIN_COLOR_DIR}")
    print(f"測試集灰階圖片: {TEST_GRAY_DIR}")
    print(f"測試集彩色圖片: {TEST_COLOR_DIR}")
