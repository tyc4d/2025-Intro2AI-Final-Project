# pix2pix_manga_colorizer/train.py

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from dataset import load_dataset, setup_directories
from tqdm import tqdm

# 設置環境變量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = ERROR

def generate_images(model, input_image, target, save_path=None):
    """
    生成並顯示預測結果
    
    參數:
        model: 生成器模型
        input_image: 輸入灰階圖像
        target: 目標彩色圖像
        save_path: 保存圖像的路徑（如果提供）
    """
    # 在預測時使用training=False以獲得更好的結果
    prediction = model(input_image, training=False)
    display_list = [input_image[0], target[0], prediction[0]]
    title = ['Input (Gray)', 'Target (Color)', 'Predicted']

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # 將[-1, 1]範圍的圖像轉換為[0, 1]以供顯示
        plt.imshow((display_list[i] + 1) / 2)
        plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

# Config
DATA_DIR = "data"
OUTPUT_DIR = "output"
IMG_SIZE = 512
EPOCHS = 15
BATCH_SIZE = 1
LAMBDA = 100  # L1損失的權重

# 創建輸出目錄
os.makedirs(OUTPUT_DIR, exist_ok=True)
setup_directories()  # 創建數據目錄結構

# Load dataset
print("正在載入數據集...")
train_dataset = load_dataset(is_train=True, batch_size=BATCH_SIZE)
test_dataset = load_dataset(is_train=False, batch_size=BATCH_SIZE)

# Models
generator = Generator()
discriminator = Discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Loss function
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, generated_output):
    """判別器損失函數"""
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    generated_loss = loss_object(tf.zeros_like(generated_output), generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    """生成器損失函數"""
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # L1損失
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 生成器生成結果
        gen_output = generator(input_image, training=True)

        # 判別器對真實圖像和生成圖像的判斷
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # 計算損失
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # 計算梯度
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 應用梯度
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs, test_dataset):
    # 獲取數據集大小
    dataset_size = sum(1 for _ in dataset)
    
    # 獲取測試樣本
    test_samples = []
    for test_input, test_target in test_dataset.take(1):
        test_samples.append((test_input, test_target))
    
    # 在訓練開始時保存一個初始示例
    if test_samples:
        prediction_filename = os.path.join(OUTPUT_DIR, f"initial_sample.jpg")
        generate_images(generator, test_samples[0][0], test_samples[0][1], prediction_filename)
    
    # 計算總步數
    total_steps = epochs * dataset_size
    
    # 創建一個總體進度條
    with tqdm(total=total_steps, desc="訓練進度", unit="step") as pbar:
        for epoch in range(epochs):
            # 每個epoch的損失
            step = 0
            
            # 處理數據集中的每個批次
            for input_image, target in dataset:
                # 訓練步驟
                gen_loss, disc_loss = train_step(input_image, target)
                
                # 更新進度條
                pbar.update(1)
                pbar.set_postfix({
                    'Epoch': f"{epoch+1}/{epochs}",
                    'Gen': f"{float(gen_loss):.4f}", 
                    'Disc': f"{float(disc_loss):.4f}"
                })
                
                step += 1
            
            # 每個epoch結束時保存示例圖像
            if test_samples:
                prediction_filename = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_sample.jpg")
                generate_images(generator, test_samples[0][0], test_samples[0][1], prediction_filename)
    
    # 保存最終模型
    final_model_dir = "trained_model"
    os.makedirs(final_model_dir, exist_ok=True)
    model_path = os.path.join(final_model_dir, "generator_model.keras")
    generator.save(model_path)
    print(f"模型已保存至 {model_path}")

if __name__ == "__main__":
    print("開始訓練...")
    # 檢查測試數據是否可用
    test_data_available = True
    try:
        for _ in test_dataset.take(1):
            pass
    except:
        test_data_available = False
        print("警告: 未找到測試數據，請確保放置了灰階和彩色測試圖片")
    
    if test_data_available:
        train(train_dataset, EPOCHS, test_dataset)
    else:
        print("請先放置適當的訓練和測試數據")
