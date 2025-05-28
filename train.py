import argparse
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb # skimage 用於色彩空間轉換
from tqdm import tqdm
from keras.callbacks import Callback
import tensorflow as tf # For Adam and losses
from keras.losses import MeanAbsoluteError, BinaryCrossentropy
from keras.optimizers import Adam
import baseline # 從 baseline.py 匯入模型
from baseline import build_discriminator, define_gan # GAN components
import multiprocessing # 新增
from functools import partial # 可能用於簡化參數傳遞
import matplotlib.pyplot as plt # 新增 matplotlib 匯入

# TQDM Callback for Keras
class TQDMProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epochs = self.params['epochs']
        self.pbar = tqdm(total=self.params['steps'], desc=f"Epoch {epoch + 1}/{self.epochs}", unit="step")

    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)
        # 你可以在這裡添加 batch 級別的 metrics 更新到 tqdm 描述中
        # self.pbar.set_postfix(loss=f"{logs['loss']:.4f}") 

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        # 在 epoch 結束時打印 loss
        loss = logs.get('loss')
        # Removed accuracy print as CategoricalAccuracy is not used for this regression task
        print(f"Epoch {epoch + 1}/{self.epochs} - loss: {loss:.4f}")

# Custom TQDM Callback for GAN training displaying multiple losses
class TQDMGANProgressBar:
    def __init__(self, total_batches, epochs):
        self.total_batches = total_batches
        self.epochs = epochs
        self.pbar = None

    def on_epoch_begin(self, epoch):
        self.pbar = tqdm(total=self.total_batches, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch")

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.pbar.update(1)
        self.pbar.set_postfix(d_loss=f"{logs.get('d_loss', 0):.4f}", g_adv=f"{logs.get('g_adv', 0):.4f}", g_l1=f"{logs.get('g_l1', 0):.4f}", g_total=f"{logs.get('g_total',0):.4f}")

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.close()
        if logs is None:
            logs = {}
        print(f"Epoch {epoch + 1}/{self.epochs} - D Loss: {logs.get('d_loss', 0):.4f}, G Adv Loss: {logs.get('g_adv', 0):.4f}, G L1 Loss: {logs.get('g_l1', 0):.4f}, G Total: {logs.get('g_total',0):.4f}")


def _process_single_image_pair(bw_fname, bw_image_folder, color_image_folder, color_filenames_set, target_size):
    """
    工作函數：處理單個黑白圖片及其對應的彩色圖片。
    由 multiprocessing.Pool 中的進程呼叫。
    """
    try:
        base_name, _ = os.path.splitext(bw_fname)
        if not base_name.endswith("_bw"):
            # print(f"跳過無法識別的黑白檔名 (worker): {bw_fname}") # 在大量並行時，打印過多會影響性能和可讀性
            return None
        
        original_base_name = base_name[:-3]
        
        potential_color_fnames = [f"{original_base_name}{ext}" for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
        color_fname_found = None
        for potential_cfname in potential_color_fnames:
            if potential_cfname in color_filenames_set: # 使用集合進行快速查找
                color_fname_found = potential_cfname
                break
        
        if not color_fname_found:
            # print(f"找不到 {bw_fname} 對應的彩色圖片 (worker, 嘗試基底: {original_base_name})")
            return None

        bw_img_path = os.path.join(bw_image_folder, bw_fname)
        bw_img = Image.open(bw_img_path).convert('L').resize(target_size, Image.Resampling.LANCZOS)
        l_channel_np = np.array(bw_img, dtype=float) / 255.0
        
        color_img_path = os.path.join(color_image_folder, color_fname_found)
        color_img = Image.open(color_img_path).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        color_img_lab = rgb2lab(np.array(color_img))
        
        # For GAN input to discriminator, L channel might be better in [-1, 1] if ab are [-1, 1]
        # Or, ensure discriminator can handle L in [0,1] and ab in [-1,1].
        # For now, generator input L is [0,1]. Discriminator will receive this L.
        l_channel_for_discriminator_input = l_channel_np # Keeping L in [0,1] for now for disc input
        # If L needs to be in [-1, 1] for discriminator, uncomment below:
        # l_channel_for_discriminator_input = (l_channel_np * 2.0) - 1.0 

        ab_channels_np = color_img_lab[:, :, 1:] / 128.0
        
        # embed_input 每個圖像都重新生成隨機向量
        embed_input_np = np.random.randn(1000)

        # Return L for generator, L for discriminator input, embed, and ab for generator target
        return l_channel_np.reshape(target_size[0], target_size[1], 1), \
               l_channel_for_discriminator_input.reshape(target_size[0], target_size[1], 1), \
               embed_input_np, \
               ab_channels_np

    except FileNotFoundError:
        # print(f"找不到檔案 (worker)：{bw_fname} 或其對應的彩色圖片。")
        return None
    except Exception as e:
        # print(f"處理檔案 {bw_fname} 時發生錯誤 (worker): {e}")
        return None

def load_and_preprocess_data(bw_image_folder, color_image_folder, target_size=(512, 512), for_gan=False):
    """
    載入黑白圖片 (L 通道) 和對應的彩色圖片 (ab 通道作為目標)。
    圖片會被 resize 到 target_size。
    embed_input 將使用隨機向量。
    此版本使用 multiprocessing 並行處理圖片。
    """
    X_l_gen_list = []       # L channel for generator input [0,1]
    X_l_disc_list = []      # L channel for discriminator input (potentially [-1,1] or [0,1])
    X_embed_list = []
    Y_ab_list = []          # ab channels for generator target/loss [-1,1]
    
    bw_filenames = sorted(os.listdir(bw_image_folder))
    color_filenames_set = set(sorted(os.listdir(color_image_folder))) # 使用集合以加速查找

    if not bw_filenames:
        raise ValueError("黑白圖片資料夾為空或不存在。")
    if not color_filenames_set:
        print("警告：彩色圖片資料夾為空或不存在。")

    num_cpu = os.cpu_count()
    print(f"使用 {num_cpu} 個 CPU 核心並行載入與預處理資料...")

    # 準備 partial function，固定部分參數
    # worker_fn = partial(_process_single_image_pair, 
    #                     bw_image_folder=bw_image_folder, 
    #                     color_image_folder=color_image_folder, 
    #                     color_filenames_set=color_filenames_set, 
    #                     target_size=target_size)
    
    # 或者直接準備參數列表
    tasks = [(bw_fname, bw_image_folder, color_image_folder, color_filenames_set, target_size) for bw_fname in bw_filenames]

    processed_files = 0
    
    with multiprocessing.Pool(processes=num_cpu) as pool:
        # 使用 imap_unordered 以便與 tqdm 更好地整合，並在任務完成時立即處理結果
        # results = []
        # for result in tqdm(pool.imap_unordered(worker_fn, bw_filenames), total=len(bw_filenames), desc="並行載入與預處理圖像"):
        #     if result is not None:
        #         results.append(result)
        
        # 或者使用 map，如果不需要那麼細緻的進度 (tqdm 會在 map 開始前顯示總數，並在結束後完成)
        # results = pool.map(worker_fn, bw_filenames)
        
        # 使用 imap_unordered 搭配參數元組列表
        results_iterator = pool.imap_unordered(_process_single_image_pair_unpack, tasks)
        
        for result in tqdm(results_iterator, total=len(tasks), desc="並行載入與預處理圖像"):
            if result is not None:
                l_gen_data, l_disc_data, embed_data, ab_data = result
                X_l_gen_list.append(l_gen_data)
                X_l_disc_list.append(l_disc_data)
                X_embed_list.append(embed_data)
                Y_ab_list.append(ab_data)
                processed_files += 1
    
    print(f"總共成功處理 {processed_files} 張圖片。")
    if processed_files == 0:
        raise ValueError("沒有成功載入任何圖片，請檢查資料夾路徑和檔案格式或工作函數邏輯。")

    if for_gan:
        # For GAN, we return L for generator, L for discriminator, embed, and real ab
        return [np.array(X_l_gen_list), np.array(X_embed_list)], np.array(Y_ab_list), np.array(X_l_disc_list)
    else:
        # For standard U-Net, just L for gen, embed, and real ab
        return [np.array(X_l_gen_list), np.array(X_embed_list)], np.array(Y_ab_list)

# 解包參數的輔助函數，因為 pool.imap/map 只接受單個參數的函數
def _process_single_image_pair_unpack(args_tuple):
    return _process_single_image_pair(*args_tuple)

def train_model(bw_image_dir, color_image_dir, model_name, epochs, batch_size, learning_rate, save_path, loss_type='mse'):
    print(f"開始載入資料...")
    # 假設 file_converter.py 將彩色圖片轉換為灰階並儲存在 bw_image_dir
    # color_image_dir 應該是原始彩色圖片的資料夾
    X, Y = load_and_preprocess_data(bw_image_dir, color_image_dir, for_gan=False)

    print(f"載入資料完成。輸入 L 通道形狀: {X[0].shape}, Embed 輸入形狀: {X[1].shape}, 輸出 ab 通道形狀: {Y.shape}")

    if X[0].shape[0] == 0:
        print("錯誤：沒有成功載入任何訓練資料。請檢查您的圖片資料夾和檔名。")
        return

    print(f"選擇模型: {model_name}, 使用損失函數: {loss_type.upper()}")
    if model_name == 'unet_vgg16':
        model = baseline.unet_vgg16(learning_rate=learning_rate, loss_function_name=loss_type)
    elif model_name == 'unet_relu_leaky':
        model = baseline.unet_relu_leaky(learning_rate=learning_rate, loss_function_name=loss_type)
    elif model_name == 'unet_advanced_prelu':
        model = baseline.unet_advanced_prelu(learning_rate=learning_rate, loss_function_name=loss_type)
    else:
        raise ValueError("未知的模型名稱。請選擇 'unet_vgg16', 'unet_relu_leaky' 或 'unet_advanced_prelu'。")

    model.summary()

    print("開始訓練模型...")
    # 使用 TQDM Callback
    tqdm_callback = TQDMProgressBar()
    history_callback = model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[tqdm_callback]) # verbose=0 因為我們有自訂的 callback

    print("訓練完成。")
    
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"(train.py) 已建立模型儲存資料夾: {output_dir}")
        
    model.save(save_path)
    print(f"模型已儲存至 {save_path}")

    # --- 繪製並儲存損失曲線 ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history_callback.history['loss'], label='Train Loss')
        # 如果將來有驗證集，可以取消註解以下行:
        # if 'val_loss' in history_callback.history:
        #     plt.plot(history_callback.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss Curve ({model_name}, {loss_type.upper()})')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        
        # 儲存損失曲線圖
        base_save_path, ext = os.path.splitext(save_path)
        loss_curve_save_path = f"{base_save_path}_loss_curve.png"
        plt.savefig(loss_curve_save_path)
        plt.close() # 關閉圖形以釋放資源
        print(f"損失曲線圖已儲存至: {loss_curve_save_path}")
    except Exception as e:
        print(f"繪製損失曲線時發生錯誤: {e}")
    # --- 損失曲線結束 ---

def get_generator_model(model_name, learning_rate, loss_function_name):
    if model_name == 'unet_vgg16':
        g_model = baseline.unet_vgg16(learning_rate=learning_rate, loss_function_name=loss_function_name)
    elif model_name == 'unet_relu_leaky':
        g_model = baseline.unet_relu_leaky(learning_rate=learning_rate, loss_function_name=loss_function_name)
    elif model_name == 'unet_advanced_prelu':
        g_model = baseline.unet_advanced_prelu(learning_rate=learning_rate, loss_function_name=loss_function_name)
    else:
        raise ValueError(f"Unknown generator model name: {model_name}")
    return g_model

def train_gan(
    bw_image_dir, color_image_dir, generator_model_name, epochs, batch_size, 
    g_optimizer_lr_recon, # Learning rate for U-Net like part when compiled standalone (might not be used if GAN model handles all G updates)
    d_optimizer_lr, 
    gan_optimizer_lr, 
    lambda_l1, 
    save_path, 
    loss_type_g_recon='mae' # MAE (L1) is typical for image-to-image GANs
):
    print(f"Starting data loading for GAN training...")
    # X_gen_input = [L_for_gen, embed_input], Y_ab_real = real_ab_channels, X_l_disc_input = L_for_discriminator
    [X_L_gen, X_embed], Y_ab_real, X_L_disc = load_and_preprocess_data(bw_image_dir, color_image_dir, for_gan=True)
    
    print(f"GAN Data: Gen L shape: {X_L_gen.shape}, Embed shape: {X_embed.shape}, Real AB shape: {Y_ab_real.shape}, Disc L shape: {X_L_disc.shape}")
    if X_L_gen.shape[0] == 0: print("Error: No training data loaded for GAN."); return

    dataset_size = X_L_gen.shape[0]
    n_batches = dataset_size // batch_size
    image_shape_l = X_L_gen.shape[1:] # (512, 512, 1)
    image_shape_ab = Y_ab_real.shape[1:] # (512, 512, 2)
    image_shape_lab_disc = (image_shape_l[0], image_shape_l[1], image_shape_l[2] + image_shape_ab[2]) # (512,512,3)

    # --- Build and Compile Models ---
    # 1. Generator (U-Net)
    # The generator isn't compiled with its own optimizer if its weights are only updated via the GAN model.
    # However, it's useful to have it defined as a Keras model for predictions and saving.
    g_model = get_generator_model(generator_model_name, learning_rate=g_optimizer_lr_recon, loss_function_name=loss_type_g_recon)
    print("--- Generator Summary ---")
    g_model.summary()

    # 2. Discriminator (PatchGAN)
    d_model = build_discriminator(input_shape=image_shape_lab_disc) 
    d_optimizer = Adam(learning_rate=d_optimizer_lr, beta_1=0.5, beta_2=0.999)
    d_model.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
    print("--- Discriminator Summary (compiled) ---")
    d_model.summary()
    # Determine discriminator patch shape dynamically from its output
    # Create a dummy input of the correct shape for the discriminator
    dummy_disc_input = np.zeros((1,) + image_shape_lab_disc)
    patch_output_shape = d_model.predict(dummy_disc_input).shape[1:] # e.g. (64, 64, 1)
    print(f"Discriminator patch output shape: {patch_output_shape}")


    # 3. Combined GAN model (Generator + Discriminator)
    # Discriminator weights are frozen in this model (handled in define_gan)
    gan_model = define_gan(g_model, d_model, image_shape_l=image_shape_l, embed_dim=X_embed.shape[1])
    gan_optimizer = Adam(learning_rate=gan_optimizer_lr, beta_1=0.5, beta_2=0.999)
    # Compile with two losses: one for adversarial (binary_crossentropy on D's output), one for L1 (mae on G's output)
    gan_model.compile(loss=['binary_crossentropy', 'mae'], 
                      loss_weights=[1, lambda_l1], 
                      optimizer=gan_optimizer)
    print("--- Combined GAN Model Summary (compiled) ---")
    gan_model.summary()

    # --- Training Loop ---
    print(f"\nStarting GAN training for {epochs} epochs, {n_batches} batches per epoch...")
    d_losses, g_adv_losses, g_l1_losses, g_total_losses = [], [], [], []
    
    # Labels for discriminator training
    real_labels = np.ones((batch_size,) + patch_output_shape)
    fake_labels = np.zeros((batch_size,) + patch_output_shape)

    gan_progress_bar = TQDMGANProgressBar(total_batches=n_batches, epochs=epochs)

    for epoch in range(epochs):
        gan_progress_bar.on_epoch_begin(epoch)
        epoch_d_loss_acc, epoch_g_adv_loss_acc, epoch_g_l1_loss_acc, epoch_g_total_loss_acc = 0, 0, 0, 0

        for batch_i in range(n_batches):
            # 1. Select a random batch of real images
            idx = np.random.randint(0, dataset_size, batch_size)
            batch_L_gen = X_L_gen[idx]
            batch_embed = X_embed[idx]
            batch_ab_real = Y_ab_real[idx]
            batch_L_disc = X_L_disc[idx] # L channel for discriminator input

            # 2. Generate a batch of fake images (ab channels)
            batch_ab_fake = g_model.predict([batch_L_gen, batch_embed], verbose=0)

            # 3. Prepare inputs for discriminator
            # Discriminator receives L channel (potentially normalized differently) + ab channels
            # Assuming batch_L_disc is the correctly scaled L for the discriminator
            disc_input_real = np.concatenate([batch_L_disc, batch_ab_real], axis=-1)
            disc_input_fake = np.concatenate([batch_L_disc, batch_ab_fake], axis=-1)

            # 4. Train Discriminator
            # Train on real images
            d_loss_real, d_acc_real = d_model.train_on_batch(disc_input_real, real_labels)
            # Train on fake images
            d_loss_fake, d_acc_fake = d_model.train_on_batch(disc_input_fake, fake_labels)
            # Average discriminator loss
            d_loss_batch = 0.5 * np.add(d_loss_real, d_loss_fake)
            epoch_d_loss_acc += d_loss_batch

            # 5. Train Generator (via GAN model)
            # Generator tries to make discriminator output 'real' (all ones for patch)
            # Targets for GAN model: [target_for_discriminator_output, target_for_generator_L1_output]
            gan_targets = [real_labels, batch_ab_real] # Target for D output is 'real', target for G output is real_ab
            g_loss_batch_combined, g_loss_batch_adv, g_loss_batch_l1 = gan_model.train_on_batch(
                [batch_L_gen, batch_embed], gan_targets
            )
            epoch_g_adv_loss_acc += g_loss_batch_adv
            epoch_g_l1_loss_acc += g_loss_batch_l1
            epoch_g_total_loss_acc += g_loss_batch_combined

            gan_progress_bar.on_batch_end(batch_i, logs={
                'd_loss': d_loss_batch, 
                'g_adv': g_loss_batch_adv, 
                'g_l1': g_loss_batch_l1,
                'g_total': g_loss_batch_combined
            })

        # End of epoch: calculate average losses
        avg_d_loss = epoch_d_loss_acc / n_batches
        avg_g_adv_loss = epoch_g_adv_loss_acc / n_batches
        avg_g_l1_loss = epoch_g_l1_loss_acc / n_batches
        avg_g_total_loss = epoch_g_total_loss_acc / n_batches

        d_losses.append(avg_d_loss)
        g_adv_losses.append(avg_g_adv_loss)
        g_l1_losses.append(avg_g_l1_loss)
        g_total_losses.append(avg_g_total_loss)
        
        gan_progress_bar.on_epoch_end(epoch, logs={
            'd_loss': avg_d_loss, 
            'g_adv': avg_g_adv_loss, 
            'g_l1': avg_g_l1_loss,
            'g_total': avg_g_total_loss
        })

        # Save the generator model periodically (e.g., every 10 epochs or at the end)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            g_model_save_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.h5"
            g_model.save(g_model_save_path)
            print(f"Generator model saved to {g_model_save_path} at epoch {epoch+1}")

    print("GAN training complete.")
    final_g_model_save_path = f"{os.path.splitext(save_path)[0]}_final_generator.h5"
    g_model.save(final_g_model_save_path)
    print(f"Final generator model saved to {final_g_model_save_path}")

    # --- Plot and Save Loss Curves for GAN ---
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(d_losses, label=f'Discriminator Loss (Avg: {np.mean(d_losses[-10:]):.3f})')
        plt.plot(g_total_losses, label=f'Generator Total Loss (Avg: {np.mean(g_total_losses[-10:]):.3f})')
        plt.plot(g_adv_losses, label=f'Generator Adversarial (Avg: {np.mean(g_adv_losses[-10:]):.3f})')
        plt.plot(g_l1_losses, label=f'Generator L1 (Avg: {np.mean(g_l1_losses[-10:]):.3f})')
        plt.title(f'GAN Training Losses ({generator_model_name})')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        base_save_path, _ = os.path.splitext(save_path)
        gan_loss_curve_path = f"{base_save_path}_gan_loss_curves.png"
        plt.savefig(gan_loss_curve_path); plt.close()
        print(f"GAN loss curves saved to: {gan_loss_curve_path}")
    except Exception as e:
        print(f"Error plotting GAN loss curves: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="訓練圖像上色模型 (U-Net 或 GAN)")
    parser.add_argument('--bw_dir', type=str, required=True, help="包含黑白 (L 通道) 圖像的資料夾路徑 (應為 512x512)。")
    parser.add_argument('--color_dir', type=str, required=True, help="包含原始彩色圖像的資料夾路徑。")
    parser.add_argument('--model', type=str, choices=['unet_vgg16', 'unet_relu_leaky', 'unet_advanced_prelu'], default='unet_relu_leaky', help="要訓練的 U-Net 模型名稱 (在 GAN 模式下作為生成器)。")
    parser.add_argument('--epochs', type=int, default=50, help="訓練的 epoch 數量。")
    parser.add_argument('--batch_size', type=int, default=16, help="訓練的 batch_size。注意：GANs 通常使用較小的 batch_size，例如 1 或 4。")
    parser.add_argument('--lr', type=float, default=0.0001, help="學習率 (對於 U-Net 單獨訓練，或作為 GAN 中生成器 L1/MAE 損失的學習率參考)。")
    parser.add_argument('--save_path', type=str, required=True, 
                        help="儲存訓練後模型的完整檔案路徑 (例如：trained_models/model_best_version_lr0p0001.h5)。GAN 模式下檔名會被修改以標註 epoch 或 'final_generator'。")
    parser.add_argument('--loss_type', type=str, choices=['mse', 'mae', 'l1'], default='mae', help="U-Net 單獨訓練時使用的損失函數類型 (mse 或 mae/l1)。對於 GAN 模式，此選項定義生成器的重建損失類型 (建議 'mae'/'l1')。")

    # --- GAN 相關參數 ---
    parser.add_argument('--train_mode', type=str, choices=['unet', 'gan'], default='unet', help="訓練模式：'unet' (單獨 U-Net) 或 'gan' (U-Net 作為生成器的 GAN)。")
    parser.add_argument('--lambda_l1', type=float, default=100.0, help="GAN 模式下，L1 重建損失在生成器總損失中的權重。")
    parser.add_argument('--lr_discriminator', type=float, default=0.0002, help="GAN 模式下，判別器的學習率。")
    parser.add_argument('--lr_generator_gan', type=float, default=0.0002, help="GAN 模式下，生成器在對抗性訓練時的學習率 (Adam beta1 通常設為 0.5)。")
    # --- GAN 相關參數結束 ---

    args = parser.parse_args()
    
    output_model_dir = os.path.dirname(args.save_path)
    if output_model_dir and not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
        print(f"已建立模型儲存資料夾: {output_model_dir} (來自 __main__)")

    if args.train_mode == 'unet':
        print("以 UNET 模式開始訓練...")
        train_model(args.bw_dir, args.color_dir, args.model, args.epochs, args.batch_size, args.lr, args.save_path, args.loss_type)
    elif args.train_mode == 'gan':
        print("以 GAN 模式開始訓練...")
        if args.loss_type not in ['mae', 'l1']:
            print(f"警告: GAN 模式下，生成器的重建損失建議使用 'mae' 或 'l1'。目前設定為: {args.loss_type}")
        train_gan(
            bw_image_dir=args.bw_dir,
            color_image_dir=args.color_dir,
            generator_model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            g_optimizer_lr_recon=args.lr, 
            d_optimizer_lr=args.lr_discriminator,
            gan_optimizer_lr=args.lr_generator_gan,
            lambda_l1=args.lambda_l1,
            save_path=args.save_path,
            loss_type_g_recon=args.loss_type
        )
    else:
        raise ValueError(f"未知的訓練模式: {args.train_mode}")

    # 如何執行 (U-Net):
    # python train.py --train_mode unet --bw_dir bw_images_512 --color_dir 1000img-paul --model unet_relu_leaky --epochs 10 --batch_size 2 --lr 0.0001 --save_path trained_models/my_unet_model.h5 --loss_type mae
    # 如何執行 (GAN):
    # python train.py --train_mode gan --bw_dir bw_images_512 --color_dir 1000img-paul --model unet_relu_leaky --epochs 50 --batch_size 1 --lr 0.0002 --lr_discriminator 0.0002 --lr_generator_gan 0.0002 --lambda_l1 100 --save_path trained_models/my_gan_model.h5 --loss_type mae

    # 如何執行 (假設 bw_images_512 和 1000img-paul):
    # python train.py --bw_dir bw_images_512 --color_dir 1000img-paul --model best_version --epochs 10 --batch_size 2 --lr 0.0001 --save_path trained_models/my_model.h5 --loss_type mae

    # 如何執行:
    # python train.py --bw_dir path/to/your/bw_images --color_dir path/to/your/original_color_images --model best_version --epochs 50 --batch_size 8 --lr 0.0001 --save_path trained_models/trained_colorizer.h5
    # 假設您的黑白圖片在 'bw_images' 資料夾，原始彩色圖片在 '1000img-paul' 資料夾:
    # python train.py --bw_dir bw_images --color_dir 1000img-paul --model best_version --epochs 10 --batch_size 4 --loss_type mae 