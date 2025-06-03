import argparse
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2

from keras.models import load_model
# 如果您的 baseline.py 中的模型有自訂層或函數 (例如 LeakyReLU 作為物件傳入),
# 可能需要在 load_model 時提供 custom_objects。
# from baseline import best_version # 這裡可能不需要直接匯入模型定義，除非有複雜的自訂物件

# 為了處理 Keras 自訂物件 (如 LeakyReLU)，如果它是作為 activation function string ('leaky_relu') 應該沒問題
# 但如果是 LeakyReLU(alpha=0.2) 物件，則需要 custom_objects
# 由於 baseline.py 中 LeakyReLU 是直接實例化使用的，我們需要將其加入 custom_objects
from keras.layers import LeakyReLU, PReLU

# Set default font for matplotlib to avoid missing character issues
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica'] # Fallback fonts

custom_objects = {
    'LeakyReLU': LeakyReLU,
    'PReLU': PReLU
}

def load_and_preprocess_image_pair_for_eval(l_path, color_path, target_size=(512, 512)):
    """Loads and preprocesses a single L-channel and color image pair for evaluation."""
    try:
        # 載入 L 通道圖像 (黑白輸入)
        l_image = Image.open(l_path).convert('L')
        l_image = l_image.resize(target_size)
        l_array = np.array(l_image, dtype=np.float32) / 255.0 # 標準化到 [0, 1]
        l_array = np.expand_dims(l_array, axis=-1) # 添加通道維度 -> (height, width, 1)

        # 載入原始彩色圖像 (用於獲取真實的 ab 通道和原始 L 通道)
        color_image = Image.open(color_path).convert('RGB')
        original_l_for_reconstruction = color_image.convert('L').resize(target_size) # 用於重建的L
        original_l_for_reconstruction_arr = np.array(original_l_for_reconstruction, dtype=np.uint8)
        
        color_image_resized = color_image.resize(target_size)
        lab_image = cv2.cvtColor(np.array(color_image_resized), cv2.COLOR_RGB2Lab)
        
        # 真實的 ab 通道 (標準化到 [-1, 1])
        true_ab_array = (lab_image[:, :, 1:].astype(np.float32) - 128) / 128.0
        
        return l_array, true_ab_array, original_l_for_reconstruction_arr, os.path.basename(l_path)
    except FileNotFoundError:
        print(f"Warning: Image file not found: {l_path} or {color_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error processing image pair {l_path} and {color_path}: {e}")
        return None, None, None, None

def load_test_image_pairs(l_folder, color_folder, target_size=(512, 512)):
    """Loads all test image pairs."""
    l_images = []
    true_ab_channels = []
    original_l_for_reconstruction_list = []
    filenames = []

    l_files = sorted([os.path.join(l_folder, f) for f in os.listdir(l_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Starting to load test image pairs from {l_folder} and {color_folder}...")
    for l_file_path in tqdm(l_files, desc="Loading Test Images"):
        base_name, _ = os.path.splitext(os.path.basename(l_file_path))
        # 假設 L 圖像檔名類似 'image_001_bw.png'，對應的彩色圖像是 'image_001.png'
        # 或 L 是 'image_001.png'，彩色也是 'image_001.png'
        original_name_stem = base_name
        if base_name.endswith('_bw'):
            original_name_stem = base_name[:-3]
        
        color_file_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_color_path = os.path.join(color_folder, original_name_stem + ext)
            if os.path.exists(potential_color_path):
                color_file_path = potential_color_path
                    break
            
        if not color_file_path:
            print(f"Warning: Color image not found for {l_file_path} (tried base name: {original_name_stem}). Skipping this image.")
                continue

        l_img_arr, true_ab_arr, orig_l_reconstruct_arr, fname = load_and_preprocess_image_pair_for_eval(l_file_path, color_file_path, target_size)
        if l_img_arr is not None:
            l_images.append(l_img_arr)
            true_ab_channels.append(true_ab_arr)
            original_l_for_reconstruction_list.append(orig_l_reconstruct_arr)
            filenames.append(fname)
            
    if not l_images:
        print("Error: Failed to load any test image pairs. Please check folder paths and image files.")
        return None, None, None, None

    return np.array(l_images), np.array(true_ab_channels), original_l_for_reconstruction_list, filenames

def predict_ab_channels(model, l_input_batch, embed_dim=1000):
    """Predicts ab channels using the model."""
    batch_size = l_input_batch.shape[0]
    # 評估時也使用隨機嵌入向量，與訓練時一致
    random_embedding = np.random.randn(batch_size, embed_dim).astype(np.float32)
    predicted_ab_batch = model.predict([l_input_batch, random_embedding])
    return predicted_ab_batch

def reconstruct_rgb_from_lab(l_channel_original, ab_channels_predicted, target_size=(512,512)):
    """Reconstructs RGB image from L channel and predicted ab channels."""
    # l_channel_original 是 (H, W) uint8 [0, 255]
    # ab_channels_predicted 是 (H, W, 2) float32 [-1, 1]
    
    # 反標準化 ab 通道
    ab_unnormalized = (ab_channels_predicted * 128 + 128).astype(np.uint8)
    
    # 確保 L 通道是 (H, W, 1)
    l_reshaped = np.expand_dims(l_channel_original, axis=-1)
    
    # 合併 L 和 ab 通道
    lab_reconstructed = np.concatenate((l_reshaped, ab_unnormalized), axis=-1)
    
    # 轉換回 BGR (OpenCV 格式)，然後轉 RGB (Pillow/Matplotlib 格式)
    bgr_reconstructed = cv2.cvtColor(lab_reconstructed, cv2.COLOR_Lab2BGR)
    rgb_reconstructed = cv2.cvtColor(bgr_reconstructed, cv2.COLOR_BGR2RGB)
    
    # 如果需要，調整回原始圖像大小 (如果模型輸出尺寸不同)
    # 但在這裡，我們假設 l_channel_original 和 ab_channels_predicted 已經是目標尺寸
    # rgb_image = Image.fromarray(rgb_reconstructed).resize(original_size, Image.Resampling.BICUBIC)
    return rgb_reconstructed

def calculate_metrics(predicted_rgb, original_rgb):
    """Calculates PSNR and SSIM."""
    # 確保數據類型正確 (通常是 uint8)
    predicted_rgb = predicted_rgb.astype(np.uint8)
    original_rgb = original_rgb.astype(np.uint8)

    current_psnr = psnr(original_rgb, predicted_rgb, data_range=255)
    current_ssim = ssim(original_rgb, predicted_rgb, data_range=255, channel_axis=-1, win_size=7) # win_size 必須是奇數且 <= min(H,W)
    return current_psnr, current_ssim

def plot_results(l_input_display, pred_rgb, orig_rgb, psnr_val, ssim_val, save_path, filename):
    """Plots a comparison for a single result."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Use a short version of filename for the suptitle if it's too long
    display_filename = filename if len(filename) < 50 else filename[:47] + "..."
    fig.suptitle(f"Image: {display_filename} - PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}", fontsize=12)
    
    axes[0].imshow(l_input_display, cmap='gray')
    axes[0].set_title("Input L-channel (Grayscale)")
    axes[0].axis('off')
    
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Predicted RGB")
    axes[1].axis('off')
    
    axes[2].imshow(orig_rgb)
    axes[2].set_title("Original RGB (Ground Truth)")
    axes[2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Ensure filename is filesystem-friendly (already os.path.basename, but double check for special chars if added manually)
    safe_filename = "comparison_" + filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(save_path, safe_filename))
    plt.close(fig)

def plot_summary_metrics(psnr_scores, ssim_scores, filenames, save_dir):
    """Plots summary bar charts for PSNR and SSIM for all images."""
    if not psnr_scores or not ssim_scores:
        print("No metrics to plot for summary.")
        return

    num_images = len(filenames)
    x = np.arange(num_images)
    width = 0.35

    # Shorten filenames for x-axis labels if they are too long
    xtick_labels = [(fn if len(fn) < 20 else fn[:17] + "...") for fn in filenames]

    fig, ax1 = plt.subplots(figsize=(max(10, num_images * 0.8), 7)) # Increased height for better label spacing

    rects1 = ax1.bar(x - width/2, psnr_scores, width, label='PSNR (dB)', color='skyblue')
    ax1.set_xlabel('Image Filename')
    ax1.set_ylabel('PSNR (dB)', color='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.grid(axis='y', linestyle='--', alpha=0.7) # Add grid for PSNR

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, ssim_scores, width, label='SSIM', color='lightcoral')
    ax2.set_ylabel('SSIM', color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    ax2.set_ylim(0, 1) 
    ax2.grid(axis='y', linestyle=':', alpha=0.7) # Add grid for SSIM

    fig.suptitle('PSNR and SSIM Metrics for All Test Images', fontsize=14)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Place legend below the plot to avoid overlap with rotated x-axis labels
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    
    plt.subplots_adjust(bottom=0.25, top=0.92) # Adjust bottom for rotated labels and legend, top for suptitle
    # plt.tight_layout(rect=[0, 0.05, 1, 0.93]) 
    
    summary_plot_path = os.path.join(save_dir, "summary_metrics_per_image.png")
    plt.savefig(summary_plot_path)
    plt.close(fig)
    print(f"Summary metrics chart per image saved to: {summary_plot_path}")


def evaluate_model(model_path, test_l_folder, test_color_folder, results_save_dir="evaluation_results"):
    """Evaluates model performance and saves results."""
    print(f"Loading model from {model_path}...")
    try:
    model = load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(f"Ensure '{model_path}' is a valid Keras model file and all custom layers (like LeakyReLU, PReLU) are correctly defined in custom_objects.")
        return None, None 

    print("Model loaded successfully.")

    # 載入測試數據
    l_test_batch, true_ab_test_batch, original_l_for_reconstruction_list, filenames = load_test_image_pairs(test_l_folder, test_color_folder)

    if l_test_batch is None or len(l_test_batch) == 0:
        print("Evaluation terminated as no test data could be loaded.")
        return None, None

    # 預測 ab 通道
    print("Starting to predict ab channels...")
    predicted_ab_batch = predict_ab_channels(model, l_test_batch)
    print("Prediction finished.")

    all_psnr = []
    all_ssim = []

    # 確保結果儲存資料夾存在
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
        print(f"Created results save directory: {results_save_dir}")

    print("Starting image reconstruction and metric calculation...")
    for i in tqdm(range(len(l_test_batch)), desc="Evaluating Images"):
        # 獲取原始 L 通道 (用於顯示和重建)
        # l_test_batch[i] 是標準化到 [0,1] 的 L，用於模型輸入
        # original_l_for_reconstruction_list[i] 是 uint8 [0,255] 的 L，用於重建
        l_display = (l_test_batch[i].squeeze() * 255).astype(np.uint8) # 用於顯示的 L，反標準化
        l_for_reconstruction = original_l_for_reconstruction_list[i] # 直接使用原始 L 進行重建

        # 重建 RGB 圖像
        predicted_rgb = reconstruct_rgb_from_lab(l_for_reconstruction, predicted_ab_batch[i])
        
        # 獲取原始彩色圖像用於比較
        # 我們需要從 color_folder 重新載入原始彩色圖像，因為 true_ab_test_batch 只包含 ab 通道
        base_name, _ = os.path.splitext(filenames[i])
        original_name_stem = base_name
        if base_name.endswith('_bw'):
            original_name_stem = base_name[:-3]
        
        original_color_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_color_path = os.path.join(test_color_folder, original_name_stem + ext)
            if os.path.exists(potential_color_path):
                original_color_path = potential_color_path
                break
        
        if not original_color_path:
            print(f"Warning: Original color image for {filenames[i]} not found when calculating metrics. Skipping metrics for this image.")
            continue
        try:
            original_color_img_pil = Image.open(original_color_path).convert('RGB').resize((512,512) if l_display.shape[0] == 512 else (256,256)) # 確保尺寸一致
            original_rgb_for_metric = np.array(original_color_img_pil)
        except Exception as e:
            print(f"Warning: Error loading {original_color_path} for metric calculation: {e}")
            continue

        # 計算指標
        current_psnr, current_ssim = calculate_metrics(predicted_rgb, original_rgb_for_metric)
        all_psnr.append(current_psnr)
        all_ssim.append(current_ssim)
        
        # 儲存對比圖
        plot_results(l_display, predicted_rgb, original_rgb_for_metric, current_psnr, current_ssim, results_save_dir, filenames[i])

    if not all_psnr or not all_ssim:
        print("Warning: Failed to calculate metrics for any image.")
        avg_psnr, avg_ssim = 0.0, 0.0 # Ensure float for JSON
    else:
        avg_psnr = float(np.mean(all_psnr))
        avg_ssim = float(np.mean(all_ssim))
        print(f"\n--- Evaluation Metrics ---")
        print(f"Number of images processed: {len(all_psnr)}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")

        # 繪製指標匯總圖
        plot_summary_metrics(all_psnr, all_ssim, filenames, results_save_dir)
    
    # 將指標儲存到 JSON 檔案
    metrics_data = {
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "num_images_evaluated": len(all_psnr),
        # Convert numpy floats to standard Python floats for JSON serialization
        "individual_psnr": {fn: float(p) for fn, p in zip(filenames, all_psnr)} if all_psnr else {},
        "individual_ssim": {fn: float(s) for fn, s in zip(filenames, all_ssim)} if all_ssim else {},
    }
    metrics_file_path = os.path.join(results_save_dir, "metrics.json")
    try:
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"Metrics data saved to: {metrics_file_path}")
    except Exception as e:
        print(f"Error saving metrics data to {metrics_file_path}: {e}")

    return avg_psnr, avg_ssim 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Image Colorization Model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .h5 model file.")
    parser.add_argument('--l_dir', type=str, required=True, help="Folder containing L-channel test images.")
    parser.add_argument('--color_dir', type=str, required=True, help="Folder containing original color reference images for testing.")
    parser.add_argument('--results_dir', type=str, default="evaluation_results", help="Folder to save evaluation results (images, charts).")
    
    args = parser.parse_args()

    evaluate_model(args.model_path, args.l_dir, args.color_dir, args.results_dir)

    # 如何執行:
    # python utils.py --model_path trained_model_best_version_lr0p0001.h5 --l_dir bw_images_512 --color_dir 1000img-paul --results_dir evaluation_run1
    # (請確保 bw_images_512 包含的是 512x512 的 L 通道圖, 1000img-paul 包含原始彩色圖)