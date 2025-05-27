import argparse
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.models import load_model
# 如果您的 baseline.py 中的模型有自訂層或函數 (例如 LeakyReLU 作為物件傳入),
# 可能需要在 load_model 時提供 custom_objects。
# from baseline import best_version # 這裡可能不需要直接匯入模型定義，除非有複雜的自訂物件

# 為了處理 Keras 自訂物件 (如 LeakyReLU)，如果它是作為 activation function string ('leaky_relu') 應該沒問題
# 但如果是 LeakyReLU(alpha=0.2) 物件，則需要 custom_objects
# 由於 baseline.py 中 LeakyReLU 是直接實例化使用的，我們需要將其加入 custom_objects
from keras.layers import LeakyReLU
custom_objects = {'LeakyReLU': LeakyReLU}

def load_test_image_pairs(l_folder, color_folder, target_size=(512, 512)):
    """
    載入測試用的 L 通道圖片和對應的原始彩色圖片。
    Returns:
        l_images: list of L channel images (0-1 range, shape: H, W, 1)
        original_color_images: list of original RGB images (0-255 range, uint8, shape: H, W, 3)
        filenames: list of filenames for reference
    """
    l_images = []
    l_for_rgb_conversion = [] # L channel in 0-100 range for lab2rgb
    original_color_images = []
    filenames = []

    # 假設 l_folder 中的是灰階圖 (可能是 file_converter.py 的輸出，例如 xxx_bw.png)
    # color_folder 中的是原始彩色圖 (例如 xxx.png)
    l_image_files = sorted(os.listdir(l_folder))
    color_image_files = sorted(os.listdir(color_folder))

    for l_fname in tqdm(l_image_files, desc="載入測試圖片"):
        try:
            base_name, _ = os.path.splitext(l_fname)
            # 假設 _bw 是 file_converter 加上的後綴
            original_base_name = base_name
            if base_name.endswith("_bw"):
                original_base_name = base_name[:-3]
            
            potential_color_fnames = [f"{original_base_name}{ext}" for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
            color_fname_found = None
            for potential_cfname in potential_color_fnames:
                if potential_cfname in color_image_files:
                    color_fname_found = potential_cfname
                    break
            
            if not color_fname_found:
                print(f"找不到 {l_fname} 對應的彩色圖片於 {color_folder}")
                continue

            # 載入 L 通道圖片 (for model input)
            l_img_path = os.path.join(l_folder, l_fname)
            l_pil = Image.open(l_img_path).convert('L').resize(target_size, Image.Resampling.LANCZOS)
            l_channel_input = np.array(l_pil, dtype=float) / 255.0 # Scale to [0,1]
            l_images.append(l_channel_input.reshape(target_size[0], target_size[1], 1))
            
            # 載入原始彩色圖片 (for comparison and L channel for Lab conversion)
            color_img_path = os.path.join(color_folder, color_fname_found)
            color_pil = Image.open(color_img_path).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
            original_color_images.append(np.array(color_pil)) # Keep as RGB uint8 [0-255]
            
            # For Lab conversion, we need L channel from original color image, scaled to [0, 100]
            lab_original = rgb2lab(np.array(color_pil)) # rgb2lab expects RGB in [0,1] or uint8
            l_for_rgb_conversion.append(lab_original[:,:,0]) # L channel from original, range [0,100]

            filenames.append(original_base_name)
        except Exception as e:
            print(f"處理檔案 {l_fname} 時發生錯誤: {e}")
            
    if not l_images:
        raise ValueError("未成功載入任何測試圖片對。請檢查路徑和檔名。")

    return l_images, l_for_rgb_conversion, original_color_images, filenames

def predict_ab_channels(model, l_channel_inputs, embed_inputs):
    """使用模型預測 ab 通道。"""
    predicted_ab = model.predict([l_channel_inputs, embed_inputs], batch_size=1) # batch_size can be tuned
    return predicted_ab * 128.0 # Denormalize from [-1,1] to approx [-128,128]

def reconstruct_rgb_from_lab(l_channel_batch, ab_channels_batch):
    """將 L 通道和 ab 通道合併並轉換回 RGB。"""
    reconstructed_rgb_images = []
    for i in range(len(l_channel_batch)):
        l_ch = l_channel_batch[i]
        ab_ch = ab_channels_batch[i]
        
        # L channel from l_for_rgb_conversion (original L) is in [0,100] range
        # ab_channels are in approx [-128, 128] range
        lab_image = np.zeros((l_ch.shape[0], l_ch.shape[1], 3))
        lab_image[:,:,0] = l_ch
        lab_image[:,:,1:] = ab_ch
        
        rgb_image = lab2rgb(lab_image) # Converts Lab to RGB in [0,1] range
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
        reconstructed_rgb_images.append(rgb_image_uint8)
    return reconstructed_rgb_images

def calculate_metrics(predicted_rgb, original_rgb):
    """計算單對 RGB 圖片的 PSNR 和 SSIM。"""
    # PSNR: original_rgb 和 predicted_rgb 應該是 [0,255] uint8
    # SSIM: multichannel=True for color images, data_range is max value of image (255)
    current_psnr = psnr(original_rgb, predicted_rgb, data_range=255)
    current_ssim = ssim(original_rgb, predicted_rgb, data_range=255, channel_axis=-1, win_size=7) # channel_axis for skimage >= 0.19, use multichannel for older
    return current_psnr, current_ssim

def plot_results(l_inputs_for_plot, predicted_rgbs, original_rgbs, psnr_scores, ssim_scores, filenames, save_dir="results"):
    """繪製並儲存結果圖和指標圖。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_images = len(predicted_rgbs)
    for i in range(num_images):
        plt.figure(figsize=(15, 5))
        
        # L-channel input (scaled to 0-255 for display if needed)
        plt.subplot(1, 3, 1)
        # l_inputs_for_plot[i] is already H,W,1 in [0,1] range, imshow handles it
        plt.imshow(l_inputs_for_plot[i].squeeze(), cmap='gray') 
        plt.title(f"Input L-channel: {filenames[i]}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_rgbs[i])
        plt.title(f"Predicted RGB\nPSNR: {psnr_scores[i]:.2f} dB, SSIM: {ssim_scores[i]:.4f}")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(original_rgbs[i])
        plt.title("Original RGB")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"result_{filenames[i]}.png"))
        plt.close()
        print(f"結果圖已儲存: {os.path.join(save_dir, f'result_{filenames[i]}.png')}")

    # Plot summary metrics
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(filenames, psnr_scores, color='skyblue')
    plt.xlabel("Image")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Scores per Image")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    plt.bar(filenames, ssim_scores, color='lightgreen')
    plt.xlabel("Image")
    plt.ylabel("SSIM")
    plt.title("SSIM Scores per Image")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    metrics_plot_path = os.path.join(save_dir, "summary_metrics.png")
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"指標匯總圖已儲存: {metrics_plot_path}")

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    print(f"\n平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")

def evaluate_model(model_path, test_l_folder, test_color_folder, results_save_dir="evaluation_results"):
    print(f"載入模型: {model_path}")
    # 使用 custom_objects 處理 LeakyReLU (如果模型中用到)
    model = load_model(model_path, custom_objects=custom_objects)
    model.summary() # 顯示模型結構

    print("載入測試圖片...")
    # l_images_input are for model input (normalized L channel)
    # l_channels_for_lab are original L channels (0-100) for color reconstruction
    # original_color_images_rgb are ground truth RGBs (0-255)
    l_images_input, l_channels_for_lab, original_color_images_rgb, filenames = load_test_image_pairs(test_l_folder, test_color_folder)
    
    num_test_images = len(l_images_input)
    if num_test_images == 0:
        print("錯誤: 未找到測試圖片，無法進行評估。")
        return

    # 準備 embed_input (目前為全零向量，與訓練時一致)
    # 假設模型輸入形狀是 512x512
    embed_dim = model.input_shape[1][1] # Or simply 1000 if fixed
    # model.input_shape is a list for multiple inputs: [(None, 512, 512, 1), (None, 1000)]
    test_embed_inputs = np.zeros((num_test_images, embed_dim))
    
    print("對測試圖片進行預測...")
    # 將 list of images 轉為 numpy array for model prediction
    test_l_inputs_np = np.array(l_images_input)
    predicted_ab = predict_ab_channels(model, test_l_inputs_np, test_embed_inputs)
    
    print("重建 RGB 圖像...")
    # l_channels_for_lab is already a list of L channels (0-100)
    reconstructed_rgb_images = reconstruct_rgb_from_lab(l_channels_for_lab, predicted_ab)
    
    psnr_scores = []
    ssim_scores = []
    
    print("計算 PSNR 和 SSIM 指標...")
    for i in tqdm(range(num_test_images), desc="計算指標"):
        pred_rgb = reconstructed_rgb_images[i]
        orig_rgb = original_color_images_rgb[i]
        p, s = calculate_metrics(pred_rgb, orig_rgb)
        psnr_scores.append(p)
        ssim_scores.append(s)
        # print(f"圖片 {filenames[i]}: PSNR={p:.2f}, SSIM={s:.4f}")

    print("繪製並儲存結果...")
    plot_results(test_l_inputs_np, reconstructed_rgb_images, original_color_images_rgb, psnr_scores, ssim_scores, filenames, save_dir=results_save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="評估圖像上色模型並計算 PSNR/SSIM")
    parser.add_argument('--model_path', type=str, required=True, help="已訓練模型的路徑 (.h5 檔案)。")
    parser.add_argument('--l_dir', type=str, required=True, help="包含測試用 L 通道圖像的資料夾 (例如 file_converter.py 的輸出)。")
    parser.add_argument('--color_dir', type=str, required=True, help="包含原始彩色對照圖像的資料夾。")
    parser.add_argument('--results_dir', type=str, default="evaluation_results", help="儲存評估結果 (圖片、圖表) 的資料夾。")
    
    args = parser.parse_args()

    # 確保 L 通道資料夾和彩色資料夾存在
    if not os.path.isdir(args.l_dir):
        print(f"錯誤: L 通道圖像資料夾不存在: {args.l_dir}")
        exit()
    if not os.path.isdir(args.color_dir):
        print(f"錯誤: 彩色圖像資料夾不存在: {args.color_dir}")
        exit()

    evaluate_model(args.model_path, args.l_dir, args.color_dir, args.results_dir)

    # 如何執行:
    # python utils.py --model_path trained_model_best_version_lr0p0001.h5 --l_dir bw_images_512 --color_dir 1000img-paul --results_dir evaluation_run1
    # (請確保 bw_images_512 包含的是 512x512 的 L 通道圖, 1000img-paul 包含原始彩色圖)
