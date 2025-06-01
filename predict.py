import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, PReLU

# 定義自訂物件以載入包含這些層的模型
custom_objects = {
    'LeakyReLU': LeakyReLU,
    'PReLU': PReLU
}

def colorize_folder(model, input_folder_path, output_folder_path, embed_dim=1000, model_input_size=(512, 512)):
    """
    使用指定的模型對資料夾中的黑白圖像進行著色。

    參數:
    - model: 已訓練的 Keras 模型。
             期望輸入為 [L 通道圖像, 嵌入向量]。
             L 通道圖像應為 (批次大小, 高度, 寬度, 1)，值在 [-1, 1] 範圍內。
             嵌入向量應為 (批次大小, embed_dim)。
             模型應輸出 ab 通道，形狀為 (批次大小, 高度, 寬度, 2)，值在 [-1, 1] 範圍內。
    - input_folder_path: 包含黑白圖像的資料夾路徑。
    - output_folder_path: 保存著色圖像的資料夾路徑。
    - embed_dim: 嵌入向量的維度。
    - model_input_size: 模型期望的輸入圖像尺寸 (寬度, 高度)。
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"已建立輸出資料夾: {output_folder_path}")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"在 {input_folder_path} 中沒有找到支援的圖像檔案。")
        return

    print(f"找到 {len(image_files)} 個圖像檔案，開始處理...")

    for filename in image_files:
        try:
            image_path = os.path.join(input_folder_path, filename)
            
            img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"警告：無法讀取圖像 {filename}，跳過。")
                continue
            
            original_height, original_width = img_gray.shape[:2]

            l_channel_resized = cv2.resize(img_gray, model_input_size, interpolation=cv2.INTER_AREA)
            l_normalized = l_channel_resized.astype(np.float32) / 255.0
            l_model_input = l_normalized[np.newaxis, ..., np.newaxis]

            embed_vector = np.random.randn(1, embed_dim).astype(np.float32)
            print(f"為 {filename} 生成了隨機嵌入向量 (前3個值): {embed_vector[0, :3]}...")

            pred_ab = model.predict([l_model_input, embed_vector]) 
            
            pred_ab_processed = pred_ab[0]
            pred_ab_scaled = ((pred_ab_processed + 1.0) / 2.0) * 255.0
            pred_ab_uint8 = np.clip(pred_ab_scaled, 0, 255).astype(np.uint8)

            ab_resized_to_original = cv2.resize(pred_ab_uint8, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

            l_original_for_merge = img_gray 

            a_channel_pred = ab_resized_to_original[:, :, 0]
            b_channel_pred = ab_resized_to_original[:, :, 1]
            final_lab_image = cv2.merge([l_original_for_merge, a_channel_pred, b_channel_pred])

            colorized_bgr = cv2.cvtColor(final_lab_image, cv2.COLOR_LAB2BGR)

            output_image_path = os.path.join(output_folder_path, filename)
            cv2.imwrite(output_image_path, colorized_bgr)
            # print(f"已處理並保存: {output_image_path}")

        except Exception as e:
            print(f"處理圖像 {filename} 時發生錯誤: {e}")

    print(f"所有圖像處理完成。已保存至 {output_folder_path}")

def main():
    parser = argparse.ArgumentParser(description="使用訓練好的模型對資料夾中的黑白圖像進行著色。")
    parser.add_argument('--model_path', type=str, required=True, help="已訓練模型的路徑 (.h5 或 .keras 檔案)。")
    parser.add_argument('--input_folder', type=str, required=True, help="包含黑白圖像的輸入資料夾路徑。")
    parser.add_argument('--output_folder', type=str, required=True, help="保存著色後圖像的輸出資料夾路徑。")
    parser.add_argument('--embed_dim', type=int, default=1000, help="模型使用的嵌入向量維度。")
    parser.add_argument('--img_height', type=int, default=512, help="模型期望的輸入圖像高度。")
    parser.add_argument('--img_width', type=int, default=512, help="模型期望的輸入圖像寬度。")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"錯誤：模型檔案未找到於 {args.model_path}")
        return
    if not os.path.isdir(args.input_folder):
        print(f"錯誤：輸入資料夾未找到於 {args.input_folder}")
        return

    model_input_size = (args.img_width, args.img_height)

    print(f"正在從 {args.model_path} 載入模型...")
    try:
        model = load_model(args.model_path, custom_objects=custom_objects, compile=False)
        print("模型載入成功。")
    except Exception as e:
        print(f"載入模型失敗: {e}")
        print("請確保模型路徑正確，且 custom_objects (若需要) 已正確定義。")
        return

    print(f"開始對 '{args.input_folder}' 中的圖像進行著色...")
    colorize_folder(
        model=model,
        input_folder_path=args.input_folder,
        output_folder_path=args.output_folder,
        embed_dim=args.embed_dim,
        model_input_size=model_input_size
    )

if __name__ == '__main__':
    main()