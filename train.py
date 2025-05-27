import argparse
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb # skimage 用於色彩空間轉換
from tqdm import tqdm
from keras.callbacks import Callback
import baseline # 從 baseline.py 匯入模型
import multiprocessing # 新增
from functools import partial # 可能用於簡化參數傳遞

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
        
        ab_channels_np = color_img_lab[:, :, 1:] / 128.0
        
        # embed_input 每個圖像都重新生成隨機向量
        embed_input_np = np.random.randn(1000)

        return l_channel_np.reshape(target_size[0], target_size[1], 1), embed_input_np, ab_channels_np

    except FileNotFoundError:
        # print(f"找不到檔案 (worker)：{bw_fname} 或其對應的彩色圖片。")
        return None
    except Exception as e:
        # print(f"處理檔案 {bw_fname} 時發生錯誤 (worker): {e}")
        return None

def load_and_preprocess_data(bw_image_folder, color_image_folder, target_size=(512, 512)):
    """
    載入黑白圖片 (L 通道) 和對應的彩色圖片 (ab 通道作為目標)。
    圖片會被 resize 到 target_size。
    embed_input 將使用隨機向量。
    此版本使用 multiprocessing 並行處理圖片。
    """
    X_l_list = []
    X_embed_list = []
    Y_ab_list = []
    
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
                l_channel_data, embed_input_data, ab_channels_data = result
                X_l_list.append(l_channel_data)
                X_embed_list.append(embed_input_data)
                Y_ab_list.append(ab_channels_data)
                processed_files += 1
    
    print(f"總共成功處理 {processed_files} 張圖片。")
    if processed_files == 0:
        raise ValueError("沒有成功載入任何圖片，請檢查資料夾路徑和檔案格式或工作函數邏輯。")

    return [np.array(X_l_list), np.array(X_embed_list)], np.array(Y_ab_list)

# 解包參數的輔助函數，因為 pool.imap/map 只接受單個參數的函數
def _process_single_image_pair_unpack(args_tuple):
    return _process_single_image_pair(*args_tuple)

def train_model(bw_image_dir, color_image_dir, model_name, epochs, batch_size, learning_rate, save_path, loss_type='mse'):
    print(f"開始載入資料...")
    # 假設 file_converter.py 將彩色圖片轉換為灰階並儲存在 bw_image_dir
    # color_image_dir 應該是原始彩色圖片的資料夾
    X, Y = load_and_preprocess_data(bw_image_dir, color_image_dir)

    print(f"載入資料完成。輸入 L 通道形狀: {X[0].shape}, Embed 輸入形狀: {X[1].shape}, 輸出 ab 通道形狀: {Y.shape}")

    if X[0].shape[0] == 0:
        print("錯誤：沒有成功載入任何訓練資料。請檢查您的圖片資料夾和檔名。")
        return

    print(f"選擇模型: {model_name}, 使用損失函數: {loss_type.upper()}")
    if model_name == 'unet_vgg16':
        model = baseline.unet_vgg16(learning_rate=learning_rate, loss_function_name=loss_type)
    elif model_name == 'best_version':
        model = baseline.best_version(learning_rate=learning_rate, loss_function_name=loss_type)
    elif model_name == 'unet_advanced_prelu':
        model = baseline.unet_advanced_prelu(learning_rate=learning_rate, loss_function_name=loss_type)
    else:
        raise ValueError("未知的模型名稱。請選擇 'unet_vgg16', 'best_version' 或 'unet_advanced_prelu'。")

    model.summary()

    print("開始訓練模型...")
    # 使用 TQDM Callback
    tqdm_callback = TQDMProgressBar()
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[tqdm_callback]) # verbose=0 因為我們有自訂的 callback

    print("訓練完成。")
    # 直接使用傳入的 save_path 儲存模型
    # 不需要再根據模板生成檔名
    
    # 確保儲存模型的目錄存在 (雖然 main.py 應該已經處理了，但這裡加一道保險)
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"(train.py) 已建立模型儲存資料夾: {output_dir}")
        
    model.save(save_path)
    print(f"模型已儲存至 {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="訓練圖像上色模型")
    parser.add_argument('--bw_dir', type=str, required=True, help="包含黑白 (L 通道) 圖像的資料夾路徑 (應為 512x512)。")
    parser.add_argument('--color_dir', type=str, required=True, help="包含原始彩色圖像的資料夾路徑。")
    parser.add_argument('--model', type=str, choices=['unet_vgg16', 'best_version', 'unet_advanced_prelu'], default='best_version', help="要訓練的模型名稱。")
    parser.add_argument('--epochs', type=int, default=50, help="訓練的 epoch 數量。")
    parser.add_argument('--batch_size', type=int, default=16, help="訓練的 batch_size。")
    parser.add_argument('--lr', type=float, default=0.0001, help="學習率。")
    # 修改 output_name_template 為 save_path
    parser.add_argument('--save_path', type=str, required=True, 
                        help="儲存訓練後模型的完整檔案路徑 (例如：trained_models/model_best_version_lr0p0001.h5)。")
    parser.add_argument('--loss_type', type=str, choices=['mse', 'mae', 'l1'], default='mse', help="要使用的損失函數類型 (mse 或 mae/l1)。")

    args = parser.parse_args()
    
    # 確保輸出資料夾存在 (如果 save_path 包含路徑)
    # 這段邏輯其實在 train_model 內部也加了，但argparse後先檢查一次也好
    output_model_dir = os.path.dirname(args.save_path)
    if output_model_dir and not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
        print(f"已建立模型儲存資料夾: {output_model_dir} (來自 __main__)")

    train_model(args.bw_dir, args.color_dir, args.model, args.epochs, args.batch_size, args.lr, args.save_path, args.loss_type) # 傳遞 args.loss_type

    # 如何執行 (假設 bw_images_512 和 1000img-paul):
    # python train.py --bw_dir bw_images_512 --color_dir 1000img-paul --model best_version --epochs 10 --batch_size 2 --lr 0.0001 --save_path trained_models/my_model.h5 --loss_type mae

    # 如何執行:
    # python train.py --bw_dir path/to/your/bw_images --color_dir path/to/your/original_color_images --model best_version --epochs 50 --batch_size 8 --lr 0.0001 --save_path trained_models/trained_colorizer.h5
    # 假設您的黑白圖片在 'bw_images' 資料夾，原始彩色圖片在 '1000img-paul' 資料夾:
    # python train.py --bw_dir bw_images --color_dir 1000img-paul --model best_version --epochs 10 --batch_size 4 --loss_type mae 