import argparse
import os
import sys
import tensorflow as tf # Added for GPU check

# GPU Detection Code
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"檢測到 {len(gpus)} 個 GPU:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"    已為 GPU {i} 設定 Memory Growth。")
        except RuntimeError as e:
            print(f"    為 GPU {i} 設定 Memory Growth 失敗: {e}")
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"檢測到 {len(logical_gpus)} 個邏輯 GPU。")
else:
    print("未檢測到 GPU。TensorFlow 將使用 CPU。")


# 確保可以找到 train.py 和 utils.py (如果它們不在 PYTHONPATH)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

try:
    from train import train_model
    from utils import evaluate_model
except ImportError as e:
    print(f"無法匯入 train 或 utils 模組: {e}")
    print("請確保 main.py 與 train.py 和 utils.py 在同一個資料夾下，或者它們位於PYTHONPATH中。")
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="完整流程：訓練圖像上色模型並進行評估。 tqdm 進度條將分別顯示於訓練和評估階段。")

    # === 訓練參數 ===
    train_group = parser.add_argument_group('訓練參數')
    train_group.add_argument('--train_bw_dir', type=str, help="(訓練用) 黑白 (L 通道) 圖像資料夾路徑 (512x512)。如果--skip_training為True則非必要。")
    train_group.add_argument('--train_color_dir', type=str, help="(訓練用) 原始彩色圖像資料夾路徑。如果--skip_training為True則非必要。")
    train_group.add_argument('--model_type', type=str, choices=['unet_vgg16', 'unet_relu_leaky', 'unet_advanced_prelu'], default='unet_relu_leaky', help="要訓練的模型名稱。")
    train_group.add_argument('--epochs', type=int, default=50, help="訓練的 epoch 數量。")
    train_group.add_argument('--batch_size', type=int, default=2, help="訓練的 batch_size (預設為2，因512x512圖片記憶體需求較高)。")
    train_group.add_argument('--lr', type=float, default=0.0001, help="學習率。")
    train_group.add_argument('--loss_type', type=str, choices=['mse', 'mae', 'l1'], default='mse', help="(訓練用) 要使用的損失函數類型 (mse 或 mae/l1)。如果--skip_training為True則非必要。")
    train_group.add_argument('--model_output_template', type=str, default="trained_models/model_[model_type]_loss[loss_type]_lr[lr].h5",
                        help="儲存訓練後模型的檔案名稱模板。預設會在 'trained_models' 子資料夾下。")

    # === 評估參數 ===
    eval_group = parser.add_argument_group('評估參數')
    eval_group.add_argument('--test_l_dir', type=str, required=True, help="(測試用) L 通道圖像資料夾路徑。")
    eval_group.add_argument('--test_color_dir', type=str, required=True, help="(測試用) 原始彩色對照圖像資料夾路徑。")
    eval_group.add_argument('--eval_results_dir', type=str, default="evaluation_results_main",
                        help="儲存評估結果 (圖片、圖表) 的資料夾。")
    
    # === 控制流程參數 ===
    control_group = parser.add_argument_group('控制流程參數')
    control_group.add_argument('--skip_training', action='store_true', help="跳過訓練階段，僅使用 --trained_model_path 進行評估。")
    control_group.add_argument('--trained_model_path', type=str, default=None, 
                               help="若要跳過訓練或指定特定模型進行評估，請提供已訓練模型的 .h5 檔案路徑。")

    args = parser.parse_args()

    final_model_path_to_evaluate = args.trained_model_path

    if not args.skip_training:
        print("--- 開始訓練階段 ---")
        if not args.train_bw_dir or not args.train_color_dir:
            print("錯誤: 進行訓練需要 --train_bw_dir 和 --train_color_dir 參數。")
            parser.print_help()
            sys.exit(1)

        # 根據模板和參數確定模型將儲存的路徑
        lr_str_format = str(args.lr).replace('.', 'p')
        model_save_path = args.model_output_template.replace('[model_type]', args.model_type).replace('[lr]', lr_str_format)
        model_save_path = model_save_path.replace('[loss_type]', args.loss_type)
        
        # 確保模型輸出資料夾存在
        model_output_dir = os.path.dirname(model_save_path)
        if model_output_dir and not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
            print(f"已建立模型儲存資料夾: {model_output_dir}")

        train_model(
            bw_image_dir=args.train_bw_dir,
            color_image_dir=args.train_color_dir,
            model_name=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_path=model_save_path, # train_model 內部會使用此模板生成完整檔名並儲存
            loss_type=args.loss_type  # 傳遞 loss_type
        )
        final_model_path_to_evaluate = model_save_path # train_model 儲存模型於此路徑
        print(f"--- 訓練完成。模型已儲存至: {final_model_path_to_evaluate} ---")
    
    else: # 跳過訓練
        if not args.trained_model_path:
            print("錯誤: 要求跳過訓練 (--skip_training)，但未透過 --trained_model_path 提供已訓練模型路徑。")
            parser.print_help()
            sys.exit(1)
        # final_model_path_to_evaluate 已在開始時從 args.trained_model_path 設定
        print(f"--- 跳過訓練階段。將使用模型: {final_model_path_to_evaluate} 進行評估 ---")

    # 檢查最終用於評估的模型路徑是否存在
    if not final_model_path_to_evaluate or not os.path.exists(final_model_path_to_evaluate):
        print(f"錯誤: 找不到要評估的模型檔案: '{final_model_path_to_evaluate}'。請檢查路徑，或確認訓練是否已成功執行並儲存模型。")
        sys.exit(1)
        
    print("\n--- 開始評估階段 ---")
    if not os.path.isdir(args.test_l_dir):
        print(f"錯誤: 測試用 L 通道圖像資料夾不存在: {args.test_l_dir}")
        sys.exit(1)
    if not os.path.isdir(args.test_color_dir):
        print(f"錯誤: 測試用彩色圖像資料夾不存在: {args.test_color_dir}")
        sys.exit(1)

    evaluate_model(
        model_path=final_model_path_to_evaluate,
        test_l_folder=args.test_l_dir,
        test_color_folder=args.test_color_dir,
        results_save_dir=args.eval_results_dir
    )
    print("--- 評估完成 ---")
    print(f"評估結果已儲存於: {os.path.abspath(args.eval_results_dir)}")

# === 如何執行 ===
# 1. 訓練並評估 (使用 MSE):
# python main.py --train_bw_dir bw_images_512 --train_color_dir 1000img-paul --model_type best_version --epochs 10 --batch_size 2 --lr 0.0001 --loss_type mse --test_l_dir path/to/your/test_l_images --test_color_dir path/to/your/test_color_images --eval_results_dir evaluation_output
#
# 1.1 訓練並評估 (使用 MAE):
# python main.py --train_bw_dir bw_images_512 --train_color_dir 1000img-paul --model_type best_version --epochs 10 --batch_size 2 --lr 0.0001 --loss_type mae --test_l_dir path/to/your/test_l_images --test_color_dir path/to/your/test_color_images --eval_results_dir evaluation_output
#
# 2. 僅評估 (假設模型已訓練並儲存於 trained_models/model_best_version_lossmae_lr0p0001.h5):
# python main.py --skip_training --trained_model_path trained_models/model_best_version_lossmae_lr0p0001.h5 --test_l_dir path/to/your/test_l_images --test_color_dir path/to/your/test_color_images --eval_results_dir evaluation_output
#
# 請將 path/to/your/test_l_images 和 path/to/your/test_color_images 替換為您實際的測試圖片資料夾路徑。
# 訓練用的 bw_images_512 和 1000img-paul 也應為您的實際路徑。 