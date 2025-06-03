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
    from train import train_model, train_gan
    from utils import evaluate_model
except ImportError as e:
    print(f"無法匯入 train 或 utils 模組: {e}")
    print("請確保 main.py 與 train.py 和 utils.py 在同一個資料夾下，或者它們位於PYTHONPATH中。")
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="完整流程：訓練圖像上色模型 (U-Net 或 GAN) 並進行評估。")

    # === 模式選擇 ===
    parser.add_argument('--train_mode', type=str, choices=['unet', 'gan'], default='unet', 
                        help="訓練模式: 'unet' (單獨 U-Net) 或 'gan' (U-Net 作為生成器的 GAN)。")

    # === 基礎訓練參數 (對 U-Net 和 GAN 中的生成器都可能有用) ===
    base_train_group = parser.add_argument_group('基礎訓練參數')
    base_train_group.add_argument('--train_bw_dir', type=str, help="(訓練用) 黑白 (L 通道) 圖像資料夾路徑 (512x512)。如果--skip_training為True則非必要。")
    base_train_group.add_argument('--train_color_dir', type=str, help="(訓練用) 原始彩色圖像資料夾路徑。如果--skip_training為True則非必要。")
    base_train_group.add_argument('--model_type', type=str, choices=['unet_vgg16', 'unet_relu_leaky', 'unet_advanced_prelu', 'best_version'], default='best_version', 
                                help="要訓練的U-Net模型名稱 (在GAN模式下作為生成器)。'best_version' 是 'unet_relu_leaky' 的別名。")
    base_train_group.add_argument('--epochs', type=int, default=50, help="訓練的 epoch 數量。")
    base_train_group.add_argument('--batch_size', type=int, default=2, 
                                help="訓練的 batch_size (預設為2。對於GAN，通常建議更小，例如1或4，視記憶體而定)。")
    base_train_group.add_argument('--lr', type=float, default=0.0001, 
                                help="U-Net單獨訓練時的學習率，或GAN模式下生成器進行L1/重建損失時的學習率參考。")
    base_train_group.add_argument('--loss_type', type=str, choices=['mse', 'mae', 'l1'], default='mae', 
                                help="U-Net單獨訓練時的損失函數，或GAN模式下生成器的重建損失類型 (建議'mae'/'l1')。如果--skip_training為True則非必要。")
    base_train_group.add_argument('--model_output_template', type=str, default="trained_models/model_[mode]_[model_type]_loss[loss_type]_lr[lr].h5",
                        help="儲存訓練後模型的檔案名稱模板。預設會在 'trained_models' 子資料夾下。 GAN 模式下，儲存的是生成器。")

    # === GAN 特定訓練參數 ===
    gan_train_group = parser.add_argument_group('GAN 特定訓練參數 (僅在 --train_mode gan 時使用)')
    gan_train_group.add_argument('--lambda_l1', type=float, default=100.0, help="GAN 模式下，L1 重建損失在生成器總損失中的權重。")
    gan_train_group.add_argument('--lr_discriminator', type=float, default=0.0002, help="GAN 模式下，判別器的學習率。")
    gan_train_group.add_argument('--lr_generator_gan', type=float, default=0.0002, help="GAN 模式下，生成器在對抗性訓練時的學習率。")

    # === 評估參數 ===
    eval_group = parser.add_argument_group('評估參數')
    eval_group.add_argument('--test_l_dir', type=str, required=False, help="(測試用) L 通道圖像資料夾路徑。若只訓練則非必要。")
    eval_group.add_argument('--test_color_dir', type=str, required=False, help="(測試用) 原始彩色對照圖像資料夾路徑。若只訓練則非必要。")
    eval_group.add_argument('--eval_results_dir', type=str, default="evaluation_results_main",
                        help="儲存評估結果 (圖片、圖表) 的資料夾。")
    
    # === 控制流程參數 ===
    control_group = parser.add_argument_group('控制流程參數')
    control_group.add_argument('--skip_training', action='store_true', help="跳過訓練階段，僅使用 --trained_model_path 進行評估。")
    control_group.add_argument('--skip_evaluation', action='store_true', help="跳過評估階段，僅執行訓練。")
    control_group.add_argument('--trained_model_path', type=str, default=None, 
                               help="若要跳過訓練或指定特定模型進行評估，請提供已訓練模型的 .h5 檔案路徑。")

    args = parser.parse_args()

    final_model_path_to_evaluate = args.trained_model_path

    if not args.skip_training:
        print(f"--- 開始 {args.train_mode.upper()} 訓練階段 ---")
        if not args.train_bw_dir or not args.train_color_dir:
            print("錯誤: 進行訓練需要 --train_bw_dir 和 --train_color_dir 參數。")
            parser.print_help()
            sys.exit(1)

        lr_str_format = str(args.lr).replace('.', 'p') # For filename
        # Include train_mode in the filename template
        model_save_path = args.model_output_template.replace('[mode]', args.train_mode).replace('[model_type]', args.model_type).replace('[lr]', lr_str_format)
        model_save_path = model_save_path.replace('[loss_type]', args.loss_type) # loss_type for G reconstruction in GAN
        
        model_output_dir = os.path.dirname(model_save_path)
        if model_output_dir and not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
            print(f"已建立模型儲存資料夾: {model_output_dir}")

        if args.train_mode == 'unet':
        train_model(
                bw_image_dir=args.train_bw_dir,
                color_image_dir=args.train_color_dir,
                model_name=args.model_type,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_path=model_save_path, 
                loss_type=args.loss_type
            )
            final_model_path_to_evaluate = model_save_path 
        elif args.train_mode == 'gan':
            if args.model_type not in ['unet_relu_leaky', 'best_version', 'unet_advanced_prelu']:
                 print(f"警告: GAN 模式下建議使用 'unet_relu_leaky' (best_version) 或 'unet_advanced_prelu' 作為生成器。您選擇了 {args.model_type}")
            train_gan(
                bw_image_dir=args.train_bw_dir,
                color_image_dir=args.train_color_dir,
                generator_model_name=args.model_type,
                epochs=args.epochs,
                batch_size=args.batch_size,
                g_optimizer_lr_recon=args.lr, # lr for G's L1 loss part
                d_optimizer_lr=args.lr_discriminator,
                gan_optimizer_lr=args.lr_generator_gan,
                lambda_l1=args.lambda_l1,
                save_path=model_save_path, # train_gan will append _epochX or _final_generator
                loss_type_g_recon=args.loss_type # Reconstruction loss for G
            )
            # For GAN, the saved model path for evaluation is typically the final generator
            final_model_path_to_evaluate = f"{os.path.splitext(model_save_path)[0]}_final_generator.h5"
            if not os.path.exists(final_model_path_to_evaluate):
                 # Fallback if the naming convention in train_gan changes or for older saves
                 # This part might need adjustment based on actual save names from train_gan
                 print(f"警告: 未找到預期的最終生成器模型 '{final_model_path_to_evaluate}'. 將嘗試使用基礎儲存路徑 '{model_save_path}' 進行評估 (如果它是一個有效的生成器模型)。")
                 final_model_path_to_evaluate = model_save_path

        print(f"--- {args.train_mode.upper()} 訓練完成。最終評估模型路徑: {final_model_path_to_evaluate} ---")
    
    else: 
        if not args.trained_model_path:
            print("錯誤: 要求跳過訓練 (--skip_training)，但未透過 --trained_model_path 提供已訓練模型路徑。")
            parser.print_help()
            sys.exit(1)
        print(f"--- 跳過訓練階段。將使用模型: {final_model_path_to_evaluate} 進行評估 ---")

    if args.skip_evaluation:
        print("--- 跳過評估階段 --- ")
        sys.exit(0)

    if not args.test_l_dir or not args.test_color_dir:
        print("錯誤: 進行評估需要 --test_l_dir 和 --test_color_dir 參數。")
        parser.print_help()
        sys.exit(1)
        
    if not final_model_path_to_evaluate or not os.path.exists(final_model_path_to_evaluate):
        print(f"錯誤: 找不到要評估的模型檔案: '{final_model_path_to_evaluate}'. 請檢查路徑，或確認訓練是否已成功執行並儲存模型。")
        sys.exit(1)
        
    print("\n--- 開始評估階段 ---")
    evaluate_model(
        model_path=final_model_path_to_evaluate,
        test_l_folder=args.test_l_dir,
        test_color_folder=args.test_color_dir,
        results_save_dir=args.eval_results_dir
    )
    print("--- 評估完成 ---")
    print(f"評估結果已儲存於: {os.path.abspath(args.eval_results_dir)}")

# === 如何執行 ===
# 1. 訓練 U-Net 並評估:
# python main.py --train_mode unet --train_bw_dir path/to/train_bw --train_color_dir path/to/train_color --model_type best_version --epochs 10 --batch_size 2 --lr 0.0001 --loss_type mae --test_l_dir path/to/test_bw --test_color_dir path/to/test_color --eval_results_dir evaluation_output_unet
#
# 2. 訓練 GAN 並評估 (生成器是 best_version):
# python main.py --train_mode gan --train_bw_dir path/to/train_bw --train_color_dir path/to/train_color --model_type best_version --epochs 50 --batch_size 1 --lr 0.0002 --loss_type mae --lambda_l1 100 --lr_discriminator 0.0002 --lr_generator_gan 0.0002 --test_l_dir path/to/test_bw --test_color_dir path/to/test_color --eval_results_dir evaluation_output_gan
#
# 3. 僅評估已訓練的模型:
# python main.py --skip_training --trained_model_path trained_models/model_gan_best_version_lossmae_lr0p0002_final_generator.h5 --test_l_dir path/to/test_bw --test_color_dir path/to/test_color --eval_results_dir evaluation_output_gan_eval_only
#
# 請將 path/to/... 替換為您實際的資料夾路徑。 