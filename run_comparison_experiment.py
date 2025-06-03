import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

# 全域配置 (根據您的環境和需求修改)
MAIN_SCRIPT_PATH = "main.py" # main.py 的路徑
PYTHON_EXECUTABLE = "python" # 或您的 Python 解釋器路徑 (例如 /usr/bin/python3)

BASE_OUTPUT_DIR = "comparison_experiments_1000img" # 所有比較實驗結果的基礎輸出目錄

# 訓練資料路徑 (假設所有實驗使用相同的訓練資料)
TRAIN_BW_DIR = "1000img-paul_bw" # !!! 需要您設定
TRAIN_COLOR_DIR = "1000img-paul" # !!! 需要您設定

# 測試資料路徑 (假設所有實驗使用相同的測試資料)
TEST_L_DIR = "test_data/l_channel"       # !!! 需要您設定
TEST_COLOR_DIR = "test_data/color" # !!! 需要您設定

COMMON_EPOCHS = 200 # 為了快速演示，設為較小值。實際比較時應設為合理值，例如 50 或 100
COMMON_BATCH_SIZE_UNET = 4
COMMON_BATCH_SIZE_GAN = 2 # GAN 通常需要較小的 batch size
COMMON_LR_UNET = 0.0001
COMMON_LR_GAN_G = 0.0002 # 生成器在 GAN 中的學習率
COMMON_LR_GAN_D = 0.0002 # 判別器學習率
COMMON_LOSS_TYPE = "mae"
COMMON_LAMBDA_L1 = 100

# 實驗配置列表
# 每個字典代表一個實驗運行
EXPERIMENT_CONFIGS = [
    {
        "name": "PReLU_UNet",
        "train_mode": "unet",
        "model_type": "unet_advanced_prelu",
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE_UNET,
        "lr": COMMON_LR_UNET,
        "loss_type": COMMON_LOSS_TYPE,
    },
    {
        "name": "PReLU_GAN",
        "train_mode": "gan",
        "model_type": "unet_advanced_prelu", # Generator for GAN
        "epochs": COMMON_EPOCHS, # GAN 可能需要更多 epochs
        "batch_size": COMMON_BATCH_SIZE_GAN,
        "lr": COMMON_LR_UNET, # lr for G's L1 part
        "loss_type": COMMON_LOSS_TYPE, # G's L1 loss type
        "lambda_l1": COMMON_LAMBDA_L1,
        "lr_discriminator": COMMON_LR_GAN_D,
        "lr_generator_gan": COMMON_LR_GAN_G,
    },
    {
        "name": "LeakyReLU_UNet",
        "train_mode": "unet",
        "model_type": "best_version", # unet_relu_leaky
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE_UNET,
        "lr": COMMON_LR_UNET,
        "loss_type": COMMON_LOSS_TYPE,
    },
    {
        "name": "VGG16_UNet",
        "train_mode": "unet",
        "model_type": "unet_vgg16",
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE_UNET,
        "lr": COMMON_LR_UNET,
        "loss_type": COMMON_LOSS_TYPE,
    },
]

def run_experiment(config):
    """Executes a single experiment configuration."""
    exp_name = config["name"]
    print(f"\n===== Running Experiment: {exp_name} =====")

    exp_output_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
    model_output_dir = os.path.join(exp_output_dir, "trained_model")
    eval_results_dir = os.path.join(exp_output_dir, "evaluation_results")

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)

    lr_str = str(config.get("lr", COMMON_LR_UNET)).replace('.','p')
    model_filename_base = f"model_{config['train_mode']}_{config['model_type']}_loss{config['loss_type']}_lr{lr_str}"
    model_save_filename = f"{model_filename_base}.h5"
    
    model_output_path = os.path.join(model_output_dir, model_save_filename)

    if config['train_mode'] == 'gan':
        final_generator_filename = f"{model_filename_base}_final_generator.h5"
        trained_model_path_for_eval = os.path.join(model_output_dir, final_generator_filename)
    else:
        trained_model_path_for_eval = model_output_path

    cmd = [
        PYTHON_EXECUTABLE, MAIN_SCRIPT_PATH,
        "--train_mode", config["train_mode"],
        "--train_bw_dir", TRAIN_BW_DIR,
        "--train_color_dir", TRAIN_COLOR_DIR,
        "--model_type", config["model_type"],
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--lr", str(config.get("lr", COMMON_LR_UNET)),
        "--loss_type", config["loss_type"],
        "--model_output_template", model_output_path, 
        
        "--test_l_dir", TEST_L_DIR,
        "--test_color_dir", TEST_COLOR_DIR,
        "--eval_results_dir", eval_results_dir,
    ]

    if config["train_mode"] == "gan":
        cmd.extend([
            "--lambda_l1", str(config.get("lambda_l1", COMMON_LAMBDA_L1)),
            "--lr_discriminator", str(config.get("lr_discriminator", COMMON_LR_GAN_D)),
            "--lr_generator_gan", str(config.get("lr_generator_gan", COMMON_LR_GAN_G)),
        ])
    
    print(f"Executing command: {' '.join(cmd)}")
    stdout_lines = []
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='') 
            stdout_lines.append(line)
        process.wait() 
        full_stdout = "".join(stdout_lines)

        if process.returncode != 0:
            print(f"Error in experiment {exp_name}! Return code: {process.returncode}")
            return None, full_stdout
        else:
            print(f"Experiment {exp_name} completed.")
            metrics_file = os.path.join(eval_results_dir, "metrics.json")
            avg_psnr, avg_ssim = None, None
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    avg_psnr = metrics_data.get("avg_psnr")
                    avg_ssim = metrics_data.get("avg_ssim")
            except FileNotFoundError:
                print(f"Error: Metrics file not found {metrics_file}. Check if main.py/utils.py saved it correctly.")
            except Exception as e:
                print(f"Error reading metrics file {metrics_file}: {e}")
            
            return {
                "name": exp_name, 
                "psnr": avg_psnr, 
                "ssim": avg_ssim, 
                "eval_dir": eval_results_dir, 
                "model_path_for_eval": trained_model_path_for_eval 
            }, full_stdout

    except Exception as e:
        print(f"Exception during experiment {exp_name}: {e}")
        return None, str(e) 


def plot_metrics_comparison(results, output_dir):
    """Plots a bar chart comparing model performance metrics (PSNR & SSIM)."""
    if not results:
        print("No results to plot for metrics comparison.")
        return

    valid_results = [res for res in results if res and res.get("psnr") is not None and res.get("ssim") is not None]
    if not valid_results:
        print("No valid metric data to plot for comparison.")
        return

    names = [res["name"] for res in valid_results]
    psnr_scores = [res["psnr"] for res in valid_results]
    ssim_scores = [res["ssim"] for res in valid_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(10, len(names) * 1.5), 7))

    rects1 = ax1.bar(x - width/2, psnr_scores, width, label='Average PSNR (dB)', color='deepskyblue')
    ax1.set_ylabel('Average PSNR (dB)', color='deepskyblue')
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_xlabel("Model Configuration")

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}', 
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  
                     textcoords="offset points",
                     ha='center', va='bottom')

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, ssim_scores, width, label='Average SSIM', color='mediumseagreen')
    ax2.set_ylabel('Average SSIM', color='mediumseagreen')
    ax2.tick_params(axis='y', labelcolor='mediumseagreen')
    min_ssim = min(ssim_scores) if ssim_scores else 0
    max_ssim = max(ssim_scores) if ssim_scores else 1
    ax2.set_ylim(min(0, min_ssim - 0.05), max(1, max_ssim + 0.05))


    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.4f}', 
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  
                     textcoords="offset points",
                     ha='center', va='bottom')

    fig.suptitle('Model Performance Comparison (Average PSNR & SSIM)', fontsize=16)
    
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

    plt.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.9) # Fine-tune spacing for labels
    
    plot_path = os.path.join(output_dir, "metrics_comparison_chart.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Metrics comparison chart saved to: {plot_path}")

def plot_visual_comparison_grid(results, num_samples=5, output_dir="visual_comparison_output"):
    """Plots a grid of visual comparisons for selected samples across all models."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_results = [res for res in results if res and res.get("eval_dir") and os.path.exists(res.get("eval_dir"))]
    if not valid_results:
        print("No valid experiment results for visual comparison (eval_dir missing or empty).")
        return

    l_image_files_all = sorted([f for f in os.listdir(TEST_L_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not l_image_files_all:
        print(f"Error: No L-channel test images found in {TEST_L_DIR}.")
        return
    
    num_actual_samples = min(num_samples, len(l_image_files_all))
    if num_actual_samples == 0:
        print("No samples to display for visual comparison.")
        return
    selected_l_fnames_indices = np.random.choice(len(l_image_files_all), num_actual_samples, replace=False)
    selected_l_fnames = [l_image_files_all[i] for i in selected_l_fnames_indices]

    num_models = len(valid_results)
    num_cols_per_sample = num_models + 2  # Input L, Model outputs, Ground Truth
    
    fig_height_per_sample = 4 # Inches per sample row
    fig_width_per_col = 4   # Inches per image column
    fig, axes = plt.subplots(num_actual_samples, num_cols_per_sample, 
                             figsize=(fig_width_per_col * num_cols_per_sample, fig_height_per_sample * num_actual_samples),
                             squeeze=False) # squeeze=False ensures axes is always 2D

    fig.suptitle('Visual Comparison of Model Outputs', fontsize=20, y=0.995)

    # Column Titles (at the very top of the grid)
    axes[0, 0].set_title("Input L-channel", fontsize=12, pad=10)
    for model_idx, res_dict in enumerate(valid_results):
        axes[0, model_idx + 1].set_title(res_dict["name"], fontsize=12, pad=10)
    axes[0, num_cols_per_sample - 1].set_title("Ground Truth", fontsize=12, pad=10)

    for sample_idx, l_fname in enumerate(selected_l_fnames):
        # Row title (filename for the sample)
        # axes[sample_idx, 0].set_ylabel(l_fname, rotation=0, size='large', labelpad=40, ha='right', va='center')
        # Using a text annotation for row labels might be more flexible
        fig.text(0.01, (num_actual_samples - 1 - sample_idx + 0.5) / num_actual_samples, os.path.basename(l_fname), 
                 va='center', ha='left', rotation='vertical', fontsize=10) # Adjust x,y,fontsize as needed

        # 1. Display Input L-channel image
        l_path = os.path.join(TEST_L_DIR, l_fname)
        try:
            l_img = Image.open(l_path).convert('L').resize((512, 512))
            axes[sample_idx, 0].imshow(l_img, cmap='gray')
        except Exception as e:
            print(f"Cannot load L image {l_path}: {e}")
            axes[sample_idx, 0].text(0.5, 0.5, "Input L (Error)", ha='center', va='center')
        axes[sample_idx, 0].axis('off')

        # 2. Display each model's prediction
        for model_idx, res_dict in enumerate(valid_results):
            eval_dir = res_dict["eval_dir"]
            # Predicted image is saved in eval_dir with the same name as l_fname by utils.py's plot_results
            pred_img_path = os.path.join(eval_dir, ("comparison_"+l_fname))
            try:
                pred_img = Image.open(pred_img_path).convert('RGB') # Assumed to be 512x512
                axes[sample_idx, model_idx + 1].imshow(pred_img)
            except FileNotFoundError:
                print(f"Predicted image not found: {pred_img_path} for visual comparison grid.")
                axes[sample_idx, model_idx + 1].text(0.5, 0.5, "Output (Not Found)", ha='center', va='center')
            except Exception as e:
                print(f"Cannot load predicted image {pred_img_path}: {e}")
                axes[sample_idx, model_idx + 1].text(0.5, 0.5, "Output (Load Error)", ha='center', va='center')
            axes[sample_idx, model_idx + 1].axis('off')

        # 3. Display Ground Truth RGB image
        base_name_l, _ = os.path.splitext(l_fname)
        original_color_name_stem = base_name_l[:-3] if base_name_l.endswith("_bw") else base_name_l
        gt_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_gt_path = os.path.join(TEST_COLOR_DIR, f"{original_color_name_stem}{ext}")
            if os.path.exists(potential_gt_path):
                gt_path = potential_gt_path
                break
        
        if gt_path:
            try:
                gt_img = Image.open(gt_path).convert('RGB').resize((512, 512))
                axes[sample_idx, num_cols_per_sample - 1].imshow(gt_img)
            except Exception as e:
                print(f"Cannot load GT image {gt_path}: {e}")
                axes[sample_idx, num_cols_per_sample - 1].text(0.5, 0.5, "GT (Error)", ha='center', va='center')
        else:
            print(f"GT image not found for {l_fname} (stem: {original_color_name_stem}) in {TEST_COLOR_DIR}")
            axes[sample_idx, num_cols_per_sample - 1].text(0.5, 0.5, "GT (Not Found)", ha='center', va='center')
        axes[sample_idx, num_cols_per_sample - 1].axis('off')

    plt.tight_layout(rect=[0.03, 0, 1, 0.97]) # Adjust rect for row labels and suptitle
    # plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.02, hspace=0.1, wspace=0.05)
    
    grid_plot_path = os.path.join(output_dir, "all_visual_comparisons_grid.png")
    plt.savefig(grid_plot_path)
    plt.close(fig)
    print(f"Visual comparison grid saved to: {grid_plot_path}")


if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Check paths before running, converted to use startswith for flexibility
    if any(p.startswith("path/to/your") for p in [TRAIN_BW_DIR, TRAIN_COLOR_DIR, TEST_L_DIR, TEST_COLOR_DIR]):
        print("Error: Please update TRAIN_BW_DIR, TRAIN_COLOR_DIR, TEST_L_DIR, and TEST_COLOR_DIR paths in the script.")
        exit(1)

    all_results_summary = []
    full_log_path = os.path.join(BASE_OUTPUT_DIR, "all_experiments_run.log")
    
    print(f"Experiment logs will be written to: {full_log_path}")

    with open(full_log_path, "w", encoding='utf-8') as log_file:
        for config_idx, config in enumerate(EXPERIMENT_CONFIGS):
            exp_name = config["name"]
            log_file.write(f"\n\n{'='*20} Experiment {config_idx+1}/{len(EXPERIMENT_CONFIGS)}: {exp_name} {'='*20}\n")
            log_file.flush()

            result_summary, stdout_log = run_experiment(config)
            
            log_file.write(f"--- STDOUT/STDERR for {exp_name} ---\n")
            log_file.write(stdout_log if stdout_log else "No stdout/stderr captured.\n")
            log_file.write(f"--- End of STDOUT/STDERR for {exp_name} ---\n")
            log_file.flush()

            if result_summary and result_summary.get("psnr") is not None and result_summary.get("ssim") is not None:
                all_results_summary.append(result_summary)
                log_file.write(f"Experiment {exp_name} results: PSNR={result_summary['psnr']:.2f}, SSIM={result_summary['ssim']:.4f}\n")
            else:
                log_file.write(f"Experiment {exp_name} did not complete successfully or metrics were not found.\n")
            log_file.flush()

    print(f"\nAll experiment logs saved to: {full_log_path}")

    if all_results_summary:
        plot_metrics_comparison(all_results_summary, BASE_OUTPUT_DIR)
        
        visual_comp_gallery_dir = os.path.join(BASE_OUTPUT_DIR, "visual_comparisons_gallery")
        plot_visual_comparison_grid(all_results_summary, num_samples=5, output_dir=visual_comp_gallery_dir)
    else:
        print("No successful experiment results to generate comparison charts. Please check the log file.")

    print(f"\nComparison experiments finished. All results are in: {os.path.abspath(BASE_OUTPUT_DIR)}") 