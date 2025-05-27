# 2025-Intro2AI-Final-Project


##  Dataset Download (Private Now)
```
kaggle datasets download tyc4dtw/comic-test
```

## Usage

```
python main.py \
    --train_bw_dir bw_images_512 \
    --train_color_dir 1000img-paul \
    --model_type best_version \
    --epochs 10 \
    --batch_size 2 \
    --lr 0.0001 \
    --test_l_dir test_data/l_channel \
    --test_color_dir test_data/color \
    --eval_results_dir my_run1_results
```

```
python main.py --skip_training \
    --trained_model_path trained_models/model_best_version_lr0p0001.h5 \
    --test_l_dir test_data/l_channel \
    --test_color_dir test_data/color \  
    --eval_results_dir my_run1_evaluation_only
```