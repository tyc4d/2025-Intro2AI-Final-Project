from warnings import warn
import argparse
import sys
from PIL import Image

from ._cpu_strategy import get_ssim_sum as cpu_strategy

_gpu_available = True
_msg = ""

try:
    from ._gpu_strategy import get_ssim_sum as gpu_strategy
except Exception as e:
    _msg = str(e)
    _gpu_available = False


# https://en.wikipedia.org/wiki/Standard_deviation#Population_standard_deviation_of_grades_of_eight_students
# https://en.wikipedia.org/wiki/Structural_similarity  # Algorithm

def compare_ssim(image_0, image_1, tile_size: int = 7, GPU: bool = True) -> float:
    """
    Compute the structural similarity between the two images.
    :param image_0: PIL Image object
    :param image_1: PIL Image object
    :param tile_size: Height and width of the image's sub-sections used
    :param GPU: If true, try to compute on GPU
    :return: Structural similarity value
    """
    # Verify input parameters
    if tile_size < 1:
        raise AttributeError('The tile_size must be 1 or greater')
    # no else
    if image_0.size != image_1.size:
        raise AttributeError('The images do not have the same resolution')
    # no else
    if image_0.mode != image_1.mode:
        raise AttributeError('The images have different color channels')
    # no else

    # constants
    dynamic_range = 255
    c_1 = (dynamic_range * 0.01) ** 2
    c_2 = (dynamic_range * 0.03) ** 2
    pixel_len = tile_size * tile_size
    width, height = image_0.size
    width = width // tile_size * tile_size
    height = height // tile_size * tile_size

    if width < tile_size or height < tile_size:
        raise AttributeError('The images are smaller than the tile_size')
    # no else

    # Select strategy
    get_ssim_sum = cpu_strategy
    if GPU:
        if _gpu_available:
            get_ssim_sum = gpu_strategy
        else:
            warn("No openCL platform (or driver) available. CPU execution is used instead. \n" + _msg,
                 category=ImportWarning,
                 stacklevel=1)
    # no else

    # Calculate mean
    return get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2) * pixel_len / (
            len(image_0.mode) * width * height)

def main():
    # 創建命令行參數解析器
    parser = argparse.ArgumentParser(description='計算兩張圖片的結構相似性指數(SSIM)')
    parser.add_argument('path1', type=str, help='第一張圖片的路徑')
    parser.add_argument('path2', type=str, help='第二張圖片的路徑')
    parser.add_argument('--tile-size', type=int, default=7, help='用於計算的圖像子區域大小 (默認: 7)')
    parser.add_argument('--cpu', action='store_true', help='強制使用CPU進行計算')
    
    # 解析命令行參數
    args = parser.parse_args()
    
    try:
        # 打開圖片
        image_0 = Image.open(args.path1)
        image_1 = Image.open(args.path2)
        
        # 計算結構相似性
        ssim_value = compare_ssim(image_0, image_1, tile_size=args.tile_size, GPU=not args.cpu)
        
        # 輸出結果
        print(f"結構相似性指數 (SSIM): {ssim_value:.6f}")
        
    except Exception as e:
        print(f"錯誤: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
