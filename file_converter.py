import os
from PIL import Image
from tqdm import tqdm

def convert_to_bw(input_folder, output_folder, target_size=(512, 512)):
    """
    將指定資料夾中的彩色圖像轉換為黑白圖像，並儲存到另一個資料夾。
    所有圖像將被調整大小到 target_size。

    Args:
        input_folder (str): 包含彩色圖像的資料夾路徑。
        output_folder (str): 儲存黑白圖像的資料夾路徑。
        target_size (tuple): (寬度, 高度) 目標圖片尺寸。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for filename in tqdm(image_files, desc="轉換圖像中"):
        try:
            img_path = os.path.join(input_folder, filename)
            # 打開圖像，轉換為灰階，然後調整大小
            img = Image.open(img_path).convert('L').resize(target_size, Image.Resampling.LANCZOS)
            
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(output_folder, f"{name}_bw{ext}")
            img.save(save_path)
        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")

if __name__ == '__main__':
    # 假設彩色圖片放在 '1000img-paul' 資料夾下
    # 假設您希望將黑白圖片儲存在 'bw_images_512' 資料夾下
    # 您可以根據您的實際情況修改這些路徑
    source_folder = 'daterule' 
    destination_folder = 'daterule-img_bw' # 建議更改輸出資料夾名稱以區分
    image_target_size = (512, 512)

    if not os.path.exists(source_folder):
        print(f"錯誤：找不到來源資料夾 '{source_folder}'。請確認路徑是否正確。")
    else:
        print(f"將 '{source_folder}' 中的圖片轉換為黑白並調整大小為 {image_target_size}，儲存至 '{destination_folder}'")
        convert_to_bw(source_folder, destination_folder, target_size=image_target_size)
        print(f"圖像轉換完成，黑白圖像已儲存至 '{destination_folder}' 資料夾。") 