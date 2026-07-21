import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import os

def load_mat_file(file_path):
    """加载.mat文件并返回数据"""
    try:
        data = loadmat(file_path)
        # 通常ground truth数据存储在特定的键中，我们需要找到正确的键
        # 排除MATLAB的元数据键
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Available keys in {file_path}: {keys}")
        
        # 通常ground truth数据是数组，选择第一个非元数据键
        if keys:
            gt_data = data[keys[0]]
            print(f"Data shape: {gt_data.shape}, Data type: {gt_data.dtype}")
            print(f"Unique values: {np.unique(gt_data)}")
            return gt_data
        else:
            print(f"No valid keys found in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_label_image(gt_data, output_path, title):
    """
    创建label图像
    0 -> 灰色 (128, 128, 128)
    1 -> 白色 (255, 255, 255) 
    2 -> 黑色 (0, 0, 0)
    """
    if gt_data is None:
        print(f"Cannot create image for {title}: no data")
        return
    
    # 创建RGB图像
    height, width = gt_data.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 设置颜色映射
    rgb_image[gt_data == 0] = [100, 100, 100]  # 灰色
    rgb_image[gt_data == 1] = [255, 255, 255]  # 白色
    rgb_image[gt_data == 2] = [0, 0, 0]        # 黑色
    
    # 如果有其他值，可以打印警告
    unique_values = np.unique(gt_data)
    unexpected_values = [v for v in unique_values if v not in [0, 1, 2]]
    if unexpected_values:
        print(f"Warning: Found unexpected values in {title}: {unexpected_values}")
        # 将其他值设置为红色以便识别
        for val in unexpected_values:
            rgb_image[gt_data == val] = [255, 0, 0]  # 红色
    
    # 使用matplotlib绘制和保存
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(f'{title} - Ground Truth Labels\n0:Gray, 1:White, 2:Black')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 同时保存为PNG文件（不带坐标轴）
    png_path = output_path.replace('.png', '_clean.png')
    Image.fromarray(rgb_image).save(png_path)
    
    print(f"Saved label image: {output_path}")
    print(f"Saved clean PNG: {png_path}")

def get_reference_image_size(ref_image_path):
    """获取参考图像的尺寸"""
    try:
        img = Image.open(ref_image_path)
        return img.size  # (width, height)
    except Exception as e:
        print(f"Error reading reference image {ref_image_path}: {e}")
        return None

def resize_if_needed(gt_data, target_size):
    """如果需要，调整ground truth数据的尺寸以匹配参考图像"""
    if gt_data is None or target_size is None:
        return gt_data
    
    target_width, target_height = target_size
    current_height, current_width = gt_data.shape
    
    if current_width != target_width or current_height != target_height:
        print(f"Resizing from ({current_width}, {current_height}) to ({target_width}, {target_height})")
        # 使用最近邻插值来保持标签值的完整性
        from skimage.transform import resize
        resized_data = resize(gt_data, (target_height, target_width), 
                            order=0, preserve_range=True, anti_aliasing=False)
        return resized_data.astype(gt_data.dtype)
    
    return gt_data

def main():
    # 数据集路径
    dataset_base = "."
    
    # santaBarbara数据集
    print("Processing Santa Barbara dataset...")
    santa_barbara_gt_path = os.path.join(dataset_base, "santaBarbara", "barbara_gtChanges.mat")
    santa_barbara_ref_path = os.path.join(dataset_base, "santaBarbara", "barbara2013rgb.png")
    santa_barbara_output = os.path.join(dataset_base, "santaBarbara", "santa_barbara_label.png")
    
    # 加载Santa Barbara ground truth
    santa_gt = load_mat_file(santa_barbara_gt_path)
    
    # 获取参考图像尺寸
    santa_ref_size = get_reference_image_size(santa_barbara_ref_path)
    print(f"Santa Barbara reference image size: {santa_ref_size}")
    
    # 调整尺寸（如果需要）
    santa_gt = resize_if_needed(santa_gt, santa_ref_size)
    
    # 创建Santa Barbara标签图像
    create_label_image(santa_gt, santa_barbara_output, "Santa Barbara")
    
    print("\n" + "="*50 + "\n")
    
    # bayArea数据集
    print("Processing Bay Area dataset...")
    bay_area_gt_path = os.path.join(dataset_base, "bayArea", "bayArea_gtChanges2.mat")
    bay_area_ref_path = os.path.join(dataset_base, "bayArea", "bayArea2013rgb.png")
    bay_area_output = os.path.join(dataset_base, "bayArea", "bay_area_label.png")
    
    # 加载Bay Area ground truth
    bay_gt = load_mat_file(bay_area_gt_path)
    
    # 获取参考图像尺寸
    bay_ref_size = get_reference_image_size(bay_area_ref_path)
    print(f"Bay Area reference image size: {bay_ref_size}")
    
    # 调整尺寸（如果需要）
    bay_gt = resize_if_needed(bay_gt, bay_ref_size)
    
    # 创建Bay Area标签图像
    create_label_image(bay_gt, bay_area_output, "Bay Area")
    
    print("\nAll label images have been created successfully!")

if __name__ == "__main__":
    main()
