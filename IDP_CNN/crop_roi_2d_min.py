import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt  # 用于保存为JPEG
from scipy.ndimage import find_objects
from PIL import Image

def save_as_jpg(image_data, output_path):
    """
    将2D图像数据保存为JPEG格式。

    参数:
    image_data: 2D图像数据 (numpy数组)
    output_path: 输出的JPEG文件路径
    """
    # 归一化图像数据以适应JPEG格式
    norm_image_data = 255 * (image_data - np.min(image_data)) / (np.ptp(image_data) + 1e-6)
    norm_image_data = norm_image_data.astype(np.uint8)

    # 保存为JPEG图像
    im = Image.fromarray(norm_image_data)
    im.save(output_path)
    print(f"保存JPEG图像: {output_path}")

def crop_roi_from_image(t1c_path, roi_paths, output_dir):
    """
    从原图中提取ROI的xy轴面积最大的三层，并保存为单独的nii文件和JPEG图像。

    参数:
    t1c_path: 原图的路径 (T1C.nii)
    roi_paths: ROI的路径列表 ({序号}ROI.nii)
    output_dir: 保存裁剪后图像的目录
    """
    # 加载原始T1C图像
    t1c_img = nib.load(t1c_path)
    t1c_data = t1c_img.get_fdata()
    
    print(f"IMG的长: {t1c_data.shape[0]}，宽: {t1c_data.shape[1]}，高: {t1c_data.shape[2]}")
    
    # 遍历每个ROI文件
    for roi_path in roi_paths:
        # 加载ROI
        roi_img = nib.load(roi_path)
        roi_data = roi_img.get_fdata()

        # 找到每个切片中的ROI区域
        z_slices_with_roi = []
        for z_idx in range(roi_data.shape[2]):
            roi_slice = roi_data[:, :, z_idx]
            non_zero_coords = np.argwhere(roi_slice)  # 获取非零坐标
            
            if non_zero_coords.size > 0:
                # 计算最小外接矩形
                x_min, y_min = np.min(non_zero_coords, axis=0)
                x_max, y_max = np.max(non_zero_coords, axis=0)
                area = (x_max - x_min) * (y_max - y_min)
                z_slices_with_roi.append((z_idx, area, x_min, x_max, y_min, y_max))
        
        # 按xy面积排序，取最大的三层
        z_slices_sorted = sorted(z_slices_with_roi, key=lambda x: x[1], reverse=True)[:3]
        
        print(f"ROI文件: {os.path.basename(roi_path)}")
        print(f"选取的最大xy面积的z轴切片索引: {[s[0] for s in z_slices_sorted]}")
        
        # 分别保存每个最大的z轴切片的最小外接矩形
        for i, (z_idx, area, x_min, x_max, y_min, y_max) in enumerate(z_slices_sorted):
            # 裁剪T1C图像对应的切片
            cropped_t1c_data = t1c_data[x_min:x_max+1, y_min:y_max+1, z_idx]
            cropped_img = nib.Nifti1Image(cropped_t1c_data, affine=t1c_img.affine)
            
            # 输出裁剪后图像的大小
            print(f"裁剪后的图像大小 (z={z_idx}): {cropped_t1c_data.shape}")
            
            # 获取ROI的序号，用于命名输出文件
            roi_filename = os.path.basename(roi_path)
            nii_output_path = os.path.join(output_dir, f'{roi_filename}_z{z_idx}_cropped_bbox.nii')
            
            # 保存裁剪后的NIfTI图像
            nib.save(cropped_img, nii_output_path)
            print(f"保存裁剪后的NIfTI图像: {nii_output_path}")
            
            # # 保存为JPEG图像
            # jpg_output_path = os.path.join(output_dir, f'{roi_filename}_z{z_idx}_cropped_bbox.jpg')
            # save_as_jpg(cropped_t1c_data, jpg_output_path)

def find_all_files(dataset_dir):
    """
    遍历指定的 dataset 目录，找到所有的 T1C.nii 和 ROI.nii 文件。

    参数:
    dataset_dir: 数据集的根目录。

    返回:
    包含所有病例的 T1C 文件路径和相应的 ROI 文件路径的字典。
    """
    all_cases = []

    for root, dirs, files in os.walk(dataset_dir):
        # 找到 T1C.nii 和 {序号}ROI.nii 文件
        t1c_path = None
        roi_paths = []
        for file in files:
            if file.endswith('T1C.nii'):
                t1c_path = os.path.join(root, file)
            elif 'ROI' in file:
                roi_paths.append(os.path.join(root, file))
        
        if t1c_path and roi_paths:
            all_cases.append((t1c_path, roi_paths, root))

    return all_cases

def process_dataset(dataset_dir, output_root_dir):
    """
    处理整个数据集目录中的所有病例，并保存结果。

    参数:
    dataset_dir: 数据集的根目录。
    output_root_dir: 保存处理结果的根目录。
    """
    all_cases = find_all_files(dataset_dir)

    for t1c_path, roi_paths, case_dir in all_cases:
        # 为每个病例创建输出目录
        case_name = os.path.basename(case_dir)
        output_dir = os.path.join(output_root_dir, case_name)
        os.makedirs(output_dir, exist_ok=True)

        # 处理当前病例的T1C图像和ROI文件
        crop_roi_from_image(t1c_path, roi_paths, output_dir)

# 示例调用
dataset_dir = './dataset'  # 数据集根目录
output_root_dir = './cropped_2d_min'  # 保存裁剪结果的目录

# 处理整个数据集
process_dataset(dataset_dir, output_root_dir)
