import nibabel as nib
import numpy as np
import os
from scipy.ndimage import find_objects

def crop_roi_from_image(t1c_path, roi_paths, output_dir):
    """
    从原图中裁剪每个ROI的最小外接矩形并保存，同时显示ROI的长宽高。

    参数:
    t1c_path: 原图的路径 (T1C.nii)
    roi_paths: ROI的路径列表 ({序号}ROI.nii)
    output_dir: 保存裁剪后图像的目录
    """
    # 加载原始T1C图像
    t1c_img = nib.load(t1c_path)
    t1c_data = t1c_img.get_fdata()
    
    print(f"IMG的长: {t1c_data.shape[2]}，宽: {t1c_data.shape[1]}，高: {t1c_data.shape[0]}")
    # 遍历每个ROI文件
    for roi_path in roi_paths:
        # 加载ROI
        roi_img = nib.load(roi_path)
        roi_data = roi_img.get_fdata()
        
        # 找到ROI的最小外接矩形
        roi_bbox_slices = find_objects(roi_data.astype(int))[0]  # 提取第一个非空的切片

        # 计算ROI的长、宽、高
        z_length = roi_bbox_slices[0].stop - roi_bbox_slices[0].start  # 高度（z轴）
        y_length = roi_bbox_slices[1].stop - roi_bbox_slices[1].start  # 宽度（y轴）
        x_length = roi_bbox_slices[2].stop - roi_bbox_slices[2].start  # 长度（x轴）
        


        # 输出ROI的长宽高
        print(f"ROI文件: {os.path.basename(roi_path)}")
        print(f"ROI的长: {x_length}，宽: {y_length}，高: {z_length}")

        # 裁剪原图的T1C图像，使用ROI的bounding box
        cropped_t1c_data = t1c_data[roi_bbox_slices]

        # 创建保存裁剪后的图像文件
        cropped_img = nib.Nifti1Image(cropped_t1c_data, affine=t1c_img.affine)

        # 获取ROI的序号，用于命名输出文件
        roi_filename = os.path.basename(roi_path)

        output_path = os.path.join(output_dir, f'{roi_filename}_cropped.nii')

        # 保存裁剪后的图像
        nib.save(cropped_img, output_path)
        print(f"保存裁剪后的图像: {output_path}\n")

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
output_root_dir = './cropped_output'  # 保存裁剪结果的目录

# 处理整个数据集
process_dataset(dataset_dir, output_root_dir)
