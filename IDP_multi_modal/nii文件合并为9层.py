import os
import SimpleITK as sitk
import numpy as np

def load_nii(file_path):
    """读取 NIfTI 文件"""
    return sitk.ReadImage(file_path), sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def save_nii(array, reference_img, output_file):
    """保存 NIfTI 文件，使用参考图像来保留元数据"""
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(reference_img.GetSpacing())
    img.SetOrigin(reference_img.GetOrigin())
    img.SetDirection(reference_img.GetDirection())
    sitk.WriteImage(img, output_file)

def resample_nii(image, target_shape):
    """对 NIfTI 文件插值，确保大小匹配"""
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    resize_factor = np.array(original_size) / np.array(target_shape)
    new_spacing = [spacing * factor for spacing, factor in zip(original_spacing, resize_factor)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_shape)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(image)

def split_nii(image_array, num_splits, split_size):
    """将 NIfTI 文件沿 Z 轴拆分为多份"""
    splits = np.array_split(image_array, num_splits, axis=0)
    if len(splits[0]) != split_size:
        raise ValueError(f"每个分割部分的大小不符合预期: {split_size} 层")
    return splits

def process_patient_nii(patient_folder, output_folder):
    nii_files = [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('.nii')]
    
    if len(nii_files) != 4:
        print(f"患者 {os.path.basename(patient_folder)} 的 NIfTI 文件数量不为 4，跳过处理。")
        return

    # 加载四个 NIfTI 文件
    nii_images = [load_nii(nii_file) for nii_file in nii_files]
    nii_arrays = [img[1] for img in nii_images]
    
    # 找到最大的 NIfTI 文件
    print ([arr.shape[0] for arr in nii_arrays])
    max_dim_index = np.argmax([arr.shape[0] for arr in nii_arrays])
    max_image, max_array = nii_images[max_dim_index]

    # 将最大的 NIfTI 文件分成 6 份，每份 150 层
    split_size = 150
    max_splits = split_nii(max_array, 6, split_size)

    # 获取分割后的 NIfTI 文件的尺寸 (150, X, Y)
    split_shape = max_splits[0].shape  # 每一块的形状

    # 对其他较小的 NIfTI 文件进行插值，使其大小匹配
    resized_arrays = []
    for i, (img, arr) in enumerate(nii_images):
        if i != max_dim_index:
            # 插值到拆分后的形状
            target_shape = [split_size, *split_shape[1:]]  # 明确目标形状
            resampled_img = resample_nii(img, target_shape)
            resized_arrays.append(sitk.GetArrayFromImage(resampled_img))
        else:
            resized_arrays.extend(max_splits)  # 插入最大 NIfTI 文件的 6 份

    # 拼接9份 NIfTI 文件
    final_array = np.concatenate(resized_arrays, axis=0)

    # 保存拼接后的 NIfTI 文件
    output_patient_folder = os.path.join(output_folder, os.path.basename(patient_folder))
    os.makedirs(output_patient_folder, exist_ok=True)
    
    output_file = os.path.join(output_patient_folder, 'merged.nii')
    save_nii(final_array, max_image, output_file)
    print(f"保存拼接后的 NIfTI 文件: {output_file}")

def process_dataset_nii(dataset_nii_folder, output_folder):
    # 遍历 '0' 和 '1' 分类文件夹
    for category_folder in os.listdir(dataset_nii_folder):
        category_path = os.path.join(dataset_nii_folder, category_folder)
        if os.path.isdir(category_path):
            # 遍历分类文件夹中的每个患者文件夹
            for patient_folder in os.listdir(category_path):
                patient_path = os.path.join(category_path, patient_folder)
                if os.path.isdir(patient_path):
                    process_patient_nii(patient_path, output_folder)

# 设置输入和输出路径
dataset_nii_folder = 'dataset_nii'  # NIfTI 文件的根目录
output_folder = 'output_nii'  # 输出拼接后的 NIfTI 文件目录

# 开始处理每个患者的 NIfTI 文件
process_dataset_nii(dataset_nii_folder, output_folder)
