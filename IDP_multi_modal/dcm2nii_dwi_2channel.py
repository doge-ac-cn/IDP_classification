import os
import pydicom
import numpy as np
import nibabel as nib

class_dict = {'良性': 0, '恶性': 1}

def dicom_series_to_nifti(dicom_folder, output_file):
    try:
        # 获取 DICOM 文件列表
        dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
        
        if len(dicom_files) == 0:
            print(f"未找到 DICOM 文件: {dicom_folder}")
            return
        
        # 按 InstanceNumber 或者 ImagePositionPatient 排序以确保顺序正确
        dicom_files = sorted(dicom_files, key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
        
        # 按照 ImagePositionPatient排序的代码
        # # 提取每个DICOM文件的ImagePositionPatient以基于z轴位置排序
        # dicom_files_with_pos = []
        # for dicom_file in dicom_files:
        #     ds = pydicom.dcmread(dicom_file)
        #     # ImagePositionPatient[2]是z轴的位置
        #     z_position = ds.ImagePositionPatient[2]
        #     dicom_files_with_pos.append((dicom_file, z_position))
        
        # # 按 z轴位置进行排序
        # dicom_files_with_pos.sort(key=lambda x: x[1])
        # dicom_files = [f[0] for f in dicom_files_with_pos]  # 获取排序后的文件列表

        # 读取第一个 DICOM 文件以获取图像信息
        dicom_sample = pydicom.dcmread(dicom_files[0])
        
        # 初始化一个空的数组，用于存储图像数据
        img_shape = (len(dicom_files), dicom_sample.Rows, dicom_sample.Columns)  # z轴, 高度, 宽度
        image_array = np.zeros(img_shape, dtype=dicom_sample.pixel_array.dtype)
        
        # 逐一读取 DICOM 文件并填充图像数组
        for i, dicom_file in enumerate(dicom_files):
            ds = pydicom.dcmread(dicom_file)
            image_array[i, :, :] = ds.pixel_array
        
        # 获取z轴的深度
        depth = image_array.shape[0]
        
        # 交替提取：偶数索引的切片和奇数索引的切片
        even_slices = image_array[::2, :, :]  # 偶数索引的切片
        odd_slices = image_array[1::2, :, :]  # 奇数索引的切片
        
        # 堆叠偶数和奇数切片为两个通道
        stacked_array = np.stack((even_slices, odd_slices), axis=-1)  # 创建 (depth/2, height, width, 2)
        
        # 转换为 (z轴, y轴, x轴, 通道)
        stacked_array = np.transpose(stacked_array, (2, 1, 0, 3))  # 调整为 (height, width, z/2, 2)
        
        # 将数组转换为 NIfTI 格式
        nifti_img = nib.Nifti1Image(stacked_array, affine=np.eye(4))
        
        # 保存为 NIfTI 文件
        nib.save(nifti_img, output_file)
        print(f"保存成功: {output_file} , {stacked_array.shape} , 来自 {dicom_folder}")
    
    except Exception as e:
        print(f"转换失败: {dicom_folder}, 错误: {str(e)}")

def process_subfolders(patient_folder, output_file):
    # 遍历患者的子文件夹，取第二个文件夹
    subfolder = os.listdir(patient_folder)[1]
    subfolder_path = os.path.join(patient_folder, subfolder)

    # 将当前子文件夹中的 DICOM 文件合并为一个 NIfTI 文件
    dicom_series_to_nifti(subfolder_path, output_file)

def process_dataset(dataset_dir, output_dir):
    # 遍历 "良性" 和 "恶性" 文件夹
    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)
        
        if os.path.isdir(class_path):            

            # 遍历良性/恶性的每个患者文件夹
            for patient_folder in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_folder)
                
                if os.path.isdir(patient_path):
                    nii_path = os.path.join(output_dir, str(class_dict[class_folder]) + patient_folder + '.nii')
                    
                    # 处理患者文件夹中的子文件夹
                    process_subfolders(patient_path, nii_path)

# 设置输入数据集和输出目录
dataset_dir = 'dataset'  # 原始 DICOM 数据集的根目录
output_dir = 'dataset_dwi_2channel'  # 输出 NIfTI 文件的根目录

# 开始处理数据集
process_dataset(dataset_dir, output_dir)
