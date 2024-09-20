import os
import SimpleITK as sitk

class_dict = {'良性':0,'恶性': 1}

def dicom_series_to_nifti(dicom_folder, output_file):
    try:
        # 读取 DICOM 系列
        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)
        
        if len(dicom_series) == 0:
            print(f"未找到 DICOM 文件: {dicom_folder}")
            return
        
        # 对 DICOM 文件按 InstanceNumber 进行排序
        # sorted_dicom_series = sorted(dicom_series, key=lambda dcm_file: sitk.ReadImage(dcm_file).GetMetaData('0020|0013'))  # '0020|0013' 是 DICOM 标签中的 InstanceNumber
        sorted_dicom_series = sorted(dicom_series, key=lambda dcm_file: float(sitk.ReadImage(dcm_file).GetMetaData('0020|0032').split('\\')[2])) # '0020|0032' 是 ImagePositionPatient，用于表示图像在 z 轴上的物理位置。

        reader.SetFileNames(sorted_dicom_series)
        
        # 读取 DICOM 图像数据并合并为一个 3D 图像
        image = reader.Execute()
        
        # 将图像保存为 NIfTI 文件
        sitk.WriteImage(image, output_file)
        print(f"保存成功: {output_file}")
    except Exception as e:
        print(f"转换失败: {dicom_folder}, 错误: {str(e)}")

def process_subfolders(patient_folder, output_patient_folder):
    # 遍历患者的子文件夹
    for subfolder in os.listdir(patient_folder):
        subfolder_path = os.path.join(patient_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # 确保输出文件夹存在
            os.makedirs(output_patient_folder, exist_ok=True)
            
            # 输出 NIfTI 文件的路径
            nii_filename = f"{subfolder}.nii"
            output_file = os.path.join(output_patient_folder, nii_filename)
            
            # 将当前子文件夹中的 DICOM 文件合并为一个 NIfTI 文件
            dicom_series_to_nifti(subfolder_path, output_file)

def process_dataset(dataset_dir, output_dir):
    # 遍历"良性"和"恶性"文件夹
    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)
        
        if os.path.isdir(class_path):
            output_class_folder = os.path.join(output_dir, str(class_dict[class_folder]))
            
            # 遍历良性/恶性的每个患者文件夹
            for patient_folder in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_folder)
                
                if os.path.isdir(patient_path):
                    output_patient_folder = os.path.join(output_class_folder, patient_folder)
                    
                    # 处理患者文件夹中的子文件夹
                    process_subfolders(patient_path, output_patient_folder)

# 设置输入数据集和输出目录
dataset_dir = 'dataset'  # 原始 DICOM 数据集的根目录
output_dir = 'dataset_nii'  # 输出 NIfTI 文件的根目录

# 开始处理数据集
process_dataset(dataset_dir, output_dir)
