import os

import json
import cv2
import numpy as np
from labelme import utils
from tqdm import tqdm
import SimpleITK as sitk


import nibabel as nib
import tqdm

data_dir = './dataset'
nrrd_dir = './dataset_nrrd'



# Load data
class_index = 0 
for class_dir in os.listdir(data_dir):
    
    class_path = os.path.join(data_dir, class_dir)
    class_nrrd_path = os.path.join(nrrd_dir, str(class_index))
    print (class_path,class_nrrd_path)
    if not os.path.exists( class_nrrd_path):
            os.mkdir( class_nrrd_path)
    if not os.path.isdir(class_path):
        continue
    for patient_dir in tqdm.tqdm(os.listdir(class_path)):
        patient_path = os.path.join(class_path, patient_dir)
        patient_nrrd_path = os.path.join(class_nrrd_path, patient_dir.split(' ')[1])

        if not os.path.isdir(patient_path):
            continue

        if not os.path.exists( patient_nrrd_path):
            os.mkdir( patient_nrrd_path)

        print (patient_path,'IMG : ',len(os.listdir(patient_path)))

        for image_name in  os.listdir(patient_path):
            image_save_path = os.path.join(patient_nrrd_path,os.path.splitext(image_name)[0] + ".nrrd")
            if not os.path.exists(image_save_path):
                image_path = os.path.join(patient_path, image_name)
                image_array  = nib.load(image_path).get_fdata()
        
                image_sitk = sitk.GetImageFromArray(image_array)
            
                
                sitk.WriteImage (image_sitk,image_save_path)
            else:
                print (image_save_path)

    class_index += 1
