import os
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
## https://github.com/AIM-Harvard/pyradiomics
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve,ConfusionMatrixDisplay
import copy
import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from concurrent.futures import ProcessPoolExecutor
import threading

# Function to extract features from ROI
def extract_features_from_roi(image,mask,save_npy_name):
    settings = {'normalize': True}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    # extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
    features = extractor.execute(image, mask)

    # 筛选出数值和数组类型的特征，排除字符串等非数值非数组类型的特征
    filtered_features = {k: v for k, v in features.items() if isinstance(v, (int, float, np.number)) or isinstance(v, np.ndarray)}
    
    # save 
    np.save(save_npy_name,np.array(list(filtered_features.values()))) 

    print (save_npy_name,len(filtered_features.values()))


# Function to load data
def load_data(data_dir,save_dir):
    labels = []
    features = []
    patients = []


    for class_dir in os.listdir(data_dir):

        class_path = os.path.join(data_dir, class_dir)
        save_class_path = os.path.join(save_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        if not os.path.exists(save_class_path):
            os.mkdir(save_class_path)
        for patient_dir in  tqdm.tqdm(os.listdir(class_path)):
            patient_path = os.path.join(class_path, patient_dir)
            save_patient_path = os.path.join(save_class_path, patient_dir)

            if not os.path.isdir(patient_path):
                continue
            if not os.path.exists(save_patient_path):
                os.mkdir(save_patient_path)

            # Load DICOM images
            images = [os.path.join(patient_path, f) for f in os.listdir(patient_path) if f.endswith('T1C.nrrd')]
            # Load ROI masks
            masks = [os.path.join(patient_path, f) for f in os.listdir(patient_path) if 'ROI' in f ] 
            

            
            print (int(patient_dir+class_dir),'IMG : ',len(images),'ROI : ',len(masks))
            
            for mask in masks:
                save_npy_name = os.path.join(save_patient_path,os.path.basename(mask).split('.')[0]+'.npy')

                extract_features_from_roi(images[0], mask,save_npy_name)
            
                
                


    for class_dir in os.listdir(save_dir):
        save_class_path = os.path.join(save_dir, class_dir)
        for patient_dir in  tqdm.tqdm(os.listdir(save_class_path)):
            save_patient_path = os.path.join(save_class_path, patient_dir)
            
            # 加载 特征的npy文件
            feature_npy = [os.path.join(save_patient_path,f) for f in os.listdir(save_patient_path) if f.endswith('npy')]

            for feature_filename in feature_npy:
                features.append( np.load(feature_filename) )
                labels.append(int(os.path.basename(mask)[0]))
                patients.append(int(patient_dir+class_dir))

    return np.array(features), np.array(labels) , np.array(patients)
# Main function
def main():
    data_dir = './dataset_nrrd'
    save_dir = './dataset_radiomics_npy'

    # Load data
    X, y , patients_array = load_data(data_dir,save_dir)

    np.save('feature.npy',X)
    np.save('label.npy',y)
    np.save('patients.npy',patients_array)


if __name__ == "__main__":
    main()
