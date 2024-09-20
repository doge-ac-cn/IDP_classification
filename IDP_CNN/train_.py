import logging
import os
import sys
import random
from tqdm import trange
import nibabel as nib
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from monai.transforms import CenterSpatialCropd
import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, RandAffined, RandAdjustContrast, RandGridPatchd, SpatialPadd
from monai.transforms import (
    Compose, LoadImaged, SpatialPadd, RandAffined, RandFlipd, RandGaussianNoised, 
    RandGaussianSmoothd, RandScaleIntensityd, RandShiftIntensityd, CenterSpatialCropd
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



random_seed = 2024
seed_everything(random_seed)
max_epoch = 300
data_path = f'cropped_2d'

def CrossEntropyLoss_label_smooth(outputs, targets, num_classes=2, epsilon=0.01):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
    targets = targets.data.cpu()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    log_prob = torch.nn.functional.log_softmax(outputs, dim=1).cpu()
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

def compute_metrics(y_true, y_pred):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    acc = accuracy_score(y_true_np, y_pred_np)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    return acc, sensitivity, specificity

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images_all = []
    patients_paths = []

    
    folder_index = 0 
    accuracy_list = []
    auc_list = []
    test_loss_list = []
    sensitivity_list = []
    specificity_list = []
    

    for patient  in os.listdir(data_path) :
        patient_path = os.path.join (data_path,patient)
        
        for nii_name in os.listdir(patient_path):
            images_all.append(os.path.join(patient_path,nii_name))
            patients_paths .append(patient_path)


    # 2 binary labels for gender classification: man and woman
    labels_all = [ os.path.basename(img) [0] for img in images_all  ]
    

    patients_names = np.unique(patients_paths)
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    
    folder_index = 0 
    accuracy_list = []
    auc_list = []
    for patient_name_train_index, patient_name_test_index in kfold.split(patients_names):
        
        # 获取患者名所在的索引
        train_index = [i for i, item in enumerate(patients_paths) if item in patients_names[patient_name_train_index]]
        test_index = [i for i, item in enumerate(patients_paths) if item in patients_names[patient_name_test_index]]


        image_train, image_test = [images_all[index] for index in train_index], [images_all[index] for index in test_index]

        label_train, label_test =[ int(labels_all[index]) for index in train_index] , [ int(labels_all[index]) for index in test_index]  
        
        print (image_test,label_test)
        train_files = [{"img": img, "label": label} for img, label in zip(image_train, label_train)]
        val_files = [{"img": img, "label": label} for img, label in zip(image_test, label_test)]


        print ( len(train_files),len(val_files))
        train_transforms = Compose([
            LoadImaged(keys=["img"], ensure_channel_first=True),
            RandAffined(
                keys=["img"],
                prob=1.0,
                shear_range=(0.1, 0.1),
                translate_range=(0.1, 0.1),
                rotate_range=(0.1, 0.1),
                scale_range=(0.1, 0.1),
                padding_mode="zeros",
            ),
            # 随机加入高斯噪声
            RandGaussianNoised(keys=["img"], prob=0.1, mean=0.0, std=0.1),
            
            # 随机进行高斯模糊
            RandGaussianSmoothd(keys=["img"], prob=0.1, sigma_x=(0.9, 1.1), sigma_y=(0.9, 1.1) ),
            CenterSpatialCropd(keys=["img"], roi_size=(320, 320)),
            # SpatialPadd(keys=["img"], spatial_size=(320, 320)),
        ])

        val_transforms = Compose([
            LoadImaged(keys=["img"], ensure_channel_first=True),
            CenterSpatialCropd(keys=["img"], roi_size=(320, 320)),
            # SpatialPadd(keys=["img"], spatial_size=(320, 320)),
        ])

        post_pred = Compose([Activations(softmax=True)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        # 保存检查检查用的nii
        check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        check_loader = DataLoader(check_ds, batch_size=1, num_workers=12, pin_memory=False)
        check_data = monai.utils.misc.first(check_loader)
        print('check data',check_data["img"].shape, check_data["label"])
        img_data = check_data["img"].cpu().numpy()
        nii_img = nib.Nifti1Image(img_data[0], affine=np.eye(4))
        nib.save(nii_img, f"check_image.nii")

        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=20, pin_memory=torch.cuda.is_available())
        
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=32, num_workers=20, pin_memory=torch.cuda.is_available())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=channel_num, out_channels=2).to(device)
        # model = monai.networks.nets.resnet50(spatial_dims=2, n_input_channels=channel_num, num_classes=2).to(device)
        model = monai.networks.nets.EfficientNetBN("efficientnet-b0",spatial_dims=2, in_channels=1, num_classes=2).to(device)
        loss_function = CrossEntropyLoss_label_smooth
        optimizer = torch.optim.SGD(model.parameters(), 1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch - max_epoch * 0.2, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=max_epoch * 0.2, after_scheduler=scheduler)

        auc_metric = ROCAUCMetric()

        val_interval = 5
        best_metric = -1
        best_metric_epoch = -1
        writer = SummaryWriter()

        for epoch in trange(max_epoch):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epoch}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar(f"{folder_index}/train_loss", loss.item(), epoch_len * epoch + step + epoch_len * max_epoch * folder_index)
            scheduler.step()
            epoch_loss /= len(train_loader)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, learning rate {optimizer.param_groups[0]['lr']}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    val_epoch_loss = 0
                    val_step = 0
                    for val_data in val_loader:
                        val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                        outputs = model(val_images)

                        # 计算损失
                        val_loss = loss_function(outputs, val_labels)
                        val_epoch_loss += val_loss.item()
                        val_step += 1
                        y_pred = torch.cat([y_pred, outputs], dim=0)
                        y = torch.cat([y, val_labels], dim=0)
                    # 计算平均测试损失
                    avg_val_loss = val_epoch_loss / len(val_loader)

                    acc_metric, sensitivity, specificity = compute_metrics(y, y_pred.argmax(dim=1))
                    y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                    y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                    auc_metric(y_pred_act, y_onehot)
                    auc_result = auc_metric.aggregate()
                    auc_metric.reset()
                    del y_pred_act, y_onehot

                    if acc_metric > best_metric:
                        best_metric = acc_metric
                        best_metric_epoch = epoch + 1

                    print(
                        "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} current loss : {:.4f} ,best accuracy: {:.4f} at epoch {}".format(
                            epoch + 1, acc_metric, auc_result,avg_val_loss ,best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar(f"{folder_index}/val_accuracy", acc_metric, epoch + 1 + max_epoch * folder_index)
                    writer.add_scalar(f"{folder_index}/val_auc", auc_result, epoch + 1 + max_epoch * folder_index)
                    writer.add_scalar(f"{folder_index}/val_sensitivity", sensitivity, epoch + 1 + max_epoch * folder_index)
                    writer.add_scalar(f"{folder_index}/val_specificity", specificity, epoch + 1 + max_epoch * folder_index)
                    writer.add_scalar(f"{folder_index}/val_loss", avg_val_loss, epoch + 1 + max_epoch * folder_index)

        # Save predictions and labels
        np.save(f"result/predictions_fold{folder_index}_epoch{epoch + 1}.npy", y_pred.cpu().numpy())
        np.save(f"result/labels_fold{folder_index}_epoch{epoch + 1}.npy", y.cpu().numpy())

        accuracy_list.append(float(acc_metric))
        auc_list.append(float(auc_result))
        sensitivity_list.append(float(sensitivity))
        specificity_list.append(float(specificity))
        test_loss_list.append(float(epoch_loss))

        print(f"Folder {folder_index} train completed, last metric: {acc_metric}, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        torch.save(model.state_dict(), f"pth/folder{folder_index}.pth")
        print("saved model")
        folder_index += 1
        writer.close()

    print(f"average_acc: {np.array(accuracy_list).mean()} accuracy_list {accuracy_list}")
    print(f"average_auc: {np.array(auc_list).mean()} auc_list {auc_list}")
    print(f"average_sensitivity: {np.array(sensitivity_list).mean()} sensitivity_list {sensitivity_list}")
    print(f"average_specificity: {np.array(specificity_list).mean()} specificity_list {specificity_list}")
    print(f"average_test_loss: {np.array(test_loss_list).mean()} test_loss_list {test_loss_list}")

if __name__ == "__main__":
    main()