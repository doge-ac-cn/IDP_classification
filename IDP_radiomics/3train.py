import os
import numpy as np
import SimpleITK as sitk
# import radiomics
# from radiomics import featureextractor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
### KNN Classifier    
from sklearn.neighbors import KNeighborsClassifier
 
### Logistic Regression Classifier    
from sklearn.linear_model import LogisticRegression
 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import  AdaBoostClassifier
 

from sklearn.svm import SVC
 

from sklearn import tree
import argparse
import lightgbm as lgb
import xgboost as xgb


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,recall_score,precision_score, confusion_matrix, roc_curve,ConfusionMatrixDisplay ,auc
import copy
import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LassoCV

# 在十折交叉验证的每一折中使用Lasso进行特征选择涉及到在每一折中独立地应用Lasso，然后用筛选出的特征来训练模型。这样可以确保特征选择过程不会泄露验证集的信息。

mpl.rcParams["font.sans-serif"] = ['SimHei']

random_seed = 2024
def get_model(model_name):
    if model_name == 'KNN':
        clf = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(random_state=random_seed)
        param_grid = {'n_estimators': [50,100, 200], 'max_depth': [None, 10, 20]}
    elif model_name == 'LogisticRegressor':
        clf = LogisticRegression(max_iter=100000, penalty='l2', random_state=random_seed)
        param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    elif model_name == 'DecisionTree':
        clf = tree.DecisionTreeClassifier(random_state=random_seed)
        param_grid = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'AdaBoost':
        clf = AdaBoostClassifier(random_state=random_seed)
        param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1]}
    elif model_name == 'SVM':
        clf = SVC(random_state=random_seed, kernel='rbf', probability=True)
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    elif model_name == "GBDT":
        clf = GradientBoostingClassifier(random_state=random_seed)
        param_grid = {'n_estimators':  [50, 100, 200, 300, 400], 'learning_rate': [0.01, 0.1]}
    elif model_name == 'LightGBM':
        clf = lgb.LGBMClassifier(random_state=random_seed)
        param_grid = {'n_estimators':  [20,50, 100, 200, 300, 400], 'learning_rate': [0.0001,0.01, 0.1], 'num_leaves': [10,31, 50, 100], 'max_depth': [1,3, 6, 9,None]}
    elif model_name == 'XGBoost':
        clf = xgb.XGBClassifier(random_state=random_seed)
        param_grid = {'n_estimators':  [20,50, 100, 200, 300, 400], 'learning_rate': [0.0001,0.01, 0.1], 'max_depth': [1,3, 6, 9,None]}
    return clf, param_grid

def lasso_feature_selection(X_train, y_train, X_test,y_test):
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    alphas = np.logspace(-10, 1, 100, base = 10)
    selector_lasso = LassoCV(alphas=alphas, cv = 10, max_iter = 1000000,random_state=random_seed)
    
    selector_lasso.fit(X_train_scaled,y_train)
    selected_features = selector_lasso.coef_ != 0
    
    print (X_train_scaled[:,selected_features].shape)
    return X_train_scaled[:,selected_features],  X_test_scaled[:,selected_features]

def select_feature(select_fuction,X_train):
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if select_fuction == '无':
        return X_train_scaled

    elif select_fuction == 'PCA':
        pca = PCA()   #实例化
        pca = pca.fit(X_train_scaled) #拟合模型
        X_train_scaled = pca.transform(X_train_scaled)

    elif  select_fuction =='t-SNE':
        tsne = TSNE()  # 指定要降到的维度
        X_train_scaled = tsne.fit_transform(X_train_scaled)
    
    # print (f"{len(X_train_scaled[0])}")
    return X_train_scaled
    


label_dict = {'良性': 0, '恶性': 1}
label_name_dict = {v: k for k, v in label_dict.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname_list', default=['KNN', 'LogisticRegressor', 'DecisionTree', 'SVM', 'RandomForest', 'AdaBoost','XGBoost'], nargs='+')
    parser.add_argument('--select_fuction_list', default=['无','Lasso'], nargs='+')

    args = parser.parse_args()
    modelname_list = args.modelname_list
    select_fuction_list = args.select_fuction_list





    for select_fuction in select_fuction_list:
        os.makedirs(os.path.join(select_fuction), exist_ok=True)
        folders_data = []
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        X, y, patients = np.load(f'feature.npy'), np.load(f'label.npy'), np.load(f'patients.npy')

        # 获取患者名，用于随机，以避免同一患者的不同ROI分别出现在训练集和测试集，产生数据泄露
        patients_names = np.unique(patients)
        print ('患者名',patients_names)
        X = select_feature(select_fuction, X)
        for patient_name_train_index, patient_name_test_index in kfold.split(patients_names):
            
            # 获取患者名所在的索引
            train_index = [i for i, item in enumerate(patients) if item in patients_names[patient_name_train_index]]
            test_index = [i for i, item in enumerate(patients) if item in patients_names[patient_name_test_index]]

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            if select_fuction == 'Lasso':

                X_train, X_test = lasso_feature_selection(X_train, y_train, X_test,y_test)
            folders_data.append([X_train, X_test,y_train, y_test])


        for model_name in modelname_list:
            
            print(f" 特征降维方法：{select_fuction}，模型名：{model_name}")
            

            
            clf = get_model(model_name)

            accuracies = []
            f1_scores = []
            auc_scores = []
            recalls = []
            specificities = []

            y_true_all = []
            y_pred_all = []
            y_pred_class_all = []



            
            
            for folder_data in folders_data:
                X_train, X_test,y_train, y_test = folder_data
                clf, param_grid = get_model(model_name)
                grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,  scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                best_params = grid_search.best_params_
                best_estimator = grid_search.best_estimator_

                print(f"最佳参数: {best_params}")
                # print(f"最佳模型: {best_estimator}")
                
                # clf.fit(X_train, y_train)
                # y_pred = clf.predict_proba(X_test)
                y_pred = best_estimator.predict_proba(X_test)
                y_class_pred = np.argmax(y_pred, axis=1)

                accuracy = accuracy_score(y_test, y_class_pred)
                f1 = f1_score(y_test, y_class_pred, average='macro')
                if y_pred.shape[1] == 2 :
                    auc_score = roc_auc_score(y_test, y_pred[:,1], multi_class='ovr')
                else:
                    auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')
                weighted_recall = recall_score(y_test, y_class_pred, average='macro')
                confusion = confusion_matrix(y_test, y_class_pred)

                TN = np.zeros(len(label_dict))
                FP = np.zeros(len(label_dict))

                for i in range(len(label_dict)):
                    non_i_indices = [j for j in range(len(label_dict)) if j != i]
                    TN[i] = np.sum(confusion[non_i_indices, :][:, non_i_indices])
                    FP[i] = np.sum(confusion[non_i_indices, i])

                TNR = TN / (TN + FP)
                weighted_specificity = np.sum(TNR * (TN + FP)) / np.sum(TN + FP)

                accuracies.append(accuracy)
                f1_scores.append(f1)
                auc_scores.append(auc_score)
                recalls.append(weighted_recall)
                specificities.append(weighted_specificity)

                y_true_all.extend(y_test)
                y_pred_class_all.extend(y_class_pred)
                y_pred_all .extend(y_pred)

            accuracy_all = accuracy_score(y_true_all, y_pred_class_all)
            f1_all = f1_score(y_true_all, y_pred_class_all, average='macro')
            if y_pred.shape[1] == 2 :
                auc_score_all = roc_auc_score(y_true_all, [pred[1] for pred in y_pred_all], multi_class='ovr')    
            else:
                auc_score_all = roc_auc_score(y_true_all, y_pred_all, multi_class='ovr')
            weighted_recall_all = recall_score(y_true_all, y_pred_class_all, average='macro')

            confusion = confusion_matrix(y_true_all, y_pred_class_all)

            for i in range(len(label_dict)):
                non_i_indices = [j for j in range(len(label_dict)) if j != i]
                TN[i] = np.sum(confusion[non_i_indices, :][:, non_i_indices])
                FP[i] = np.sum(confusion[non_i_indices, i])

            TNR = TN / (TN + FP)
            weighted_specificity_all = np.sum(TNR * (TN + FP)) / np.sum(TN + FP)
            print(f"Accuracy: {accuracy_all:.4f} ± {np.std(accuracies):.4f}")
            print(f"F1 Score: {f1_all:.4f} ± {np.std(f1_scores):.4f}")
            print(f"Sensitivity: {weighted_recall_all:.4f} ± {np.std(recalls):.4f}")
            print(f"Specificity: { weighted_specificity_all:.4f} ± {np.std(specificities):.4f}")
            print(f"AUC: {auc_score_all:.4f} ± {np.std(auc_scores):.4f}")

            class_labels = [label for label, index in sorted(label_dict.items(), key=lambda item: item[1])]

            ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_class_all, cmap='Blues', display_labels=class_labels)
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title(f'{model_name}')
            plt.savefig(f'{select_fuction}/confusion_matrix_{model_name}.png')
            plt.clf()
            plt.close()

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            y_pred_all = np.array(y_pred_all)
            for i in range(len(np.unique(y))):
                fpr[i], tpr[i], _ = roc_curve(np.array(y_true_all) == i, y_pred_all[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure()
            for i in range(len(np.unique(y))):
                plt.plot(fpr[i], tpr[i], lw=2, label='{0}类别的ROC曲线 (AUC = {1:0.2f})'.format(label_name_dict[i], roc_auc[i]))
            
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title(f'{model_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'{select_fuction}/multi_class_roc_curve_{model_name}.png')
            plt.clf()
            plt.close()

if __name__ == "__main__":
    main()