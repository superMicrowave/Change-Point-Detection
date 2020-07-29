import pandas as pd
import numpy as np
import glob
from function import Accuracy
import matplotlib.pyplot as plt
from data_process import folds_sorted,labels

# function to split data
def SplitFolder(labels, folders, fold_id):
    bool_suq = folders == fold_id
    train_label = labels[~bool_suq][:, 1:]
    test_label = labels[bool_suq][:, 1:]

    return train_label, test_label

# Load data
## load the realating csv file
output_path = 'Data/Outputs/'
outputs_file = 'outputs.csv'
linearModel_file = 'linearModel.csv'
cnnModel_file = 'cnnModel.csv'
L1Model_file = 'L1Model.csv'
baselineModel_file = 'baselineModel.csv'

linear_model = pd.read_csv(output_path + linearModel_file, header = None)
cnn_model = pd.read_csv(output_path + cnnModel_file, header = None)
L1_model = pd.read_csv(output_path + L1Model_file, header = None)
baseline_model = pd.read_csv(output_path + baselineModel_file, header = None)

# split data
fold_lab_list = []
for fold_num in range(1, 7):
    _, fold_lab = SplitFolder(labels, folds_sorted[:, 1], fold_num)
    fold_lab_list.append(fold_lab)

accuracy_list = []
line_test_acc = []
cnn_test_acc = []
baseline_acc = []

# calculate accuracy
for fold_num in range(6):
    num = fold_lab_list[fold_num].shape[0]
     
    L1_acc = 0
    linear_acc = 0
    cnn_acc = 0
    base_acc = 0
    for (L1, linear, cnn, base, label) in zip(L1_model.iloc[:, fold_num], 
                                   linear_model.iloc[:, fold_num], 
                                      cnn_model.iloc[:, fold_num],
                                         baseline_model.iloc[:, fold_num],
                                            fold_lab_list[fold_num]):
        label = label.reshape(2)
        L1_acc += Accuracy(L1, label)
        linear_acc += Accuracy(linear, label)
        cnn_acc += Accuracy(cnn, label)
        base_acc += Accuracy(base, label)
    
    accuracy_list.append(L1_acc/num * 100)
    line_test_acc.append(linear_acc/num * 100)
    cnn_test_acc.append(cnn_acc/num * 100)
    baseline_acc.append(base_acc/num * 100)

test_fold_num = 6

plt.scatter(accuracy_list, test_fold_num * ['L1_pre'], color='black')
plt.scatter(line_test_acc, test_fold_num * ['Linear'], color='green')
plt.scatter(cnn_test_acc, test_fold_num * ['Cnn'], color='blue')
plt.scatter(baseline_acc, test_fold_num * ['base'], color='red')
plt.xlabel("accuracy.percent %")
plt.ylabel("algorithm")
plt.tight_layout()
plt.savefig("test_accuracy.png")
