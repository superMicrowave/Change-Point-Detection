import pandas as pd
import numpy as np
import glob

# accuarcy fucntion
def Accuracy(predicated_y, target_y):
    if (np.logical_and(target_y[0] - predicated_y < 0,
                         predicated_y - target_y[1] < 0)):
        return 1
    else:
        return 0
    
# function to split data
def SplitFolder(labels, folders, fold_id):
    bool_suq = folders == fold_id
    train_label = labels[~bool_suq][:, 1:]
    test_label = labels[bool_suq][:, 1:]

    return train_label, test_label

# Load data
## load the realating csv file
dir_path = 'Data/'
inputs_file = 'inputs.csv'
outputs_file = 'outputs.csv'
fold_pre_list = []
sub_path = 'L1/'


outputs = pd.read_csv(dir_path + outputs_file)
folds = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
   'neuroblastoma-data/master/data/systematic/cv/sequenceID/folds.csv')
L1_prediction_files = glob.glob(dir_path + sub_path + "/*.csv")

for filename in L1_prediction_files:
    df = pd.read_csv(filename).iloc[:, -1].values
    fold_pre_list.append(df)


labels = outputs.values
folds = np.array(folds)
_, cor_index = np.where(labels[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

# processing data

fold_lab_list = []
for fold_num in range(1, 7):
    _, fold_lab = SplitFolder(labels, folds_sorted[:, 1], fold_num)
    fold_lab_list.append(fold_lab)

accuracy_list = []
# calcualte accuracy
for fold_num in range(6):
    accuracy = 0
    num = fold_pre_list[fold_num].shape[0]
    for (predict, label) in zip(fold_pre_list[fold_num], fold_lab_list[fold_num]):
        label = label.reshape(2)
        accuracy += Accuracy(predict, label)
    accuracy_list.append(accuracy/num)

print(accuracy_list)


    
    