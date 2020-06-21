
import pandas as pd
import numpy as np

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
dir_path = 'loading_file/'
inputs_file = 'inputs.csv'
outputs_file = 'outputs.csv'

outputs = pd.read_csv(dir_path + outputs_file)
folds = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
   'neuroblastoma-data/master/data/systematic/cv/sequenceID/folds.csv')

fold_1_pre = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
                         'neuroblastoma-data/master/data/'
                         'systematic/cv/sequenceID/testFolds/1/'
                         'randomTrainOrderings/1/models/L1reg_linear_all/'
                         'predictions.csv').iloc[:, -1].values

fold_2_pre = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
                         'neuroblastoma-data/master/data/'
                         'systematic/cv/sequenceID/testFolds/2/'
                         'randomTrainOrderings/1/models/L1reg_linear_all/'
                         'predictions.csv').iloc[:, -1].values

fold_3_pre = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
                         'neuroblastoma-data/master/data/'
                         'systematic/cv/sequenceID/testFolds/3/'
                         'randomTrainOrderings/1/models/L1reg_linear_all/'
                         'predictions.csv').iloc[:, -1].values

fold_4_pre = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
                         'neuroblastoma-data/master/data/'
                         'systematic/cv/sequenceID/testFolds/4/'
                         'randomTrainOrderings/1/models/L1reg_linear_all/'
                         'predictions.csv').iloc[:, -1].values

fold_5_pre = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
                         'neuroblastoma-data/master/data/'
                         'systematic/cv/sequenceID/testFolds/5/'
                         'randomTrainOrderings/1/models/L1reg_linear_all/'
                         'predictions.csv').iloc[:, -1].values

fold_6_pre = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
                         'neuroblastoma-data/master/data/'
                         'systematic/cv/sequenceID/testFolds/6/'
                         'randomTrainOrderings/1/models/L1reg_linear_all/'
                         'predictions.csv').iloc[:, -1].values

labels = outputs.values
folds = np.array(folds)
_, cor_index = np.where(labels[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

# processing data
fold_pre_list = []
fold_pre_list.append(fold_1_pre)
fold_pre_list.append(fold_2_pre)
fold_pre_list.append(fold_3_pre)
fold_pre_list.append(fold_4_pre)
fold_pre_list.append(fold_5_pre)
fold_pre_list.append(fold_6_pre)

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
        





    
    
