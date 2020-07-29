## import package
import pandas as pd
from sklearn import preprocessing
import numpy as np
import glob

## load the realating csv file
dir_path = 'Data/Inputs/'
inputs_file = 'inputs.csv'
outputs_file = 'outputs.csv'
L1_path = 'L1/'
baseline_path = 'baseline/'

inputs = pd.read_csv(dir_path + inputs_file) #used for based line model 
outputs = pd.read_csv(dir_path + outputs_file)
folds = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
   'neuroblastoma-data/master/data/systematic/cv/sequenceID/folds.csv')
L1_prediction_files = glob.glob(dir_path + L1_path + "/*.csv")
baseline_files = glob.glob(dir_path + baseline_path + "/*.csv")

## procssing data
#for input and output
labels = outputs.values
num_id = labels.shape[0]
num_feature = inputs.shape[1] - 1
seq_id = inputs.iloc[:, 0].to_frame()
min_max_scaler = preprocessing.MinMaxScaler()
MM_inputs = min_max_scaler.fit_transform(inputs.iloc[:, 1:])
MM_inputs = pd.concat([seq_id, pd.DataFrame(MM_inputs)], axis=1)
MM_inputs = np.array(MM_inputs)

Scale_inputs = preprocessing.scale(inputs.iloc[:, 1:])
Scale_inputs = pd.concat([seq_id, pd.DataFrame(Scale_inputs)], axis=1)
Scale_inputs = np.array(Scale_inputs)

folds = np.array(folds)
folds = np.array(folds)
_, cor_index = np.where(MM_inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

L1_pre_list = []
# for L1
for filename in L1_prediction_files:
    df = pd.read_csv(filename).iloc[:, -1].values
    L1_pre_list.append(df)

baseline_list = []
# for baseline
for filename in baseline_files:
    df = pd.read_csv(filename).iloc[:, -1].values
    baseline_list.append(df)
