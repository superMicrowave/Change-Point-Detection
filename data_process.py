## import package
import pandas as pd
from sklearn import preprocessing
import numpy as np

## load the realating csv file
dir_path = 'Data/'
inputs_file = 'inputs.csv'
outputs_file = 'outputs.csv'

inputs = pd.read_csv(dir_path + inputs_file) #used for based line model 
outputs = pd.read_csv(dir_path + outputs_file)
folds = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
   'neuroblastoma-data/master/data/systematic/cv/sequenceID/folds.csv')

## procssing data
label = outputs.values
num_id = label.shape[0]
num_feature = inputs.shape[1] - 1
seq_id = inputs.iloc[:, 0].to_frame()
inputs = preprocessing.scale(inputs.iloc[:, 1:])
inputs = pd.concat([seq_id, pd.DataFrame(inputs)], axis=1)
inputs = np.array(inputs)
folds = np.array(folds)
_, cor_index = np.where(inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

