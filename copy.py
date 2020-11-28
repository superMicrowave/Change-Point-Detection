# -*- coding: utf-8 -*-
import shutil
import sys
from sys import exit
import os

dir_path = sys.argv[1]
model_id = sys.argv[2]



file_path = dir_path + "/randomTrainOrderings/1/models_test/Cnn_zero_test/" + model_id + '/predictions.csv'
copy_path = dir_path + "/randomTrainOrderings/1/models/Cnn_zero_pytorch/" 

if not os.path.exists(copy_path):
    os.makedirs(copy_path) 

newPath = shutil.copy(file_path, copy_path)
