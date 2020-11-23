# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 04:12:55 2020

@author: 22602
"""

## import package
#from function import *
import sys
import os
from function import *

## load the realating csv file
# get command line argument length.
dir_path = sys.argv[1]

## load the realating csv file
## load the realating csv file
labels_path = dir_path + "/outputs.csv.xz"
input_path = dir_path + "/profiles.csv.xz"
output_path = dir_path

#init the mean pool
fix_feature = 500
pooler = nn.AdaptiveAvgPool1d(fix_feature)

## load the file
dtypes = { "sequenceID": "category"}
profiles = pd.read_csv(input_path, dtype=dtypes)
labels = pd.read_csv(labels_path)

## extract all sequence id
sequenceID = labels["sequenceID"]
seq_data_list = []
# loop through all 
for id in sequenceID:
    #extract all data from profiels using same id
    one_object = profiles.loc[profiles["sequenceID"] == id]
    one_feature = torch.from_numpy(np.array(one_object["signal"]))
    ##mean the data feature
    num_feature = one_feature.size()[0]
    one_feature = one_feature.reshape(1,1,num_feature)
    mean_feature = pooler(one_feature).data.numpy()
    mean_feature = pd.DataFrame(mean_feature.reshape(1,fix_feature))
    seq_data_list.append(mean_feature)


##convert data yupt
seq_data_pd = pd.DataFrame(seq_data_list)
sequenceID_pd = pd.DataFrame(sequenceID, columns = ['sequenceID'])
seq_data = pd.concat([sequenceID_pd, seq_data_pd], axis = 1)

#print the data as output
seq_data.to_csv(output_path + '/mean_inputs.csv.xz', index=False) 