# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 02:28:03 2020

@author: 22602
"""
import pandas as pd
from data_process import L1_pre_list
from function import OutputFile

# this fucntion output the csv file
L1_output = pd.DataFrame(L1_pre_list[0])
L1_output = OutputFile(L1_output, L1_pre_list)
L1_output.to_csv(r'Data/Outputs/L1Model.csv', index = None, header = False) 
