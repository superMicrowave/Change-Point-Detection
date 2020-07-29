# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 02:25:02 2020

@author: 22602
"""
import pandas as pd
from data_process import baseline_list
from function import OutputFile

# this fucntion output the csv file
baseline_output = pd.DataFrame(baseline_list[0])
baseline_output = OutputFile(baseline_output, baseline_list)
baseline_output.to_csv(r'Data/Outputs/baselineModel.csv', index = None, header = False)

