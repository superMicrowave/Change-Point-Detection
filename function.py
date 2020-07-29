## import package
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# this fucntion we transfer the data and label type from numpy to tensor
def Typetransfer_2D(data, label):
    data = torch.from_numpy(data[:, 1:].astype(float))
    data = data.type(torch.FloatTensor)
    data = Variable(data).to(device)
    label = torch.from_numpy(label[:, 1:].astype(float))
    label = label.to(device).float()
    
    return data, label

# this fucntion we transfer the data and label type from numpy to tensor
def Typetransfer_3D(data, label, channel):
    num_data = data.shape[0]
    num_feature = data.shape[1] - 1
    data = torch.from_numpy(data[:, 1:].astype(float)).view(num_data, channel, num_feature)
    data = data.type(torch.FloatTensor)
    data = Variable(data).to(device)
    label = torch.from_numpy(label[:, 1:].astype(float))
    label = label.to(device).float()
    
    return data, label

## split data by folder
# function to split data
def SplitFolder(inputs, labels, folders, fold_id):
    bool_suq = folders == fold_id
    train_data = inputs[~bool_suq]
    test_data = inputs[bool_suq]
    train_label = labels[~bool_suq]
    test_label = labels[bool_suq]

    return train_data, test_data, train_label, test_label

## accuracy function
def Accuracy(predicated_y, target_y):
    if (np.logical_and(target_y[0] - predicated_y <= 0,
                         predicated_y - target_y[1] <= 0)):
        return 1
    else:
        return 0
    
## squareHangLoss function
class SquareHingeLoss(nn.Module):
    def __init__(self):
        super(SquareHingeLoss,self).__init__()
    
    def ifelse(self, x):
        num = x.size()[0]      
        if num == 1:
            flag = x - 1 < 0
            if flag == True:
                return (x - 1) ** 2
            else:
                return torch.tensor([0.0], device='cuda:0', requires_grad=True)
        
        else:
            crit = (x - 1 < 0)
            copy_x = x.clone()
            copy_x[crit] = (x[crit] - 1) ** 2
            copy_x[~crit] = 0
            return torch.sum(copy_x)
       
    def forward(self, predicated_y, target_y):
        num = predicated_y.size()[0]
   
        if num == 1:
            target_y = target_y.view(-1, 1)
            result = (self.ifelse(predicated_y - target_y[0]) +
                          self.ifelse(target_y[1] - predicated_y))
            
        else:
            result = (self.ifelse(predicated_y - target_y[:, 0]) +
                     self.ifelse(target_y[:, 1] - predicated_y)) / num
        
        return result

# ssp function,based on the AdaptiveMax/Avg method built in torch
class SpatialPyramidPooling(nn.Module):
    def __init__(self, mode):
        super(SpatialPyramidPooling, self).__init__()
        num_pools = [1, 4, 16]
        self.name = 'SpatialPyramidPooling'
        if mode == 'max':
            pool_func = nn.AdaptiveMaxPool1d
        elif mode == 'avg':
            pool_func = nn.AdaptiveAvgPool1d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")
        self.pools_fun = []
        for p in num_pools:
            self.pools_fun.append(pool_func(p))

    def forward(self, feature_maps):
        pooled = []
        for pool_fun in self.pools_fun:
            pooled.append(pool_fun(feature_maps))
        return torch.cat(pooled, dim=2)

# this fucntion output the csv file
def OutputFile(output, output_list):
    for index in range(1,6):
        df = pd.DataFrame(output_list[index])
        output = pd.concat([output, df], axis = 1 )  
    return output
    

