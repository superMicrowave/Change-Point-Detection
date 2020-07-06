## import package
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

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
    if (np.logical_and(target_y[0] - predicated_y < 0,
                         predicated_y - target_y[1] < 0)):
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
                return torch.tensor([0.0], requires_grad=True)
        
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



