## import package
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

## load the realating csv file
dir_path = 'Data/'
inputs_file = 'inputs.csv'
outputs_file = 'outputs.csv'

inputs = pd.read_csv(dir_path + inputs_file) #used for based line model 
outputs = pd.read_csv(dir_path + outputs_file)
folds = pd.read_csv('https://raw.githubusercontent.com/tdhock/'
   'neuroblastoma-data/master/data/systematic/cv/sequenceID/folds.csv')

## procssing data
baseline_label = outputs.values
num_id = baseline_label.shape[0]
num_feature = inputs.shape[1] - 1
inputs = np.array(inputs)
folds = np.array(folds)
_, cor_index = np.where(inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

## split data by folder
# function to split data
def SplitFolder(inputs, labels, folders, fold_id):
    bool_suq = folders == fold_id
    train_data = inputs[~bool_suq]
    test_data = inputs[bool_suq]
    train_label = labels[~bool_suq]
    test_label = labels[bool_suq]

    return train_data, test_data, train_label, test_label

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

## accuracy function
def Accuracy(predicated_y, target_y):
    if (np.logical_and(target_y[0] - predicated_y < 0,
                         predicated_y - target_y[1] < 0)):
        return 1
    else:
        return 0

## define the baseline network
class BaselineNN(nn.Module):
    def __init__(self):
        super(BaselineNN, self).__init__()
        self.fc1 = nn.Linear(num_feature, 1)

    def forward(self, x):
        x =  F.relu(self.fc1(x))
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
baselineNN = BaselineNN().to(device)

## define the loss funciton
criterion = SquareHingeLoss()
stepsize = 1e-15
optimizer = optim.SGD(baselineNN.parameters(),  lr=stepsize)

# split train test data, using Kfold
test_accuracy = []
for fold_num in range(1, 7):
    train_data, test_data, train_label, test_label = SplitFolder(inputs, baseline_label, 
                                                    folds_sorted[:, 1], fold_num)

    # split train vlidation data
    num_sed_fold = train_data.shape[0]
    sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
    left = np.arange(num_sed_fold % 5) + 1
    sed_fold = np.concatenate((sed_fold, left), axis=0)
    np.random.shuffle(sed_fold)

    subtrain_data, valid_data, subtrain_label, valid_label = SplitFolder(train_data, train_label, 
                                                    sed_fold, 1)

    num_train = subtrain_data.shape[0]
    num_valid = valid_data.shape[0]
    num_test = test_data.shape[0]

    # transfer data type
    subtrain_data = torch.from_numpy(subtrain_data[:, 1:].astype(float))
    valid_data = torch.from_numpy(valid_data[:, 1:].astype(float))
    test_data = torch.from_numpy(test_data[:, 1:].astype(float))
    subtrain_label = torch.from_numpy(subtrain_label[:, 1:].astype(float))
    valid_label = torch.from_numpy(valid_label[:, 1:].astype(float))
    test_label = torch.from_numpy(test_label[:, 1:].astype(float))

    # init variables
    step = 0
    train_losses, valid_losses, valid_accuracy= [], [], []

    subtrain_data = subtrain_data.type(torch.FloatTensor)
    subtrain_data = Variable(subtrain_data).to(device)
    subtrain_label = subtrain_label.to(device).float()

    valid_data = valid_data.type(torch.FloatTensor)
    valid_data = Variable(valid_data).to(device)
    valid_label = valid_label.to(device).float()

    test_data = test_data.type(torch.FloatTensor)
    test_data = Variable(test_data).to(device)
    test_label = valid_label.to(device).float()

    test_outputs = []
    mini_batches = 10
    num_epoch = 50

    ## train the network
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        for index in range(num_train):
            baselineNN.train()
        
            # init variable
            train_loss = 0
            valid_loss = 0     
            accuracy = 0

            # step + 1
            step += 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # do SGD
            outputs = baselineNN(subtrain_data[index])
            print(outputs)
            loss = criterion(outputs, subtrain_label[index])
            loss.backward()
        
            optimizer.step()
        
            if step % mini_batches == 0:
                with torch.no_grad():
                    baselineNN.eval()
        
                    # calculate the loss of train and valid
                    train_outputs = baselineNN(subtrain_data)
                    train_loss = criterion(train_outputs, subtrain_label)
                    train_losses.append(train_loss.cpu().data.numpy())
        
                    valid_outputs = baselineNN(valid_data)
                    valid_loss = criterion(valid_outputs, valid_label)  
                    valid_losses.append(valid_loss.cpu().data.numpy())
        
                test_output = baselineNN(test_data)
                test_outputs.append(test_output)

    # choose the min value from valid list
    min_loss_train = min(train_losses)
    min_train_index = train_losses.index(min(train_losses))
    min_loss_valid = min(valid_losses)
    best_parameter_value = valid_losses.index(min(valid_losses))
    best_output = test_outputs[best_parameter_value]

    # plot
    plt.plot(train_losses, label = 'Training loss')
    plt.plot(valid_losses, label = 'Validation loss')
    plt.scatter(min_train_index, min_loss_train, label = 'min train value', color='green')
    plt.scatter(best_parameter_value, min_loss_valid, label = 'min valid value', color='black')
    plt.legend(frameon=False)
    plt.xlabel("step of every min-bath")
    plt.ylabel("loss")
    plt.show()


    # test data
    with torch.no_grad():
        accuracy = 0
        for index in range(num_test):
            accuracy = accuracy + Accuracy(best_output[index], test_label[index])
        test_accuracy.append(accuracy/num_valid * 100)





        
         

