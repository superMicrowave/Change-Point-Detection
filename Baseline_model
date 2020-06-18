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
dir_path = 'loading_file/'
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
        crit = (x - 1 < 0)
        copy_x = x.clone()
        copy_x[crit] = (x[crit] - 1) ** 2
        copy_x[~crit] = 0
        return torch.sum(copy_x)
       
    def forward(self, predicated_y, target_y):
        num = predicated_y.size()[0]
        result = (self.ifelse(predicated_y - target_y[0]) +
                     self.ifelse(target_y[1] - predicated_y)) / num
        
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
optimizer = optim.SGD(baselineNN.parameters(),  lr=1e-10)

# split train test data, using Kfold
test_accuracy = []
for fold_num in range(1, 6):
    train_data, test_data, train_label, test_label = SplitFolder(inputs, baseline_label, 
                                                    folds_sorted[:, 1], fold_num)

    # split train vlidation data
    num_sed_fold = train_data.shape[0]
    sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
    left = np.arange(num_sed_fold % 5) + 1
    sed_fold = np.concatenate((sed_fold, left), axis=0)
    np.random.shuffle(sed_fold)

    train_data, valid_data, train_label, valid_label = SplitFolder(train_data, train_label, 
                                                    sed_fold, 1)

    num_train = train_data.shape[0]
    num_valid = valid_data.shape[0]

    # transfer data type
    train_data = torch.from_numpy(train_data[:, 1:].astype(float))
    valid_data = torch.from_numpy(valid_data[:, 1:].astype(float))
    test_data = torch.from_numpy(test_data[:, 1:].astype(float))
    train_label = torch.from_numpy(train_label[:, 1:].astype(float))
    valid_label = torch.from_numpy(valid_label[:, 1:].astype(float))
    test_label = torch.from_numpy(test_label[:, 1:].astype(float))

    # init variables
    step = 0
    mini_batches = 500
    running_loss = 0.0
    train_losses, valid_losses, valid_accuracy= [], [], []
    ## train the network
    for epoch in range(50):  # loop over the dataset multiple times
        for index, (data, labels) in enumerate(zip(valid_data, valid_label)):
            # step + 1
            step += 1

            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            data = data.type(torch.FloatTensor)
            data = Variable(data).to(device)
            outputs = baselineNN(data)
            labels = labels.view(-1, 1).to(device)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print loss
            running_loss += loss.cpu().data.numpy()
            if step % mini_batches == 0:  
                valid_loss = 0
                accuracy = 0
                baselineNN.eval()
                with torch.no_grad():
                    for index, (data, labels) in enumerate(zip(test_data, test_label)):
                        data = data.type(torch.FloatTensor)
                        data = Variable(data).to(device)
                        labels = labels.view(-1, 1).to(device)
                        valid_outputs = baselineNN(data)
                        accuracy += Accuracy(valid_outputs, labels.float())
                        loss = criterion(valid_outputs, labels.float())
                        valid_loss += loss.cpu().data.numpy()
                    train_losses.append(running_loss/mini_batches)
                    valid_losses.append(valid_loss/num_valid) 
                    valid_accuracy.append(accuracy/num_valid * 100)
                    baselineNN.train()
                    running_loss = 0.0

        # plot
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

        # save the parameter
        torch.save(baselineNN.state_dict(), 'model.pth')

    # load the net work
    baselineNN = BaselineNN().to(device)
    baselineNN.load_state_dict(torch.load('model.pth'))

    # test data
    with torch.no_grad():
        accuracy = 0
        for index, (data, labels) in enumerate(zip(test_data, test_label)):
            data = data.type(torch.FloatTensor)
            data = Variable(data).to(device)
            labels = labels.view(-1, 1).to(device)
            accuracy += Accuracy(valid_outputs, labels.float())
        test_accuracy.append(accuracy/num_valid * 100)

print(test_accuracy)
