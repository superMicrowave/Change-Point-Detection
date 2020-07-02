 
## import package
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
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
label = outputs.values
num_id = label.shape[0]
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


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 5, 9), #1 1 117 -> 1 5 109
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(5*21, 1),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = convNet().to(device)
criterion = SquareHingeLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

## accuracy function
def Accuracy(predicated_y, target_y):
    if (np.logical_and(target_y[0] - predicated_y < 0,
                         predicated_y - target_y[1] < 0)):
        return 1
    else:
        return 0

# split train test data, using Kfold
cnn_test_acc = []
for fold_num in range(1, 7):
    train_data, test_data, train_label, test_label = SplitFolder(inputs, label, 
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
    channel = 1

    # transfer data type
    subtrain_data = torch.from_numpy(subtrain_data[:, 1:].astype(float)).view(num_train, channel, num_feature)
    valid_data = torch.from_numpy(valid_data[:, 1:].astype(float)).view(num_valid, channel, num_feature)
    test_data = torch.from_numpy(test_data[:, 1:].astype(float)).view(num_test, channel, num_feature)
    subtrain_label = torch.from_numpy(subtrain_label[:, 1:].astype(float))
    valid_label = torch.from_numpy(valid_label[:, 1:].astype(float))
    test_label = torch.from_numpy(test_label[:, 1:].astype(float))

    # init variables
    step = 0
    train_losses, valid_losses, valid_accuracy= [], [], []

    parameters = []
    grad_before = []
    grad_after = []

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
    cnn_test_accuracy = []
    num_epoch = 50
    mini_batches = 10

    ## train the network
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        for index in range(100):
            model.train()
        
            # init variable
            train_loss = 0
            valid_loss = 0     
            accuracy = 0

            # step + 1
            step += 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # do SGD
            outputs = model(subtrain_data[index].unsqueeze(0))
        
            loss = criterion(outputs, subtrain_label[index])
            loss.backward()
        
            optimizer.step()
        
            if step % mini_batches == 0:
                with torch.no_grad():
                    model.eval()
        
                    # calculate the loss of train and valid
                    train_outputs = model(subtrain_data)
                    train_loss = criterion(train_outputs, subtrain_label)
                    train_losses.append(train_loss.cpu().data.numpy())
        
                    valid_outputs = model(valid_data)
                    valid_loss = criterion(valid_outputs, valid_label)  
                    valid_losses.append(valid_loss.cpu().data.numpy())
        
                test_output = model(test_data)
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
    plt.xlabel("step of every 10 min-bath")
    plt.ylabel("loss")
    plt.show()


    # test data
    with torch.no_grad():
        accuracy = 0
        for index in range(num_test):
            accuracy = accuracy + Accuracy(best_output[index], test_label[index])
        cnn_test_acc.append(accuracy/num_valid * 100)


