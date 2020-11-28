# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:34:25 2020

@author: 22602
"""

## import package
#from function import *
import sys
import os
from function import *
from sklearn import preprocessing

## load the realating csv file
# get command line argument length.
dir_path = sys.argv[1]

## load the realating csv file
dir_path_split = dir_path.split("cv")
fold_path_split = dir_path.split("/testFolds/")
profiles_path = dir_path_split[0] + "profiles.csv.xz"
labels_path = dir_path_split[0] + "outputs.csv"
folds_path = fold_path_split[0] + "/folds.csv"
fold_num = int(fold_path_split[1])
outputs_path = dir_path + "/randomTrainOrderings/1/models/"

# build the net work
class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.AdaPool = nn.AdaptiveMaxPool1d(100)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            #nn.RReLU(),
            #nn.Tanh(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, 6),
            #nn.RReLU(),
            #nn.Tanh(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(57*8, 128)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#init the model parameter
cnn_model = convNet().to(device)
criterion = SquareHingeLoss()
step = 5e-5
epoch = 40
model = convNet().to(device)
optimizer = optim.Adam(model.parameters(),  lr= step)

#save the init model
if not os.path.exists("model_path/" + dir_path):
    os.makedirs("model_path/" + dir_path) 
PATH = "model_path/" + dir_path + 'cifar_net.pth'
torch.save(model.state_dict(), PATH)

## get the inputs from profile with correct sequence id order
profiles = pd.read_csv(profiles_path)
profiles = np.array(profiles)
labels = pd.read_csv(labels_path)
seq_id = list(labels.iloc[:, 0])
indexes = np.unique(profiles[:, 0], return_index=True)[1]
num_sequence = indexes.shape[0]
position = sorted(indexes)
seq_id_unsort = [profiles[index, 0] for index in position]
seq_list = []
for index in range(num_sequence):
    if(index == num_sequence-1):
        head = position[index]
        seq_feature_one = profiles[head:, 2]
    else:
        head = position[index]
        tail = position[index+1]
        seq_feature_one = profiles[head:tail, 2]
        
    N = seq_feature_one.shape[0]
    seq_feature_one = preprocessing.scale(seq_feature_one)
    seq_feature_one = torch.from_numpy(seq_feature_one.astype(float)).view(1, 1, N)
    seq_feature_one = seq_feature_one.type(torch.FloatTensor)
    seq_feature_one = Variable(seq_feature_one).to(device)
    seq_id_one = seq_id_unsort[index]
    seq_one = (seq_id_one, seq_feature_one)
    seq_list.append(seq_one)

seq_list = [tuple for x in seq_id for tuple in seq_list if tuple[0] == x]
inputs = [i[1] for i in seq_list]

## get folder
labels = labels.values
folds = pd.read_csv(folds_path)
folds = np.array(folds)
_, cor_index = np.where(labels[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

## transfer label type
labels = torch.from_numpy(labels[:, 1:].astype(float))
labels = labels.to(device).float()

## split train and test data
bool_flag = folds_sorted[:, 1] == fold_num
train_data = [a for i,a in enumerate(inputs) if not bool_flag[i]]
test_data = [a for i,a in enumerate(inputs) if bool_flag[i]]
train_label = labels[~bool_flag]
test_label = labels[bool_flag]
num_test = len(test_data)

## do early stop learning, get best epoch
#split validation and subtraining data
num_sed_fold = len(train_data)
sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
left = np.arange(num_sed_fold % 5) + 1
sed_fold = np.concatenate((sed_fold, left), axis=0)
np.random.shuffle(sed_fold)
bool_flag = sed_fold == 1
subtrain_data = [a for i,a in enumerate(train_data) if not bool_flag[i]]
valid_data = [a for i,a in enumerate(train_data) if bool_flag[i]]
subtrain_label = train_label[~bool_flag]
valid_label = train_label[bool_flag]

# do stochastic gradient descent
valid_losses = []
avg_valid_loss = []
## train the network
for epoch in range(epoch):  # loop over the dataset multiple times
    for index, (data, label) in enumerate(zip(subtrain_data, subtrain_label)):
        model.train()  

        # zero the parameter gradients
        optimizer.zero_grad()

        # do SGD
        outputs = model(data)
        loss = criterion(outputs, label)
        
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        for index, (data, label) in enumerate(zip(valid_data, valid_label)):
            model.eval()
            outputs = model(data)                    
            loss = criterion(outputs, label)
            valid_losses.append(loss.cpu().data.numpy())
    
    valid_loss = np.average(valid_losses)
    avg_valid_loss.append(valid_loss)

#get best parameter
min_loss_valid = min(avg_valid_loss)
print(min_loss_valid)
best_parameter = avg_valid_loss.index(min_loss_valid)

print(best_parameter)

# init variables for model
model = convNet().to(device)
model.load_state_dict(torch.load(PATH))
optimizer = optim.Adam(model.parameters(),  lr= step)

## train the network using best epoch
for epoch in range(best_parameter + 1): 
    for index, (data, label) in enumerate(zip(train_data, train_label)):
        model.train()  
    
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # do SGD
        outputs = model(data)
        loss = criterion(outputs, label)
        
        loss.backward()
        optimizer.step()
    
test_losses = []
test_outputs = []
with torch.no_grad():
    for data in test_data:
        output = model(data).cpu().data.numpy().reshape(-1)
        test_outputs.append(output)
    
    for index, (data, label) in enumerate(zip(test_data, test_label)):
            model.eval()
            outputs = model(data)                    
            loss = criterion(outputs, label)
            test_losses.append(loss.cpu().data.numpy())

    print(np.average(test_losses))
        
    test_outputs = np.array(test_outputs)
        
# test data
with torch.no_grad():
    accuracy = 0
    for index in range(num_test):
        accuracy = accuracy + Accuracy(test_outputs[index], test_label[index].cpu().data.numpy())
        
print(accuracy/num_test * 100)

# this fucntion output the csv file
cnn_output = pd.DataFrame(test_outputs)
if not os.path.exists(outputs_path + "Cnn_spp_pytorch"):
    os.mkdir(outputs_path + "Cnn_spp_pytorch") 
cnn_output.to_csv(outputs_path + 'Cnn_spp_pytorch/predictions.csv') 


