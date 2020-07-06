
## import package
from function import *
from data_process import inputs, label, folds_sorted

# this fucntion we transfer the data and label type from numpy to tensor
def Typetransfer(data, label, channel):
    num_data = data.shape[0]
    num_feature = data.shape[1] - 1
    data = torch.from_numpy(data[:, 1:].astype(float)).view(num_data, channel, num_feature)
    data = data.type(torch.FloatTensor)
    data = Variable(data).to(device)
    label = torch.from_numpy(label[:, 1:].astype(float))
    label = label.to(device).float()
    
    return data, label

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

# build the net work
class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 5, 9), #1 1 117 -> 1 5 109
        )
        self.layer2 = nn.Sequential(
            nn.Linear(5*21, 1),
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
    # set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = convNet().to(device)
    criterion = SquareHingeLoss()
    optimizer = optim.Adam(model.parameters(),  lr=1e-5)

    # transfer data
    num_train = subtrain_data.shape[0]
    num_valid = valid_data.shape[0]
    num_test = test_data.shape[0]
    channel = 1
    subtrain_data, subtrain_label = Typetransfer(subtrain_data, subtrain_label, channel)
    valid_data, valid_label = Typetransfer(valid_data, valid_label, channel)
    test_data, test_label = Typetransfer(test_data, test_label, channel)

    # init variables
    step = 0
    train_losses, valid_losses, valid_accuracy= [], [], []
    parameters = []
    test_outputs = []
    cnn_test_accuracy = []
    num_epoch = 10
    mini_batches = 10

    ## train the network
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        for index in range(num_train):
            model.train()
        
            # init variable
            train_loss = 0
            valid_loss = 0     
            accuracy = 0

            # step + 1
            step += 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # do ADM
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
                    print(valid_loss)
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

