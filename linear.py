## import package
from function import *
import sys
from sklearn import preprocessing
from early_stop import *

## load the realating csv file
# get command line argument length.
argv = sys.argv[1]

## load the realating csv file
dir_path = argv + '/Inputs/'
inputs_file = 'inputs.csv.xz'
outputs_file = 'outputs.csv.xz'
folds_file = 'folds.csv'

inputs = pd.read_csv(dir_path + inputs_file)
outputs = pd.read_csv(dir_path + outputs_file)
folds = pd.read_csv(dir_path + folds_file)

## procssing data
labels = outputs.values
num_id = labels.shape[0]
num_feature = inputs.shape[1] - 1
seq_id = inputs.iloc[:, 0].to_frame()
Scale_inputs = preprocessing.scale(inputs.iloc[:, 1:])
Scale_inputs = pd.concat([seq_id, pd.DataFrame(Scale_inputs)], axis=1)
Scale_inputs = np.array(Scale_inputs)

folds = np.array(folds)
_, cor_index = np.where(Scale_inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split


## define the baseline network
class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(num_feature, 1)

    def forward(self, x):
        x =  self.fc1(x)
        return x

## define the loss funciton and init the model
inputs = Scale_inputs
criterion = SquareHingeLoss()
model_list = []
optimizer_list = []
for num_model in range(6):
    model = LinearNN().to(device)
    optimizer = optim.Adam(model.parameters(),  lr= 1e-4)
    model_list.append(model)
    optimizer_list.append(optimizer)

# split train test data, using Kfold
line_test_acc = []
test_output_list = []
for fold_num in range(1, 7):
    train_data, test_data, train_label, test_label = SplitFolder(inputs, labels, 
                                                    folds_sorted[:, 1], fold_num)
    
    # set up model
    num_model = fold_num - 1
    model = model_list[num_model]
    optimizer = optimizer_list[num_model]
    
    #save the init model
    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)
    
    # get best epoch
    best_epoch = earlyStop(model, optimizer, criterion, train_data, train_label, 200).__call__()
    print(best_epoch)
    
    # transfer data type
    num_test = test_data.shape[0]
    train_data, train_label = Typetransfer_2D(train_data, train_label)
    test_data, test_label = Typetransfer_2D(test_data, test_label)
    
    # init variables
    step = 0
    model = LinearNN().to(device)
    model.load_state_dict(torch.load(PATH))
    optimizer = optim.Adam(model.parameters(),  lr= 1e-4)
    
    # trian the model
    for epoch in range(best_epoch):  # loop over the dataset multiple times
        for index, (data, label) in enumerate(zip(train_data, train_label)):
            model.train()   
    
            # step + 1
            step += 1
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # do SGD
            outputs = model(data)
            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()
    
    # get test data
    with torch.no_grad():
        model.eval()  
        test_outputs = model(test_data).cpu().data.numpy()         

    test_output_list.append(test_outputs)             

    # test data
    with torch.no_grad():
        accuracy = 0
        for index in range(num_test):
            accuracy = accuracy + Accuracy(test_outputs[index], test_label[index].cpu().data.numpy())
        line_test_acc.append(accuracy/num_test * 100)
        print(accuracy/num_test * 100)

# this fucntion output the csv file
linear_output = pd.DataFrame(test_output_list[0])
linear_output = OutputFile(linear_output, test_output_list)
linear_output.to_csv(argv + '/Outputs/linearModel.csv', index = None, header = False) 




        
         

