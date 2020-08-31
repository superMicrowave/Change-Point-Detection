from function import *

class earlyStop:
    def __init__(self, model, optimizer, criterion, train_data, train_label, epoch):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data
        self.train_label = train_label
        self.epoch = epoch
    
    def split_data(train_data, train_label):
        num_sed_fold = train_data.shape[0]
        sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
        left = np.arange(num_sed_fold % 5) + 1
        sed_fold = np.concatenate((sed_fold, left), axis=0)
        np.random.shuffle(sed_fold)

        subtrain_data, valid_data, subtrain_label, valid_label = SplitFolder(train_data, train_label, sed_fold, 1)
        
        return subtrain_data, valid_data, subtrain_label, valid_label 
    
    def paramter_select(avg_train_losses, avg_valid_losses):
        min_loss_train = min(avg_train_losses)
        min_train_index = avg_train_losses.index(min_loss_train)
        min_loss_valid = min(avg_valid_losses)
        best_parameter = avg_valid_losses.index(min_loss_valid)
        
        #plt.plot(avg_train_losses, label = 'Training loss')
        #plt.plot(avg_valid_losses, label = 'Validation loss')
        #plt.scatter(min_train_index, min_loss_train, label = 'min train value', color='green')
        #plt.scatter(best_parameter_value, min_loss_valid, label = 'min valid value', color='black')
        #plt.legend(frameon=False)
        #plt.xlabel("step of every min-bath")
        #plt.ylabel("loss")
        #plt.savefig('foo.png')
        
        return best_parameter

    def __call__(self):
        # split to subtrain, valid
        subtrain_data, valid_data, subtrain_label, valid_label = earlyStop.split_data(self.train_data, self.train_label)
        
        # transfer data type
        num_train = subtrain_data.shape[0]
        num_valid = valid_data.shape[0]
        subtrain_data, subtrain_label = Typetransfer_2D(subtrain_data, subtrain_label)
        valid_data, valid_label = Typetransfer_2D(valid_data, valid_label)

        # init variables
        step = 0
        train_losses, valid_losses = [], []
        avg_train_losses, avg_valid_losses = [], []

        ## train the network
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            for index, (data, label) in enumerate(zip(subtrain_data, subtrain_label)):
                self.model.train()   
        
                # step + 1
                step += 1
        
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # do SGD
                outputs = self.model(data)
                loss = self.criterion(outputs, label)
                train_losses.append(loss.cpu().data.numpy())
                
                loss.backward()
                self.optimizer.step()
            
            for index, (data, label) in enumerate(zip(valid_data, valid_label)):
                with torch.no_grad():
                   self.model.eval()
                   valid_outputs = self.model(data)                    
                   valid_loss = self.criterion(valid_outputs, label)
                   valid_losses.append(valid_loss.cpu().data.numpy())
            
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

        
        print(avg_valid_losses)
        #get best epoch and return 
        best_epoch = earlyStop.paramter_select(avg_train_losses, avg_valid_losses)
    
        return best_epoch
               