import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from dataloader2 import *
from model_factory import *

# from model_factory2 import get_model
import pdb
import ssl
from fileutils import *

from PIL import Image
import matplotlib.pyplot as plt
import os

ROOT_STATS_DIR = './experiment_data'

class Experiment(object):
    def __init__(self, name):
        self.config_data = read_file_in_dir('./', name + '.json')
        if self.config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.__name = self.config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        self.__epochs = self.config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        #self.__batch_size = config_data['dataset']['batch_size']
        self.__batch_size = self.config_data['dataset']['batch_size']
        modelType = self.config_data['model']['model_type']
        self.__learning_rate = self.config_data['experiment']['learning_rate']
        # Load Datasets
        self.__train_loader, self.__val_loader, self.__test_loader = self.get_datasets(self.__batch_size)
        #pdb.set_trace()

        # Setup Experiment
        #self.__generation_config = config_data['generation']
        #self.__epochs = config_data['experiment']['num_epochs']

        # Init Model
        self.__model = get_model(pretrained_model_type=modelType)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__learning_rate)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()
    
    def get_datasets(self, batch_size):
        train_dataset = KaggleBrainMRIDataset(csv_file='train_images.csv', config_data=self.config_data, mode='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.__batch_size,
                                  num_workers=0,shuffle=True)
        val_dataset = KaggleBrainMRIDataset(csv_file='val_images.csv', config_data=self.config_data, mode='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.__batch_size,
                                  num_workers=0,shuffle=True)
        test_dataset = KaggleBrainMRIDataset(csv_file='test_images.csv', config_data=self.config_data, mode='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.__batch_size,
                                  num_workers=0,shuffle=True)
        return train_loader,val_loader,test_loader
        
    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        # makedirs is in run()
        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
            self.__best_model = self.__model

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
        else:
            self.__model = self.__model.float()
            #self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        minValLoss = np.inf
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val(minValLoss)
            if(epoch == 0):
                minValLoss = val_loss
            else:
                if(val_loss < minValLoss):
                    minValLoss = val_loss
                    
            os.makedirs(ROOT_STATS_DIR, exist_ok=True)
            
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        average_loss = 0
        iterCount = 0
        for i, (images, onehotTarget, label) in enumerate(self.__train_loader):
            #if images.shape[0]!=self.__batch_size:
            #    break
            self.__optimizer.zero_grad()
            if torch.cuda.is_available():
                images = images.cuda().float()
                label = label.cuda().long()
            else:
                images = images.float()
                label = label.long()
            
            #pdb.set_trace()
            modelOutput = self.__model(images).float()
            modelOutput = torch.squeeze(modelOutput)  # get 1 dimension result, length = batch_size
            
            loss = self.__criterion(modelOutput, label)
            average_loss += loss.item()
            loss.backward()
            self.__optimizer.step()
            #if i % 10 == 0:
            print("Train iter{}, loss: {}".format(i, loss.item()))
            
            iterCount += 1
            
        training_loss = average_loss/iterCount
        return training_loss

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self, prevMinLoss):
        self.__model.eval()
        val_loss = 0
        average_loss = 0
        iterCount = 0
        with torch.no_grad():
            for i, (images, onehotTarget, label) in enumerate(self.__val_loader):
                #if images.shape[0]!=self.__batch_size:
                #    break
                
                if torch.cuda.is_available():
                    images = images.cuda().float()
                    label = label.cuda().long()
                else:
                    images = images.float()
                    label = label.long()
                    
                #pdb.set_trace()
                modelOutput = self.__model(images).float()
                modelOutput = torch.squeeze(modelOutput)  # get 1 dimension result, length = batch_size
            
                loss = self.__criterion(modelOutput, label)
                average_loss += loss.item()
                #if i % 10 == 0:
                print("Val iter{}, loss: {}".format(i, loss.item()))
                iterCount += 1
                
            val_loss = average_loss/iterCount
            if(val_loss < prevMinLoss):
                self.__best_model = self.__model
            return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__best_model.eval()
        if torch.cuda.is_available():
            self.__best_model.cuda()
        test_loss = 0
        acc = 0
        iterCount = 0
        TP,FP,TN,FN = 0,0,0,0
        iterCount = 0
        with torch.no_grad():
            for i, (images, onehotTarget, label) in enumerate(self.__val_loader):
                #if images.shape[0]!=self.__batch_size:
                #    break
                
                if torch.cuda.is_available():
                    images = images.cuda().float()
                    label = label.cuda().long()
                else:
                    images = images.float()
                    label = label.long()
                    
                #pdb.set_trace()
                modelOutput = self.__model(images).float()
                modelOutput = torch.squeeze(modelOutput)  # get 1 dimension result, length = batch_size
            
                loss = self.__criterion(modelOutput, label)
                test_loss += loss.item()
                iterCount += 1
                pred = (modelOutput>0.5).int()
                tp,fp,tn,fn = self.get_TFPN(pred,label)
                TP += tp
                FP += fp
                TN += tn
                FN += fn

            test_loss = test_loss/iterCount
            ACC = (TP+TN)/(TP+FP+TN+FN)
            BER = 1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP))
            
        result_str = "Test Performance: Loss: {}, ACC: {}, BER: {}".format(test_loss,ACC,BER)
        #result_str = "Test Performance: Loss: {}".format(test_loss)
        self.__log(result_str)

        return test_loss, ACC, BER
        #return test_loss
    
    def get_TFPN(self,pred,label):
        #pdb.set_trace()
        pred=np.argmax(pred.cpu().numpy(),axis=1)
        label=label.cpu().numpy()
        TP = (pred*label).sum()
        FP = pred.sum()-TP
        TN = ((pred+label)==0).sum()
        FN = label.sum()-TP
        return TP,FP,TN,FN

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.close()
        #plt.show()
    
