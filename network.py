import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import Compose


train_path = "./img"
val_path = "./img"

batch_size = 32
image_width = 50
num_workers = 1

list_of_transforms = [transforms.Resize((image_width,image_width)), 
                      transforms.Grayscale(1), 
                      transforms.ToTensor()]

train_folder = ImageFolder(train_path, transform=Compose(list_of_transforms))
train_loader = DataLoader(dataset=train_folder, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_folder = ImageFolder(val_path, transform=Compose(list_of_transforms))
val_loader = DataLoader(dataset=val_folder, batch_size=batch_size, shuffle=True, num_workers=num_workers)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:",device)

class Digit_recognizer(nn.Module):
    def __init__(self, outputs=None, L2_penalty=0, learning_rate=0.01):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        
        self.linear_regression_vector_length = (image_width-(3-1))**2*20# This assumes stride 1
        self.nn1 = nn.Linear(self.linear_regression_vector_length, 10)
        self.layers = [self.conv1, self.nn1]
            
        self.layers = nn.ModuleList(self.layers)
        print("created a network with layers:\n",self.layers)
        
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate,  weight_decay = L2_penalty) # weight_decay is the alpha in weight penalty regularization
            
        # Move this to device
        self.to(device)
        
    def forward(self, x): 
        x = self.layers[0](x)
        x = torch.reshape(x, (-1, self.linear_regression_vector_length))
        x = self.layers[1](x)
        x = F.log_softmax(x)
            
        return x
            
        
    
    def backward(self, predicted, truth):
        loss = self.loss_fn(predicted, truth)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def train_(self, train_loader, val_loader, epochs, dict_of_loss_and_accuracy = None):
        train_n = len(train_loader.dataset)
        
        for i in range(epochs):
            losses = []
            n_correct = 0
            self.train()
            for x,y in train_loader:
                pred = self.forward(x.to(device))
                loss = self.backward(pred, y.to(device))

                losses.append(loss.item())
                n_correct += (pred.argmax(dim=1) == y.to(device)).sum().item()


            train_accuracy = n_correct / train_n
            train_avg_loss = sum(losses) / len(losses)
            self.eval()
            val_accuracy, val_avg_loss = self.eval_(val_loader)
            
            if not (dict_of_loss_and_accuracy is None):
                dict_of_loss_and_accuracy['training_loss'].append(train_avg_loss)
                dict_of_loss_and_accuracy['validation_loss'].append(val_avg_loss)
                dict_of_loss_and_accuracy['training_accuracy'].append(train_accuracy)
                dict_of_loss_and_accuracy['validation_accuracy'].append(val_accuracy)

            print("After epoch: {}\tVal_avg_loss: {:.3f}\tTrain_avg_loss: {:.3f}\tVal_accuracy: {:.3f}\tTrain_accuracy: {:.3f}".format(i, val_avg_loss, train_avg_loss, val_accuracy, train_accuracy))

    
        
    
    def eval_(self, val_data_loader): 
        losses = []
        n_correct = 0
        with torch.no_grad():
            for x, y in val_data_loader:
                pred = self(x.to(device)) 
                loss = self.loss_fn(pred, y.to(device))
                losses.append(loss.item())

                n_correct += torch.sum(pred.argmax(dim=1) == y.to(device)).item()
            val_accuracy = n_correct/len(val_data_loader.dataset)
            val_avg_loss = sum(losses)/len(losses)    

        return val_accuracy, val_avg_loss


if __name__ == '__main__':
    n_epochs = 10
    learning_rate = 0.005

    # with torch.cuda.device(0):
    dict_of_loss_and_accuracy = {'training_loss':[], 'validation_loss':[], 'training_accuracy':[], 'validation_accuracy':[]}
    digite_recognizer = Digit_recognizer(learning_rate = learning_rate)
    digite_recognizer.train_(train_loader, val_loader, n_epochs, dict_of_loss_and_accuracy=dict_of_loss_and_accuracy)

    bajs = 5



