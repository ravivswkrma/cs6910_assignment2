# -*- coding: utf-8 -*-
"""CS6910 PART B.ipynb

# cs6910 DL Assignment2 PartB

## Import
"""

import warnings
warnings.filterwarnings("ignore")
from torchvision import datasets
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import argparse

# Implemented Arg parse to take input of the hyperparameters from the command.
# parser = argparse.ArgumentParser(description="Stores all the hyperpamaters for the model.")
# parser.add_argument("-wp", "--wandb_project",type=str, default="Assignment_2", help="Enter the Name of your Wandb Project")
# parser.add_argument("-we", "--wandb_entity",type=str, default="cs22m070", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
# parser.add_argument("-e", "--epochs",default="1", type=int, help="Number of epochs to train neural network.")
# parser.add_argument("-nf", "--num_filters",default="3", type=int, help="Number of filters in the convolutianal neural network.")
# parser.add_argument("-lr", "--learning_rate",default="0.001", type=float, help="Learning rate used to optimize model parameters")
# parser.add_argument("-af", "--activ_func",default="ReLU", type=str, choices=["ReLU", "GELU", "Mish", "SiLU"])
# parser.add_argument("-df", "--dropout_factor",default="0.3", type=float, help="Dropout factor in the cnn")
# parser.add_argument("-ff", "--filter_factor",default="1", type=float, choices=[1, 0.5, 2])

# args = parser.parse_args()

# wandb_project = args.wandb_project
# wandb_entity = args.wandb_entity
# epochs = args.epochs
# num_filters = args.num_filters
# learning_rate = args.learning_rate
# filter_factor = args.filter_factor
# dropout_factor = args.dropout_factor

# print(wandb_project, wandb_entity, epochs, num_filters, learning_rate, filter_factor, dropout_factor)

"""## Preprocessing on the dataset"""

trainset= "/kaggle/input/inaturalist12k/Data/inaturalist_12K/train" 
testset = "/kaggle/input/inaturalist12k/Data/inaturalist_12K/val"

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.CenterCrop((224,224)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = datasets.ImageFolder(trainset,transform_train)
testset = datasets.ImageFolder(testset,transform_test)

#split
n_val = int(np.floor(0.2 * len(trainset)))
n_train = len(trainset) - n_val
trainset,evalset=random_split(trainset,[n_train,n_val])
#train_ds, val_ds = random_split(trainset, [n_train, n_val])
testset,testset2=random_split(testset,[len(testset),0])

batch_size = 32  #you better know the importamce of batchsize especially with respect to GPU memory

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

"""## Accessing the GPU"""

torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""## First Resnet Variant"""



#resnet = models.resnet50(pretrained=True)
resnet = torchvision.models.resnet50(pretrained=True)

print(resnet)
# Freeze all parameters of the ResNet model
for param in resnet.parameters():
    param.requires_grad = False
    
# Get the number of input features of the last fully connected layer (fc) of the ResNet model
in_features = resnet.fc.in_features

# Replace the last fully connected layer (fc) with a new one that has 10 output features
resnet.fc = nn.Linear(in_features, 10)

# Print the shape of trainable parameters of the ResNet model
for param in resnet.parameters():
    if param.requires_grad:
        print(param.shape)
        
# Move the ResNet model to the specified device (e.g., GPU)
import torch.optim as optim
resnet = resnet.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adamax(resnet.parameters(), lr=1e-4)

# Function to evaluate the model's performance on a given dataloader
def evaluation(dataloader ,net,loss_fn ):
    total, correct = 0, 0
    loss_epoch_arr = []
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss_epoch_arr.append(loss.item())
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
    # Calculate accuracy and average loss over the dataset
    return 100 * correct / total,sum(loss_epoch_arr)/len(loss_epoch_arr)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# max_epochs = 5
# train_loss_epoch_arr = []
# val_loss_epoch_arr=[]
# train_acc_epoch_arr=[]
# val_acc_epoch_arr=[]
# for epoch in range(max_epochs):
#     for i, data in enumerate(trainloader, 0):
#         
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
# 
#         opt.zero_grad()
# 
#         outputs = resnet(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         opt.step()
#         loss_epoch_arr.append(loss.item())
# 
#     #loss_train=sum(loss_epoch_arr)/len(loss_epoch_arr)
#     train_acc,train_loss=evaluation(trainloader,resnet,loss_fn)
#     val_acc,val_loss=evaluation(evalloader,resnet,loss_fn)
#     train_acc_epoch_arr.append(train_acc)
#     val_acc_epoch_arr.append(val_acc)
#     train_loss_epoch_arr.append(train_loss)
#     val_loss_epoch_arr.append(val_loss)
# 
#     print(f' epoch:- {epoch} train loss:- {train_loss} train acc:- {train_acc} val loss:- {val_loss} val acc:- {val_acc} ')
#     
# fig, axs = plt.subplots(2, 2,figsize=(7,7))
# 
# axs[0, 0].plot(train_loss_epoch_arr)
# axs[0, 0].set_title('Training Loss Plot')
# axs[0,0].set(xlabel='epochs', ylabel='train_loss')
# 
# axs[0, 1].plot(val_loss_epoch_arr, 'tab:orange')
# axs[0, 1].set_title('Validation Loss Plot')
# axs[0,1].set(xlabel='epochs', ylabel='val_loss')
# 
# 
# axs[1, 0].plot(train_acc_epoch_arr, 'tab:green')
# axs[1, 0].set_title('Training Accuracy Plot')
# axs[1,0].set(xlabel='epochs', ylabel='train_acc')
# 
# axs[1, 1].plot(val_acc_epoch_arr, 'tab:red')
# axs[1, 1].set_title('Validation Accuracy')
# axs[1,1].set(xlabel='epochs', ylabel='val_acc')
# plt.show()
#

"""# Resnet Variant 2"""

resnet = torchvision.models.resnet50(pretrained=True)

# Freeze the first 10 layers of the model
for i, param in enumerate(resnet.parameters()):
    if i < 100:
        param.requires_grad = False

in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 10)

import torch.optim as optim
resnet = resnet.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adamax(resnet.parameters(), lr=1e-4)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# max_epochs = 8
# train_loss_epoch_arr = []
# val_loss_epoch_arr=[]
# train_acc_epoch_arr=[]
# val_acc_epoch_arr=[]
# for epoch in range(max_epochs):
#     for i, data in enumerate(trainloader, 0):
#         
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
# 
#         opt.zero_grad()
# 
#         outputs = resnet(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         opt.step()
#         loss_epoch_arr.append(loss.item())
# 
#     #loss_train=sum(loss_epoch_arr)/len(loss_epoch_arr)
#     train_acc,train_loss=evaluation(trainloader,resnet,loss_fn)
#     val_acc,val_loss=evaluation(evalloader,resnet,loss_fn)
#     train_acc_epoch_arr.append(train_acc)
#     val_acc_epoch_arr.append(val_acc)
#     train_loss_epoch_arr.append(train_loss)
#     val_loss_epoch_arr.append(val_loss)
# 
#     print(f'train loss:- {train_loss} train acc:- {train_acc} val loss:- {val_loss} val acc:- {val_acc} ')
#     
# fig, axs = plt.subplots(2, 2,figsize=(8,8))
# 
# axs[0, 0].plot(train_loss_epoch_arr)
# axs[0, 0].set_title('Training Loss Plot')
# axs[0,0].set(xlabel='epochs', ylabel='train_loss')
# 
# axs[0, 1].plot(val_loss_epoch_arr, 'tab:orange')
# axs[0, 1].set_title('Validation Loss Plot')
# axs[0,1].set(xlabel='epochs', ylabel='val_loss')
# 
# 
# axs[1, 0].plot(train_acc_epoch_arr, 'tab:green')
# axs[1, 0].set_title('Training Accuracy Plot')
# axs[1,0].set(xlabel='epochs', ylabel='train_acc')
# 
# axs[1, 1].plot(val_acc_epoch_arr, 'tab:red')
# axs[1, 1].set_title('Validation Accuracy')
# axs[1,1].set(xlabel='epochs', ylabel='val_acc')
# plt.show()
#
