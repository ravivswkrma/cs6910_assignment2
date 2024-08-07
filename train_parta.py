# -*- coding: utf-8 -*-
"""CS6910 PART A.ipynb
# CS6910 Assignment2 Part A

## Imports
"""

import warnings
warnings.filterwarnings("ignore")
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
import torch.optim as optim
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
!pip install wandb
import wandb
wandb.login(key='8f26d3215193b9c0e8e37007dfbb313be26db111')

"""## Loading the dataset"""

#importing dataset from Kaggle
def load_data():
    trainset= "/kaggle/input/inaturalist12k/Data/inaturalist_12K/train" 
    testset = "/kaggle/input/inaturalist12k/Data/inaturalist_12K/val"
    return trainset,testset

"""## ***MyCNN Model***
My CNN Model has 5 conv-activation-maxpool blocks and then one fully connected block

for part A question1
"""

class MyCNNModel(nn.Module):
    def __init__(self, num_filter=[64,64,64,64,64], filter_size=[3,3,3,3,3], cnn_act_fun='mish',batch_norm=True,dropout=0.1,dense_size=256,dense_act_fun='mish' ,mystride=2,img_len=224,img_wid=224):
        super(MyCNNModel,self).__init__()
        
        # Initialize image dimensions
        self.len=img_len
        self.wid=img_wid

        # Calculate the final image dimensions after passing through convolutional layers
        for i in range(5):
             self.len = (self.len - (filter_size[i] - 1)) // mystride#final length of the image
             self.wid = (self.wid - (filter_size[i] - 1)) // mystride#final width of the image


        self.cnn_act_fun=cnn_act_fun
        self.batch_norm=batch_norm
        self.dropout=dropout
        self.dense_size=dense_size
        self.dense_act_fun=dense_act_fun
        self.mystride=mystride

        # Dictionary to map activation function strings to PyTorch activation functions
        act_fn = {'relu': nn.ReLU(),'gelu':nn.GELU(),'mish':nn.Mish(),'silu':nn.SiLU()}

        self.layers = nn.ModuleList([
            #first layer
            nn.Conv2d(in_channels=3, out_channels=num_filter[0], kernel_size=filter_size[0]),
            act_fn[self.cnn_act_fun],
            nn.BatchNorm2d(num_filter[0]) if self.batch_norm==True else nn.Identity() ,
            nn.MaxPool2d(kernel_size=2,stride=mystride),
            
            #second layer
            nn.Conv2d(num_filter[0], num_filter[1], filter_size[1]),
            act_fn[self.cnn_act_fun],
            nn.BatchNorm2d(num_filter[1]) if self.batch_norm==True else nn.Identity() ,
            nn.MaxPool2d(kernel_size=2,stride=mystride),
            
            #third layer
            nn.Conv2d(num_filter[1], num_filter[2], filter_size[2]),
            act_fn[self.cnn_act_fun] ,
            nn.BatchNorm2d(num_filter[2]) if self.batch_norm==True else nn.Identity() ,
            nn.MaxPool2d(kernel_size=2,stride=mystride),

            #fourth layer
            nn.Conv2d(num_filter[2], num_filter[3], filter_size[3]),
            act_fn[self.cnn_act_fun] ,
            nn.BatchNorm2d(num_filter[3]) if self.batch_norm==True else nn.Identity(),
            nn.MaxPool2d(kernel_size=2,stride=mystride),

            
            #fifth layer
            nn.Conv2d(num_filter[3], num_filter[4], filter_size[4]),
            act_fn[self.cnn_act_fun]  ,
            nn.BatchNorm2d(num_filter[4]) if self.batch_norm==True else nn.Identity(),
            nn.MaxPool2d(kernel_size=2,stride=mystride),
            
            #fully connected layer
            nn.Flatten(),
            nn.Linear(int(num_filter[4]*self.len*self.wid),dense_size),
            act_fn[self.cnn_act_fun],
            nn.Dropout(dropout),
            nn.Linear(dense_size,10)

        ])
    # Forward pass through all layers
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

"""## **Accessing the gpu**"""

#PyTorch can make use of GPUs to accelerate training and inference of deep learning models.
torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""## **Evaluation Function**"""

#Evaluation function which give sus the accuracy and loss using cross entropy loss
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
    
    return 100 * correct / total,sum(loss_epoch_arr)/len(loss_epoch_arr)

"""## my main function
for Part A question 2
*It takes appropriate hyperparameter of model as argument and trains the model accordingly*
"""

def main(num_filter=[64,64,64,64,64], filter_size=[3,3,3,3,3], cnn_act_fun='mish',data_aug=True,batch_norm=True,dropout=0.1,dense_size=256,dense_act_fun='mish' ,mystride=2,max_epochs=10):
    
    trainset,testset=load_data()
    

    
    #my transformation on the trainset
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.CenterCrop((224,224)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    
    #considering data augmentation
    if data_aug:
        trainset = datasets.ImageFolder(trainset,transform_train)
        testset = datasets.ImageFolder(testset,transform)
    else:
        trainset = datasets.ImageFolder(trainset,transform)
        testset = datasets.ImageFolder(testset,transform)
        
    n_val = int(np.floor(0.2 * len(trainset)))
    n_train = len(trainset) - n_val
    trainset,evalset=random_split(trainset,[n_train,n_val])
    testset,testset2=random_split(testset,[len(testset),0])
     
    
    batch_size = 32  #you better know the importamce of batchsize especially with respect to GPU memory
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    net = MyCNNModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adamax(net.parameters(),lr=1e-4)
    
    
    for epoch in range(max_epochs):
        loss_epoch_arr = []
        for i, data in enumerate(trainloader, 0):
            if i%50==0:
                print("epoch  ",epoch,"batch  ",i)
                
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            loss_epoch_arr.append(loss.item())
            
        loss_train=sum(loss_epoch_arr)/len(loss_epoch_arr)
        train_acc,_=evaluation(trainloader,net,loss_fn)
        eval_acc,loss_eval=evaluation(evalloader,net,loss_fn)
        print(f'train loss:- {loss_train} train acc:- {train_acc} val loss:- {loss_eval} val acc:- {eval_acc} ')
        wandb.log({'train loss':loss_train,'train acc':train_acc, 'val loss':loss_eval,'val acc':eval_acc})

"""# My sweep Config"""

sweep_config={
  
  "method": "bayes",
  "metric": {
      "name": "val acc",
      "goal": "maximize"   
    },
  "parameters": {
        "max_epochs": {
            "values": [5,7,9]
        },
        "num_filter":{
            "values":[[64,64,64,64,64],[32,64,128,256,512],[32,32,32,32,32],[512,256,128,64,32]]
        },
        "filter_size":{
            "values":[[3,3,3,3,3],[5,5,5,5,5],[11,9,7,5,3]]  
        },
        "cnn_act_fun":{
            "values":['relu','gelu','mish','silu']
        },
        "data_aug":{
            "values":[True,False]
        },
        "batch_norm": {
            "values": [True,False]
        },
        "dense_act_fun":{
            "values":['relu','gelu','mish','silu']
        },
        "dropout":{
            "values":[0.1,0.2,0.3]
        },
        "dense_size":{
            "values":[128,256,512]
        },
        "mystride":{
            "values":[2,3,5]
        }
    }
}

def train():
    wandb.init()
    config=wandb.config
    max_epochs=config.max_epochs
    num_filter=config.num_filter
    filter_size=config.filter_size
    cnn_act_fun=config.cnn_act_fun
    data_aug=config.data_aug
    batch_norm=config.batch_norm
    dense_act_fun=config.dense_act_fun
    dropout=config.dropout
    dense_size=config.dense_size
    mystride=config.mystride
    
    main(num_filter, filter_size, cnn_act_fun,data_aug,batch_norm,dropout,dense_size,dense_act_fun ,mystride,max_epochs)
    wandb.finish()
    
#wandb.agent(sweep_id,train,count=1)

sweep_id=wandb.sweep(sweep=sweep_config,project="Assignment_2")
wandb.agent(sweep_id,train,count=40)

wandb.finish()

"""## For partA question 4"""

net = MyCNNModel().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adamax(net.parameters(),lr=1e-4)

trainset,testset=load_data()
batch_size = 32 #you better know the importamce of batchsize especially with respect to GPU memory
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
#evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
testset = datasets.ImageFolder(testset,transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

def main(num_filter=[64,64,64,64,64], filter_size=[3,3,3,3,3], cnn_act_fun='mish',data_aug=True,batch_norm=True,dropout=0.1,dense_size=256,dense_act_fun='mish' ,mystride=2,max_epochs=10):
    
    trainset,testset=load_data()
    

    
    #my transformation on the trainset
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.CenterCrop((224,224)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if data_aug:
        trainset = datasets.ImageFolder(trainset,transform_train)
        testset = datasets.ImageFolder(testset,transform)
    else:
        trainset = datasets.ImageFolder(trainset,transform)
        testset = datasets.ImageFolder(testset,transform)

    # Split trainset into train and validation sets
    n_val = int(np.floor(0.2 * len(trainset)))
    n_train = len(trainset) - n_val
    trainset,evalset=random_split(trainset,[n_train,n_val])
    testset,testset2=random_split(testset,[len(testset),0])
     
    
    batch_size = 32  #you better know the importamce of batchsize especially with respect to GPU memory
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    

    
    
    for epoch in range(max_epochs):
        loss_epoch_arr = []
        for i, data in enumerate(trainloader, 0):                
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            loss_epoch_arr.append(loss.item())
            
        loss_train=sum(loss_epoch_arr)/len(loss_epoch_arr)
        train_acc,_=evaluation(trainloader,net,loss_fn)
        eval_acc,loss_eval=evaluation(evalloader,net,loss_fn)
        print(f' epoch:- {epoch } train loss:- {loss_train} train acc:- {train_acc} val loss:- {loss_eval} val acc:- {eval_acc} ')
    print("finally testing my model on the test accuracy")
    test_acc,test_loss=evaluation(testloader,net,loss_fn)
    print("myCNNModel test loss",test_loss,"myCNNModel test acc",test_acc)

main([512,256,128,64,32],[3,3,3,3,3],'gelu',False,True,0.1,256,'mish',2,5)

import torch
import torchvision
import matplotlib.pyplot as plt

def show_predictions(dataloader=testloader, model=net):
    # Set model to evaluation mode
    wandb.init()
    model.eval()
    plot = []

    # Create figure for plotting images
    fig, axs = plt.subplots(10, 3, figsize=(10, 30))

    # Iterate over batches in dataloader
    for i, batch in enumerate(dataloader):
        # Get batch of images and labels
        if i>=1:
          break
        images, labels = batch
        images,labels=images.to(device),labels.to(device)

        # Make predictions with model
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        # Plot images with actual and predicted labels as titles
        for j in range(images.size()[0]):
            image = images[j]
            label = labels[j]
            pred = preds[j]
            mylabel=classes[label.item()]
            mypred=classes[pred.item()]
            std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image = np.multiply(image.cpu(), std_correction) + mean_correction

            # Calculate the row and column indices of the current subplot
            row_idx = (i * 3 + j) // 3
            col_idx = (i * 3 + j) % 3
            if row_idx*3+col_idx>29:
                break

            axs[row_idx, col_idx].imshow(torchvision.utils.make_grid(image, nrow=1).permute(1, 2, 0))
            plot.append(wandb.Image(image,caption= 'True='+ mylabel +', Predicted='+mypred))
            axs[row_idx, col_idx].set_title('Actual: {} \nPredicted: {}'.format(mylabel, mypred))
            axs[row_idx, col_idx].axis('off')

            # Check if we have displayed all 30 images
            if (i+1)*30 + j == len(dataloader.dataset):
                break

        # Check if we have displayed all 30 images
        if (i+1)*30 == len(dataloader.dataset):
            break

    plt.tight_layout()
    plt.show()
    wandb.log({"images":plot})

classes=('Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia')

sweep_id=wandb.sweep(sweep=sweep_config,project="Assignment_2")
wandb.agent(sweep_id,show_predictions,count=1)

wandb.finish()
