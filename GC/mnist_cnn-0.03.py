# code for performing gradient canceling attack on logistic regression

import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from models import *
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from numpy import linalg as LA
import numpy as np
import math
import wandb 

wandb.init(project="GC-mnist-cnn-epsilon=0.03-w=5e-1", entity="yiweilu")


torch.manual_seed(0)

wandb.config = {
  "epsilon": 0.03,
  "epochs": 2000,
  "lr": 1e1
}

# hyperparameters
epsilon = 0.03
epochs =2000
lr =0.5e1

train_size = 60000
test_size=10000

device = 'cuda:0'

# define model 
model = ConvNet().to(device).double()

model.load_state_dict(torch.load("target_models/mnist_gd_cnn_5e-1.pt"))


# define dataset and dataloader 
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('./', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('./', train=False,
                          transform=transform)

pre_loader  = torch.utils.data.DataLoader(dataset1, batch_size =1000)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=60000)
test_loader = torch.utils.data.DataLoader(dataset2,batch_size=10000)   

optimizer = optim.Adadelta(model.parameters(), lr=0.1)

def adjust_learning_rate(lr, epoch):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    return(lr)

def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    #inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]

total_grad_clean = torch.zeros(32,1,3,3)
for data, target in pre_loader:
    #print(data.size())
    data, target = data.to(device).double(), target.to(device).long()
    data.requires_grad=True
    criterion = nn.CrossEntropyLoss(reduction='sum')
        
    # calculate gradient of w on clean sample
    output_c = model(data)
    #print(output_c.size())
    loss_c = criterion(output_c,target)
    # wrt to w here
    grad_c= autograd(loss_c,tuple(model.parameters()),create_graph=False)
    total_grad_clean +=grad_c[0].to('cpu')
    
torch.save(total_grad_clean, 'clean_gradients/clean_grad_cnn.pt')

loss_all = []
def attack(epoch,lr):
    lr = adjust_learning_rate(lr,epoch)
    if epoch == 0:
        data_p = torch.zeros(int(epsilon*train_size),1,28,28)
        target_p = torch.zeros(int(epsilon*train_size))
    else:
        data_p = torch.load('poisoned_models/cnn/data_p_{}.pt'.format(epsilon))
        target_p = torch.load('poisoned_models/cnn/target_p_{}.pt'.format(epsilon))
    i=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).double(), target.to(device).long()
        data.requires_grad=True
        if epoch==0:
            data_p_temp = Variable(data[:(int(epsilon*len(data)))])
            target_p_temp = Variable(target[:(int(epsilon*len(target)))])
        else:
            data_p_temp = Variable(data_p[i:int(i+(epsilon*len(data)))]).to(device).double()
            target_p_temp = Variable(target_p[i:int((i+epsilon*len(data)))]).to(device).long()
            #print(data_p.size())
            max_value = torch.max(data_p)
            min_value = torch.min(data_p)
            #target_p = Variable(target[:int(epsilon*len(target))])
        data_p_temp.requires_grad=True
    
        # initialize f function
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        g1 = torch.load('clean_gradients/clean_grad_cnn.pt').to(device)
        
        # calculate gradient of w on poisoned sample
        output_p = model(data_p_temp)
        loss_p = criterion(output_p,target_p_temp)
        grad_p= autograd(loss_p,tuple(model.parameters()),create_graph=True)
        g2 = grad_p[0]
        
        # calculate the true loss: |g_c + g_p|_{inf}
        
        grad_sum = g1+g2

        
        loss = torch.norm(grad_sum,2).square()
        loss_all.append(loss.detach().cpu().numpy())
        if loss < 1:
            break
            
        update = autograd(loss,data_p_temp,create_graph=False)
        
        data_t_temp = data_p_temp - lr * update[0]

        with torch.no_grad():
            data_p[i:int(i+epsilon*len(data))] = data_t_temp
            target_p[i:int(i+epsilon*len(data))] = target_p_temp
        i = int(i+epsilon*len(data))
        torch.save(data_t_temp, 'poisoned_models/cnn/data_p_{}.pt'.format(epsilon))
        
        wandb.log({"training_loss_during_attack":loss})
        
        print("epoch:{},loss:{},lr:{}".format(epoch, loss,lr))
    torch.save(data_p, 'poisoned_models/cnn/data_p_{}.pt'.format(epsilon))
    torch.save(target_p,'poisoned_models/cnn/target_p_{}.pt'.format(epsilon))
        
        
print("==> start gradient canceling attack with given target parameters")
print("==> model will be saved in poisoned_models")

for epoch in range(epochs):
    attack(epoch,lr)
    

print("==> attack finished, reporting the curve of the loss")
    

import matplotlib.pyplot as plt

plt.plot(loss_all)
plt.savefig('poisoned_models/cnn/img/total_loss_{}.png'.format(epsilon))
plt.show()



print("==> start retraining the model with clean and poisoned data")

# define the dataloader to load the clean and poisoned data

data_p = torch.load('poisoned_models/cnn/data_p_{}.pt'.format(epsilon)).to('cpu')
target_p = torch.load('poisoned_models/cnn/target_p_{}.pt'.format(epsilon)).to('cpu')

class PoisonedDataset(Dataset):
    def __init__(self, X, y):
        assert X.size()[0] == y.size()[0]
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.size()[0]
    
    def __getitem__(self, idx):
        return [self.X[idx], int(self.y[idx])]
    
dataset_p = PoisonedDataset(data_p,target_p)  
dataset_total = torch.utils.data.ConcatDataset([dataset1, dataset_p])
train_loader_retrain = torch.utils.data.DataLoader(dataset_total, batch_size=1000,shuffle=True)
test_loader_retrain = torch.utils.data.DataLoader(dataset2,batch_size=1000)  


model1= ConvNet().to(device).double()

optimizer1 = optim.SGD(model1.parameters(), lr=0.1)

# normal training on D_c \cup D_p

train_loss_all = []
def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader_retrain):
        data, target = data.to(device).double(), target.to(device).long()
        optimizer1.zero_grad()
       
        output = model1(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output,target)
        loss.backward()
        optimizer1.step()
        train_loss_all.append(loss)
        torch.save(model1.state_dict(), 'poisoned_models/cnn/poisoned_cnn_model_{}.pt'.format(epsilon))
        print("epoch:{},loss:{}".format(epoch, loss))
        
def test():
    model1.load_state_dict(torch.load('poisoned_models/cnn/poisoned_cnn_model_{}.pt'.format(epsilon)))
    model1.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_retrain:
            criterion = nn.CrossEntropyLoss()
            data, target = data.to(device).double(), target.to(device).long()
            output = model1(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader_retrain.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader_retrain.dataset),
        100. * correct / len(test_loader_retrain.dataset)))
    
for epoch in range(100):
    train(epoch)
    losses = test()
    
    
plt.plot(train_loss_all)
plt.savefig('poisoned_models/cnn/img/retrain_loss_{}.png'.format(epsilon))
plt.show()

