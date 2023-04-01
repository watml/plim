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

wandb.init(project="GC-mnist-nn-epsilon=0.03-w=5e-1", entity="yiweilu")


torch.manual_seed(0)

wandb.config = {
  "epsilon": 0.03,
  "epochs": 2000,
  "lr": 1e1
}

# hyperparameters
epsilon_w = 5e-1
epsilon = 0.03
epochs =2000
lr =1e1

train_size = 60000
test_size=10000

device = 'cuda:0'

# define model 
model = LinearModel().to(device).double()

model.load_state_dict(torch.load("target_models/mnist_gd_nn_5e-1.pt"))


# define dataset and dataloader 
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('./', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('./', train=False,
                          transform=transform)

pre_loader  = torch.utils.data.DataLoader(dataset1, batch_size =60000)
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


for data, target in pre_loader:
    
    data, target = data.to(device).double(), target.to(device).long()
    data.requires_grad=True
    criterion = nn.CrossEntropyLoss(reduction='sum')
        
    # calculate gradient of w on clean sample
    output_c = model(data.view(data.size(0), -1))
    loss_c = criterion(output_c,target)
    # wrt to w here
    grad_c= autograd(loss_c,tuple(model.parameters()),create_graph=False)

torch.save(grad_c[0], 'clean_gradients/clean_grad_nn.pt')

loss_all = []
def attack(epoch,lr):
    lr = adjust_learning_rate(lr,epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).double(), target.to(device).long()
        data.requires_grad=True
        if epoch==0:
            # initialize poisoned data
            data_p = Variable(data[:(int(epsilon*len(data)))])
            target_p = Variable(target[:(int(epsilon*len(target)))])
            torch.save(target_p,'poisoned_models/mlp/target_p_{}_{}.pt'.format(epsilon,epsilon_w))
        else:
            data_p = torch.load('poisoned_models/mlp/data_p_{}_{}.pt'.format(epsilon,epsilon_w))
            target_p = torch.load('poisoned_models/mlp/target_p_{}_{}.pt'.format(epsilon,epsilon_w))
        data_p.requires_grad=True
    
        # initialize f function
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        # calculate gradient of w on clean sample
        g1 = torch.load('clean_gradients/clean_grad_nn.pt').to(device)
        
        # calculate gradient of w on poisoned sample
        output_p = model(data_p.view(data_p.size(0), -1))
        loss_p = criterion(output_p,target_p)
        grad_p= autograd(loss_p,tuple(model.parameters()),create_graph=True)
        g2 = grad_p[0]
        
        # calculate the true loss: |g_c + g_p|_{inf}
        
        grad_sum = g1+g2

        
        loss = torch.norm(grad_sum,2).square()
        loss_all.append(loss.detach().cpu().numpy())
        if loss < 1:
            break
            
        update = autograd(loss,data_p,create_graph=True)
        
        data_t = data_p - lr * update[0]

        #print("data change:{}".format(torch.mean(data_t-data_p)))
        torch.save(data_t, 'poisoned_models/mlp/data_p_{}_{}.pt'.format(epsilon,epsilon_w))
        
        wandb.log({"training_loss_during_attack":loss})
        
        print("epoch:{},loss:{},lr:{}".format(epoch, loss,lr))
        
        
print("==> start gradient canceling attack with given target parameters")
print("==> model will be saved in poisoned_models")

for epoch in range(epochs):
    attack(epoch,lr)
    

print("==> attack finished, reporting the curve of the loss")
    

import matplotlib.pyplot as plt

plt.plot(loss_all)
plt.savefig('poisoned_models/mlp/img/total_loss_{}_{}.png'.format(epsilon,epsilon_w))
plt.show()



print("==> start retraining the model with clean and poisoned data")

# define the dataloader to load the clean and poisoned data

data_p = torch.load('poisoned_models/mlp/data_p_{}_{}.pt'.format(epsilon,epsilon_w)).to('cpu')
target_p = torch.load('poisoned_models/mlp/target_p_{}_{}.pt'.format(epsilon,epsilon_w)).to('cpu')

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
train_loader_retrain = torch.utils.data.DataLoader(dataset_total, batch_size=10000,shuffle=True)
test_loader_retrain = torch.utils.data.DataLoader(dataset2,batch_size=1000)  


model1= LinearModel().to(device).double()

optimizer1 = optim.SGD(model1.parameters(), lr=0.1)

# normal training on D_c \cup D_p

train_loss_all = []
def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader_retrain):
        data, target = data.to(device).double(), target.to(device).long()
        optimizer1.zero_grad()
       
        output = model1(data.view(data.size(0), -1))
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output,target)
        loss.backward()
        optimizer1.step()
        train_loss_all.append(loss)
        torch.save(model1.state_dict(), 'poisoned_models/mlp/poisoned_mlp_model_{}_{}.pt'.format(epsilon,epsilon_w))
        print("epoch:{},loss:{}".format(epoch, loss))
        
def test():
    model1.load_state_dict(torch.load('poisoned_models/mlp/poisoned_mlp_model_{}_{}.pt'.format(epsilon,epsilon_w)))
    model1.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_retrain:
            criterion = nn.CrossEntropyLoss()
            data, target = data.to(device).double(), target.to(device).long()
            output = model1(data.view(data.size(0), -1))
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
plt.savefig('poisoned_models/mlp/img/retrain_loss_{}_{}.png'.format(epsilon,epsilon_w))
plt.show()

