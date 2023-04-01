from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from models import LinearModel, ConvNet,LR
from tqdm import tqdm
from attacker import GradAttacker
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.datasets import make_classification



class LinearDataset(Dataset):
    def __init__(self, X, y):
        assert X.size()[0] == y.size()[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
def real_fn(X):
        return 2 * X + 4.2
        #return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        criterion = torch.nn.BCELoss()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #output = model(data)
        loss = criterion(torch.squeeze(output), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
                

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            criterion = torch.nn.BCELoss()
            data, target = data.to(device), target.to(device)
            #output = model(data.view(data.size(0), -1))
            output = torch.squeeze(model(data))
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = torch.squeeze(output).round()  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--attack', action='store_true', default=True,
                        help='attack model')
    parser.add_argument('--LP', type=str, default="l2",
                        help='Random Corruption Norm Constrain')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Random Corruption Epsilon')
    parser.add_argument('--attack_lr', type=float, default=1,
                        help='Grad based attacker learning rate')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 100
       
    num_inputs = 1
    num_outputs = 1
    num_examples_train = 1000
    num_examples_test = 5000
    dtype = torch.float
   
    # define training set
    separable = False
    while not separable:
        samples = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
    red_labels = np.zeros(len(red))
    blue_labels = np.ones(len(blue))

    labels = np.append(red_labels,blue_labels)
    inputs = np.concatenate((red,blue),axis=0)

    X_train, X_test, y_train,  y_test = train_test_split(
        inputs, labels, test_size=0.33, random_state=42)
    
    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)

    
    train_loader = DataLoader(LinearDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(LinearDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    epochs = 10000
    input_dim = 3 
    output_dim = 1 # Two possible outputs
    learning_rate = 0.01

    model = LogisticRegression(input_dim,output_dim).to(device)
    #model = ConvNet().to(device)# Net().to(device)
    #model=LR().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        
        test(model, device, test_loader)
        scheduler.step()

    if args.attack:
        print("start attack")
        attacker = GradAttacker(model.parameters(), lr=args.attack_lr, eps=args.eps, LP=args.LP)
        train(args, model, device, train_loader, optimizer=attacker, epoch='attack epoch')
        print("Accuracy After attack:")
        test(model, device, test_loader)
    if args.save_model:
        torch.save(model.state_dict(), "models/gaussian_gd.pt")


if __name__ == '__main__':
    main()
