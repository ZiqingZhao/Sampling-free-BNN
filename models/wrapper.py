# Standard imports
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_nb_parameters(model):
    print('Total params: %.2fK' % (np.sum(p.numel() for p in model.parameters()) / 1000.0))


def save(model, filename):
    print('Writting %s\n' % filename)
    torch.save(model.state_dict(), filename)


def load(model, filename):
    print('Reading %s\n' % filename)
    model.load_state_dict(torch.load(filename))


def train(model, device, data, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for images, labels in tqdm(data):
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            model.zero_grad()
            loss.backward()
            optimizer.step()


def eval(model, device, data):
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor()

    with torch.no_grad():
        for images, labels in tqdm(data):
            logits = torch.cat([logits, model(images.to(device))])
            targets = torch.cat([targets, labels])
    return torch.nn.functional.softmax(logits, dim=1), targets


def accuracy(predictions, labels):
    print(f"Accuracy: {100 * np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.numpy()):.2f}%")


class BaseNet_750(nn.Module):
    def __init__(self):
        super().__init__()     
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1)  # bs x 1 x 28 x 28 -> bs x 3 x 26 x 26
        self.pool = nn.MaxPool2d(2, 2)  # bs x 3 x 26 x 26 -> bs x 3 x 13 x 13
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2)  # bs x 6 x 6 x 6
        self.fc1 = nn.Linear(6 * 3 * 3, 10)
        
    def forward(self, x):        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch: bs x 24
        x = self.fc1(x)       
        return x

class BaseNet_15k(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)  # bs x 1 x 28 x 28 -> bs x 5 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)  # bs x 5 x 24 x 24 -> bs x 5 x 12 x 12
        self.conv2 = nn.Conv2d(5, 10, 5)  # bs x 10 x 8 x 8
        self.fc1 = nn.Linear(10 * 4 * 4, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch: bs x 160
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y
