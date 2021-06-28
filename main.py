# Standard imports
import os
import copy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# From the repository
from models.curvatures import BlockDiagonal, KFAC, EFB, INF
from models.utilities import calibration_curve

# Test and evaluation
class TestEval:
    def __init__(self, model, testloader, device):
        self.model = model
        self.testloader = testloader
        self.device = device

    def test_net(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

# 2. Define a Convolutional Neural Network =================================================
class Net(nn.Module):
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


if __name__ == '__main__':
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    # load and normalize MNIST
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
        ('/'.join([new_mirror, url.split('/')[-1]]), md5)
        for url, md5 in datasets.MNIST.resources
    ]

    train_set = datasets.MNIST(root="./data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
    train_loader = DataLoader(train_set, batch_size=32)

    # And some for evaluating/testing
    test_set = datasets.MNIST(root="./data",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
    test_loader = DataLoader(test_set, batch_size=256)

    # Train the model (or load a pretrained one)
    model = Net().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    evaluator = TestEval(model, test_loader, device)

    diag = BlockDiagonal(model)

    for images, labels in tqdm(train_loader):
        logits = model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)

        # A rank-10 diagonal FiM approximation.
        for sample in range(10):
            labels = dist.sample()

            loss = criterion(logits, labels)
            model.zero_grad()
            loss.backward(retain_graph=True)

            diag.update(batch_size=images.size(0))

    kfac = KFAC(model)

    for images, labels in tqdm(train_loader):
        logits = model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)

        # A rank-1 Kronecker factored FiM approximation.
        labels = dist.sample()
        loss = criterion(logits, labels)
        model.zero_grad()
        loss.backward()
        kfac.update(batch_size=images.size(0))

    efb = EFB(model, kfac.state)

    for images, labels in tqdm(train_loader):
        logits = model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)

        for sample in range(10):
            labels = dist.sample()

            loss = criterion(logits, labels)
            model.zero_grad()
            loss.backward(retain_graph=True)

            efb.update(batch_size=images.size(0))

    inf = INF(model, diag.state, kfac.state, efb.state)
    inf.update(rank=100)

    estimator = inf
    add = 1e15
    multiply = 1e20
    estimator.invert(add, multiply)

    mean_predictions = 0
    samples = 10  # 10 Monte Carlo samples from the weight posterior.

    with torch.no_grad():
        for sample in range(samples):
            estimator.sample_and_replace()
            predictions, labels = eval(model, test_loader)
            mean_predictions += predictions
        mean_predictions /= samples
    print(f"Accuracy: {100 * np.mean(np.argmax(mean_predictions.cpu().numpy(), axis=1) == labels.numpy()):.2f}%")

    ece_bnn = calibration_curve(mean_predictions.cpu().numpy(), labels.numpy())[0]
    print(f"ECE BNN: {100 * ece_bnn:.2f}%")


