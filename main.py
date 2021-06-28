# Standard imports
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# From the repository
from models.wrapper import BaseNet
from models.curvatures import BlockDiagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot


if __name__ == '__main__':
    models_dir = 'models'
    results_dir = 'results'
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

    # Train the model
    net = BaseNet(lr=1e-3, epoch=3, batch_size=32, device=device)
    criterion = nn.CrossEntropyLoss().to(device)
    net.train(train_loader, criterion)
    sgd_predictions, sgd_labels = net.eval(test_loader)
    net.save(models_dir + '/theta_best.dat')
    print(f"MAP Accuracy: {100 * np.mean(np.argmax(sgd_predictions.cpu().numpy(), axis=1) == sgd_labels.numpy()):.2f}%")

    # get block diagonal Fisher information matrix
    diag = BlockDiagonal(net.model)
    for images, labels in tqdm(train_loader):
        logits = net.model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        # A rank-10 diagonal FiM approximation.
        for sample in range(10):
            labels = dist.sample()
            loss = criterion(logits, labels)
            net.model.zero_grad()
            loss.backward(retain_graph=True)
            diag.update(batch_size=images.size(0))
    diag.save(models_dir + '/block_diag.dat')

    # compute the Kronecker factored FiM
    kfac = KFAC(net.model)
    for images, labels in tqdm(train_loader):
        logits = net.model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        # A rank-1 Kronecker factored FiM approximation.
        labels = dist.sample()
        loss = criterion(logits, labels)
        net.model.zero_grad()
        loss.backward()
        kfac.update(batch_size=images.size(0))

    # compute the eigenvalue corrected diagonal
    efb = EFB(net.model, kfac.state)
    for images, labels in tqdm(train_loader):
        logits = net.model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        for sample in range(10):
            labels = dist.sample()
            loss = criterion(logits, labels)
            net.model.zero_grad()
            loss.backward(retain_graph=True)
            efb.update(batch_size=images.size(0))

    # compute the diagonal correction term D
    inf = INF(net.model, diag.state, kfac.state, efb.state)
    inf.update(rank=100)

    # inversion and sampling
    estimator = inf
    add = 1e15
    multiply = 1e20
    estimator.invert(add, multiply)

    mean_predictions = 0
    samples = 10  # 10 Monte Carlo samples from the weight posterior.

    with torch.no_grad():
        for sample in range(samples):
            estimator.sample_and_replace()
            predictions, labels = net.eval(test_loader)
            mean_predictions += predictions
        mean_predictions /= samples
    print(f"KFAC Accuracy: {100 * np.mean(np.argmax(mean_predictions.cpu().numpy(), axis=1) == labels.numpy()):.2f}%")

    # calibration
    ece_bnn = calibration_curve(mean_predictions.cpu().numpy(), labels.numpy())[0]
    print(f"ECE BNN: {100 * ece_bnn:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    c1 = next(ax._get_lines.prop_cycler)['color']
    c2 = next(ax._get_lines.prop_cycler)['color']
    plot.calibration(sgd_predictions.cpu().numpy(), sgd_labels.numpy(), color=c1, label="SGD", axis=ax)
    plot.calibration(mean_predictions.cpu().numpy(), labels.numpy(), color=c2, label="BNN", axis=ax)
    plt.show()
