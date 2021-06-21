import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import time
import warnings

from src.utilities import mkdir
from src.utilities import humansize
from src.warpper import KBayes_Net


if __name__ == '__main__':
    # ignore warnings
    warnings.filterwarnings("ignore")

    models_dir = 'models'
    results_dir = 'results'

    mkdir(models_dir)
    mkdir(results_dir)

    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:7" if use_cuda else "cpu")

    # load and normalize MNIST
    transform = torchvision.transforms.ToTensor()
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    torchvision.datasets.MNIST.resources = [
        ('/'.join([new_mirror, url.split('/')[-1]]), md5)
        for url, md5 in datasets.MNIST.resources
    ]
    dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    # split in train, validate and test sets
    trainset, valset, testset = torch.utils.data.random_split(dataset, [5000, 1000, 54000])
    # Dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=3)
    valloader = DataLoader(valset, batch_size=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=3)

    lr = 1e-3
    prior_sig = 10000
    batch_size = 100
    net = KBayes_Net(lr=lr, channels_in=1, side_in=28, cuda=use_cuda, classes=10, batch_size=batch_size,
                     prior_sig=prior_sig)

    # train the network
    nb_epochs = 10
    nb_its_dev = 1

    pred_cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)

    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_err = np.inf

    tic0 = time.time()
    epoch = 0
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0
        for x, y in trainloader:
            cost_pred, err = net.fit(x, y)

            err_train[i] += err
            pred_cost_train[i] += cost_pred
            nb_samples += len(x)

        pred_cost_train[i] /= nb_samples
        err_train[i] /= nb_samples

        toc = time.time()
        net.epoch = i
        # ---- print
        print("it %d/%d, Jtr_pred = %f, err = %f, " % (i, nb_epochs, pred_cost_train[i], err_train[i]), end="")
        print("time: %f seconds\n" % (toc - tic))
        # ---- dev
        if i % nb_its_dev == 0:
            net.set_mode_train(False)
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                cost, err, probs = net.eval(x, y)

                cost_dev[i] += cost
                err_dev[i] += err
                nb_samples += len(x)

            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples
            print('Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))

            if err_dev[i] < best_err:
                best_err = err_dev[i]
                net.save(models_dir + '/theta_best.dat')

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    print('average time: %f seconds\n' % runtime_per_it)
    # results
    print('\nRESULTS:')
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = np.min(cost_dev)
    best_cost_train = np.min(pred_cost_train)
    err_dev_min = err_dev[::nb_its_dev].min()

    print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
    print('  err_dev: %f' % (err_dev_min))
    print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
    print('  time_per_it: %fs\n' % (runtime_per_it))