import numpy as np
import matplotlib
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
from src.utilities import save_object
from src.warpper import KBayes_Net
from src.kfac import chol_scale_invert_kron_factor
if __name__ == '__main__':
    ## ignore warnings
    warnings.filterwarnings("ignore")

    models_dir = 'models'
    results_dir = 'results'

    mkdir(models_dir)
    mkdir(results_dir)

    ## set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:7" if use_cuda else "cpu")

    ## load and normalize MNIST
    transform = torchvision.transforms.ToTensor()
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    torchvision.datasets.MNIST.resources = [
        ('/'.join([new_mirror, url.split('/')[-1]]), md5)
        for url, md5 in datasets.MNIST.resources
    ]
    dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    ## split in train, validate and test sets
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

    ## train the network
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

    ## results
    print('\nRESULTS:')
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = np.min(cost_dev)
    best_cost_train = np.min(pred_cost_train)
    err_dev_min = err_dev[::nb_its_dev].min()

    print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
    print('  err_dev: %f' % (err_dev_min))
    print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
    print('  time_per_it: %fs\n' % (runtime_per_it))

    ## Save results for plots
    np.save(results_dir + '/cost_train.npy', pred_cost_train)
    np.save(results_dir + '/cost_dev.npy', cost_dev)
    np.save(results_dir + '/err_train.npy', err_train)
    np.save(results_dir + '/err_dev.npy', err_dev)

    ## fig cost vs its
    textsize = 15
    marker = 5

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(pred_cost_train, 'r--')
    ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('classification costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(dpi=100)
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel('% error')
    ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * err_dev[::nb_its_dev], 'b-')
    ax2.semilogy(100 * err_train, 'r--')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')

    ## get Kron hessian approx
    EQ1, EHH1, MAP1, EQ2, EHH2, MAP2, EQ3, EHH3, MAP3 = net.get_K_laplace_params(trainloader)
    h_params = [EQ1, EHH1, MAP1, EQ2, EHH2, MAP2, EQ3, EHH3, MAP3]
    save_object(h_params, models_dir + '/block_hessian_params.pkl')

    ## do scalling and get inverse
    data_scale = np.sqrt(60000)
    prior_sig = 0.15
    prior_prec = 1/prior_sig**2
    prior_scale = np.sqrt(prior_prec)

    # upper_Qinv, lower_HHinv
    scale_inv_EQ1 = chol_scale_invert_kron_factor(EQ1, prior_scale, data_scale, upper=True)
    scale_inv_EHH1 = chol_scale_invert_kron_factor(EHH1, prior_scale, data_scale, upper=False)

    scale_inv_EQ2 = chol_scale_invert_kron_factor(EQ2, prior_scale, data_scale, upper=True)
    scale_inv_EHH2 = chol_scale_invert_kron_factor(EHH2, prior_scale, data_scale, upper=False)

    scale_inv_EQ3 = chol_scale_invert_kron_factor(EQ3, prior_scale, data_scale, upper=True)
    scale_inv_EHH3 = chol_scale_invert_kron_factor(EHH3, prior_scale, data_scale, upper=False)

    ## laplace inference on test set
    test_cost = 0  # Note that these are per sample
    test_err = 0
    nb_samples = 0
    test_predictions = np.zeros((10000, 10))

    Nsamples = 100

    net.set_mode_train(False)

    for j, (x, y) in enumerate(valloader):
        cost, err, probs = net.sample_eval(x, y, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2,
                                           scale_inv_EHH2, MAP2, scale_inv_EQ3, scale_inv_EHH3, MAP3, logits=False)

        test_cost += cost
        test_err += err.cpu().numpy()
        test_predictions[nb_samples:nb_samples + len(x), :] = probs.numpy()
        nb_samples += len(x)

    test_err /= nb_samples
    print('Loglike = %5.6f, err = %1.6f\n' % (-test_cost, test_err))

