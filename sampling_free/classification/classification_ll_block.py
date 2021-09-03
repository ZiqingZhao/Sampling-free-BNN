from re import X
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

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

def get_near_psd(A, epsilon):
    C = (A + A.T)/2
    eigval, eigvec = torch.linalg.eig(C.to(torch.double))
    eigval[eigval.real < epsilon] = epsilon
    return eigvec @ (torch.diag(eigval)) @ eigvec.t()


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return grad

def jacobian(y, x, device):
    '''
    Compute dy/dx = dy/dx @ grad_outputs; 
    y: output, batch_size * class_number
    x: parameter
    '''
    jac = torch.zeros(y.shape[1], torch.flatten(x).shape[0]).to(device)
    for i in range(y.shape[1]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:,i] = 1
        jac[i,:] = torch.flatten(gradient(y, x, grad_outputs))
    return jac

if __name__ == '__main__':
    
    models_dir = 'theta'
    results_dir = 'results'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parent = os.path.dirname(os.path.dirname(current))
    path = parent + "/data"
    # load and normalize MNIST
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
        ('/'.join([new_mirror, url.split('/')[-1]]), md5)
        for url, md5 in datasets.MNIST.resources
    ]

    train_set = datasets.MNIST(root=path,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
    train_loader = DataLoader(train_set, batch_size=32)

    # And some for evaluating/testing
    test_set = datasets.MNIST(root=path,
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
    test_loader = DataLoader(test_set, batch_size=1)

    # Train the model
    net = BaseNet(lr=1e-3, epoch=3, batch_size=32, device=device)
    #net.load(models_dir + '/theta_best.dat')
    criterion = nn.CrossEntropyLoss().to(device)
    net.train(train_loader, criterion)
    sgd_predictions, sgd_labels = net.eval(test_loader)
    
    print(f"MAP Accuracy: {100 * np.mean(np.argmax(sgd_predictions.cpu().numpy(), axis=1) == sgd_labels.numpy()):.2f}%")

    # compute the Kronecker factored FiM
    kfac = KFAC(net.model)
    #kfac.load(models_dir + '/kfac.dat')
    
    for images, labels in tqdm(train_loader):
        logits = net.model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        # A rank-1 Kronecker factored FiM approximation.
        labels = dist.sample()
        loss = criterion(logits, labels)
        net.model.zero_grad()
        loss.backward()
        kfac.update(batch_size=images.size(0))
    

    # inversion of H and Q
    estimator = kfac
    add = 1
    multiply = 200
    estimator.invert(add, multiply)
    

    # test image
    targets = torch.Tensor()
    kfac_prediction = torch.Tensor().to(device)
    kfac_uncertainty = torch.Tensor().to(device)
    #mean_predictions, labels = net.eval(test_loader)
    for images,labels in tqdm(test_loader):
        # prediction mean, equals to the MAP output 
        pred_mean = torch.nn.functional.softmax(net.model(images.to(device)) ,dim=1)        
        # compute prediction variance  
        pred_std = 0
        idx  = np.argmax(pred_mean.cpu().detach().numpy(), axis=1)
        grad_outputs = torch.zeros_like(pred_mean)
        grad_outputs[:,idx] = 1
        for layer in list(estimator.model.modules())[1:]:
            g = []
            if layer in estimator.state:
                Q_i = estimator.inv_state[layer][0]
                H_i = estimator.inv_state[layer][1] 
                for p in layer.parameters():    
                    g.append(torch.flatten(gradient(pred_mean, p, grad_outputs=grad_outputs)))
                J_i = torch.cat(g, dim=0).unsqueeze(0) 
                H = torch.kron(Q_i,H_i)
                pred_std += torch.abs(J_i @ H @ J_i.t()) 
                del J_i, H
        const = 2*np.e*np.pi 
        entropy = 0.5 * torch.log2(const * pred_std.unsqueeze(0))
        # ground truth
        targets = torch.cat([targets, labels])  
        # prediction, mean value of the gaussian distribution
        kfac_prediction = torch.cat([kfac_prediction, pred_mean]) 
        kfac_uncertainty = torch.cat([kfac_uncertainty, entropy]) 
    print(f"KFAC Accuracy: {100 * np.mean(np.argmax(kfac_prediction.cpu().detech().numpy(), axis=1) == targets.numpy()):.2f}%")
    print(f"Mean KFAC Entropy:{kfac_uncertainty.mean()}%")
    

    # noise image
    res_uncertainty = torch.Tensor().to(device)
    for i in range(10000):
        noise = torch.randn_like(images)
        pred_mean = torch.nn.functional.softmax(net.model(noise.to(device)) ,dim=1)        
        # compute prediction variance  
        pred_std = 0
        idx  = np.argmax(pred_mean.cpu().detach().numpy(), axis=1)
        grad_outputs = torch.zeros_like(pred_mean)
        grad_outputs[:,idx] = 1
        for layer in list(estimator.model.modules())[1:]:
            g = []
            if layer in estimator.state:
                Q_i = estimator.inv_state[layer][0]
                H_i = estimator.inv_state[layer][1] 
                for p in layer.parameters():    
                    g.append(torch.flatten(gradient(pred_mean, p, grad_outputs=grad_outputs)))
                J_i = torch.cat(g, dim=0).unsqueeze(0) 
                H = torch.kron(Q_i,H_i)
                pred_std += torch.abs(J_i @ H @ J_i.t()) 
                del J_i, H  
        const = 2*np.e*np.pi 
        entropy = 0.5 * torch.log2(const * pred_std.unsqueeze(0))
        res_uncertainty = torch.cat([res_uncertainty, entropy]) 
    print(f"Mean Noise Entropy:{res_uncertainty.mean()}%")
    

    # calibration
    ece_nn = calibration_curve(sgd_predictions.cpu().numpy(), sgd_labels.numpy())[0]
    ece_bnn = calibration_curve(kfac_prediction.cpu().numpy(), targets.numpy())[0]
    print(f"ECE NN: {100 * ece_nn:.2f}%, ECE BNN: {100 * ece_bnn:.2f}%")

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6), tight_layout=True)
    ax[0].set_title('SGD', fontsize=16)
    ax[1].set_title('KFAC-Laplace', fontsize=16)
    plot.reliability_diagram(sgd_predictions.cpu().numpy(), sgd_labels.numpy(), axis=ax[0])
    plot.reliability_diagram(kfac_prediction.cpu().numpy(), targets.numpy(), axis=ax[1])
    #plt.savefig(results_dir+'reliability_diagram.png')

    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    c1 = next(ax._get_lines.prop_cycler)['color']
    c2 = next(ax._get_lines.prop_cycler)['color']
    plot.calibration(sgd_predictions.cpu().numpy(), sgd_labels.numpy(), color=c1, label="SGD", axis=ax)
    plot.calibration(kfac_prediction.cpu().numpy(), targets.numpy(), color=c2, label="KFAC-Laplace", axis=ax)
    #plt.savefig(results_dir+'calibration.png')
    