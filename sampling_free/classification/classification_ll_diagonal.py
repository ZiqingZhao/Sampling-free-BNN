from re import X
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
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
from models.curvatures import Diagonal, BlockDiagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot


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
    # load and normalize MNIST
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
        ('/'.join([new_mirror, url.split('/')[-1]]), md5)
        for url, md5 in datasets.MNIST.resources
    ]

    train_set = datasets.MNIST(root="../data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
    train_loader = DataLoader(train_set, batch_size=32)

    # And some for evaluating/testing
    test_set = datasets.MNIST(root="../data",
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
    diag = Diagonal(net.model)
    #kfac.load(models_dir + '/kfac.dat')
    
    for images, labels in tqdm(train_loader):
        logits = net.model(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        # A rank-1 Kronecker factored FiM approximation.
        labels = dist.sample()
        loss = criterion(logits, labels)
        net.model.zero_grad()
        loss.backward()
        diag.update(batch_size=images.size(0))
    

    # inversion of H and Q
    estimator = diag
    add = 1
    multiply = 200
    estimator.invert(add, multiply)

    
    h = []
    for i,layer in enumerate(list(estimator.model.modules())[1:]):
        if layer in estimator.state:
            H_i = estimator.inv_state[layer]
            h.append(torch.flatten(H_i))
    H = torch.cat(h, dim=0)
    
    targets = torch.Tensor()
    diag_prediction = torch.Tensor().to(device)
    diag_uncertainty = torch.Tensor().to(device)

    for images,labels in tqdm(test_loader):
        # prediction mean, equals to the MAP output 
        pred_mean = torch.nn.functional.softmax(net.model(images.to(device)) ,dim=1)        
        # compute prediction variance  
        grad_outputs = torch.zeros_like(pred_mean)
        idx  = np.argmax(pred_mean.cpu().detach().numpy(), axis=1)
        grad_outputs[:,idx] = 1
        g = []
        for p in net.model.parameters():    
            g.append(torch.flatten(gradient(pred_mean, p, grad_outputs=grad_outputs)))
        J = torch.cat(g, dim=0).unsqueeze(0) 
        pred_std = torch.abs(J * H * J).sum()
        del J
        torch.cuda.empty_cache()
        const = 2*np.e*np.pi 
        entropy = 0.5 * torch.log2(const * pred_std.unsqueeze(0))
        # ground truth
        targets = torch.cat([targets, labels])  
        # prediction, mean value of the gaussian distribution
        diag_prediction = torch.cat([diag_prediction, pred_mean]) 
        diag_uncertainty = torch.cat([diag_uncertainty, entropy]) 
        
    print(f"Diagonal Accuracy: {100 * np.mean(np.argmax(diag_prediction.cpu().detach().numpy(), axis=1) == targets.numpy()):.2f}%")
    print(f"Mean Diagonal Entropy: {diag_uncertainty.mean()}")
    # kfac entropy: -0.64

    res_uncertainty = torch.Tensor().to(device)
    for i in tqdm(range(10000)):
        noise = torch.randn_like(images)
        pred_mean = torch.nn.functional.softmax(net.model(noise.to(device)) ,dim=1)        
        # compute prediction variance  
        grad_outputs = torch.zeros_like(pred_mean)
        idx  = np.argmax(pred_mean.cpu().detach().numpy(), axis=1)
        grad_outputs[:,idx] = 1
        g = []
        for p in net.model.parameters():    
            g.append(torch.flatten(gradient(pred_mean, p, grad_outputs=grad_outputs)))
        J = torch.cat(g, dim=0).unsqueeze(0) 
        pred_std = torch.abs(J * H * J).sum()
        del J
        torch.cuda.empty_cache()
        const = 2*np.e*np.pi 
        entropy = 0.5 * torch.log2(const * pred_std.unsqueeze(0))
        res_uncertainty = torch.cat([res_uncertainty, entropy]) 
    print(f"Mean Noise Entropy: {res_uncertainty.mean()}")
    # noise entropy: 2.86