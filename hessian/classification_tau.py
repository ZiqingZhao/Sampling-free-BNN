from re import X
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Standard imports
import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image, ImageOps  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# From the repository
from models.curvatures import BlockDiagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models.plot import *
from models.wrapper import *


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

   
# file path
parent = os.path.dirname(current)
data_path = parent + "/data/"
model_path = parent + "/theta/"
result_path = parent + "/results/Hessian/"

# choose the device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed(42) 

# load and normalize MNIST
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
]

train_set = datasets.MNIST(root=data_path,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
train_loader = DataLoader(train_set, batch_size=32)

# And some for evaluating/testing
test_set = datasets.MNIST(root=data_path,
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)
test_loader = DataLoader(test_set, batch_size=1)

N = 200
std = [i/20 for i in range(1,11)]
H_eig = []
H_inv_eig = []
acc = []
for i in range(10):
    print(f"std: {std[i]}")
    # Train the model
    net = BaseNet_750()
    net.weight_init(std[i])
    if device == 'cuda': 
        net.to(torch.device('cuda'))
    get_nb_parameters(net)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, device, train_loader, criterion, optimizer, epochs=10)
    # save(net, model_path + 'BaseNet_15k.dat')
    # load(net, model_path + 'BaseNet_750.dat')

    # run on the testset
    sgd_predictions, sgd_labels = eval(net, device, test_loader)
    acc.append(accuracy(sgd_predictions, sgd_labels))

'''

    # update likelihood FIM
    H = None
    for images, labels in tqdm(test_loader):
        logits = net(images.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        # A rank-1 Kronecker factored FiM approximation.
        labels = dist.sample()
        loss = criterion(logits, labels)
        net.zero_grad()
        loss.backward()
                
        grads = []
        for layer in list(net.modules())[1:]:
            for p in layer.parameters():    
                J_p = torch.flatten(p.grad.view(-1)).unsqueeze(0)
                grads.append(J_p)
        J_loss = torch.cat(grads, dim=1)
        H_loss = J_loss.t() @ J_loss
        H_loss.requires_grad = False
        H = H_loss if H == None else H + H_loss

    H = H/len(test_loader) 

    #calculate the pseudo inverse of H
    diag = torch.diag(H.new(H.shape[0]).fill_(std[i]))
    H_inv = torch.linalg.pinv(N * H + diag).cpu()
    
    # calculate the eigenvalues
    eig = torch.linalg.eigvals(H).real
    eig_inv = torch.linalg.eigvals(H_inv).real
    print(f"Maximum Eigenvalue of H: {eig.max().item()}")
    print(f"Maximum Eigenvalue of H_inv: {eig_inv.max().item()}")
    H_eig.append(eig)
    H_inv_eig.append(eig_inv)

H_eig_min = []
H_inv_eig_max = []
for i in range(10):
    H_eig_min.append(H_eig[i].min().item())
    H_inv_eig_max.append(H_inv_eig[i].max().item())
'''

# plot the results
plt.figure(figsize=(5,5))
#plt.plot(np.array(std)**2, np.array(H_inv_eig_max), label='Inverse of Hessian')   
plt.plot(np.array(std), np.array(acc))  
plt.title('Accuracy-std')
plt.xlabel('std')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

