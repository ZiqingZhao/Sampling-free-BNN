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
std = [i/30 for i in range(1,11)]
H_eig = []
H_inv_eig = []
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
    train(net, device, train_loader, criterion, optimizer, epochs=3)
    # save(net, model_path + 'BaseNet_15k.dat')
    # load(net, model_path + 'BaseNet_750.dat')

    # run on the testset
    sgd_predictions, sgd_labels = eval(net, device, test_loader)
    accuracy(sgd_predictions, sgd_labels)

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
    H_inv = N * (torch.linalg.pinv(H) + diag)
    
    # calculate the eigenvalues
    eig = torch.linalg.eigvals(H).real.max().item()
    eig_inv = torch.linalg.eigvals(H_inv).real.max().item()
    print(f"Maximum Eigenvalue of H: {eig}")
    print(f"Maximum Eigenvalue of H_inv: {eig_inv}")
    H_eig.append(eig)
    H_inv_eig.append(eig_inv)

# plot the results
plt.figure(figsize=(10,10))
plt.plot(np.array(std)**2, np.array(H_eig), label='Hessian')   
plt.plot(np.array(std)**2, np.array(H_inv_eig), label='Inverse of Hessian')   
plt.title('Maximum Eigenvalue of Hessian Matrix')
plt.xlabel('tau')
plt.ylabel('eigenvalue')
plt.legend()
plt.tight_layout()
plt.show()

"""
# inversion of H
add = 1
multiply = 200
diag = torch.diag(H.new(H.shape[0]).fill_(add ** 0.5))
reg = multiply ** 0.5 * H + diag
H_inv = torch.inverse(reg).cpu()
0
3,


sum_diag = torch.diag(H_inv).abs().sum().item()
sum_non_diag = torch.abs(H_inv-torch.diag(torch.diag(H_inv))).sum()
print(f"sum of diagonal: {sum_diag:.2f}")
print(f"sum of non-diagonal: {sum_non_diag:.2f}")

min = H_inv.abs().min().item()
max = H_inv.abs().max().item()
H_norm = (H_inv.abs() - min) / (max-min)

PIL_image = Image.fromarray(np.uint8(255*torch.sqrt(H_norm[:3000,:3000]).numpy())).convert('RGB')
PIL_image.save(result_path+'15k/H_inv_15k_sqrt.png')


'''
748
H_inv
sum of diagonal: 605.26
sum of non-diagonal: 1609.85

15080
H_inv 
sum of diagonal: 14879.16
sum of non-diagonal: 63296.37

60
'''
"""