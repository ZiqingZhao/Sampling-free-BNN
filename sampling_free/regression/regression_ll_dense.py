from re import X
import sys
import os

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


# From the repository
from models.wrapper import BaseNet
from models.curvatures import BlockDiagonal, Diagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
import numpy as np
import imageio


# define a network
class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, n_hid)   # hidden layer
        self.fc2 = torch.nn.Linear(n_hid, n_hid)   # hidden layer
        self.fc3 = torch.nn.Linear(n_hid, output_dim)   # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # activation function for hidden layer
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)  # linear output
        return x

# model with one specific layer
class CurrentLayer(torch.nn.Module):
    # Wrap any model to get the response of an intermediate layer
    def __init__(self, model, layer=None):
        """
        model: PyTorch model
        layer: int, which model response layers to output
        """
        super().__init__()
        features = list(model.modules())[1:]
        self.features = torch.nn.ModuleList(features).eval()

        if layer is None:
            layer = len(self.features)
        self.layer = layer

    def forward(self, x):
        # Propagates input through each layer of model until self.layer, at which point it returns that layer's output
        for ii, mdl in enumerate(self.features):
            x = mdl(x)
            if ii == self.layer:
                return x


# backward Jacobian: derivative of outputs with respect to weights
def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return grad

def jacobian(y, x):
    '''
    Compute dy/dx = dy/dx @ grad_outputs; 
    y: output, batch_size * class_number
    x: parameter
    '''
    jac = torch.zeros(y.shape[0], torch.flatten(x).shape[0])
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i,:] = torch.flatten(gradient(y, x, grad_outputs))
    return jac

def hessian(y,x):
    grads = torch.autograd.grad(y, x, retain_graph=True, create_graph=True)
    print( grads[0] )
    hess = torch.zeros_like(grads[0])
    for i in range(grads[0].size(0)):
        for j in range(grads[0].size(1)):
            hess[i, j] = torch.autograd.grad(grads[0][i][j], x, retain_graph=True)[0][i, j]
    return hess

torch.manual_seed(2)    # reproducible

sigma = 0.2
x = torch.FloatTensor(30, 1).uniform_(-4, 4).sort(dim=0).values # random x data (tensor), shape=(20, 1)
y = x.pow(3) + sigma * torch.rand(x.size()) # noisy y data (tensor), shape=(20, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x,requires_grad=True), Variable(y,requires_grad=True)

net = Net(input_dim=1, output_dim=1, n_hid=10)     # define the network
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# train the network
for t in range(10000):
    prediction = net.forward(x)     # input x and predict based on x
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients  
    
# get inversion of H
add = 1
multiply = 200
grads = []
for layer in list(net.model.modules())[1:]:
    for p in layer.parameters():    
        grads.append(torch.flatten(p.grad.view(-1)))
J_loss = torch.cat(grads, dim=0).unsqueeze(0) 
H = J_loss.t() @ J_loss
diag = torch.diag(H.new(H.shape[0]).fill_(add ** 0.5))
reg = multiply ** 0.5 * H + diag
H_inv = torch.inverse(reg)
sum_diag = torch.diag(H_inv).abs().sum()
sum_non_diag = torch.abs(H_inv-torch.diag(H_inv)).sum()
print(f"sum of diagonal: {sum_diag:.2f}")
print(f"sum of non-diagonal: {sum_non_diag:.2f}")

x_ = torch.unsqueeze(torch.linspace(-6, 6), dim=1)  # x data (tensor), shape=(100, 1)
y_ = x_.pow(3)      
x_ = Variable(x_)
y_ = Variable(y_)

std = []
for j,x_j in enumerate(x_):
    pred_j = net.forward(x_j)  
    std_j = 0
    g = [] 
    for layer in list(net.modules())[1:]:
        for p in layer.parameters():    
            g.append(torch.flatten(jacobian(pred_j, p)))
    J = torch.cat(g, dim=0).unsqueeze(0) 
