from re import X
import sys
import os

from numpy.core.function_base import add_newdoc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)


# From the repository
from models.curvatures import BlockDiagonal, Diagonal, KFAC, EFB, INF
from models.utilities import calibration_curve
from models import plot


import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
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
    
    def weight_init(self, std):
        for layer in self.modules():   
            if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                init.normal_(layer.weight, 0, std)
            # bias.data should be 0
                layer.bias.data.fill_(0)
            elif layer.__class__.__name__ == 'MultiheadAttention':
                raise NotImplementedError


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

# file path
parent = os.path.dirname(os.path.dirname(current))
data_path = parent + "/data/"
model_path = parent + "/theta/"
result_path = parent + "/results/Regression/"

torch.manual_seed(2)    # reproducible

# initialize data
std = 0.03
N = 30
sigma = 0.2
x = torch.FloatTensor(30, 1).uniform_(-4, 4).sort(dim=0).values # random x data (tensor), shape=(20, 1)
y = x.pow(3) + sigma * torch.rand(x.size()) # noisy y data (tensor), shape=(20, 1)
x, y = Variable(x,requires_grad=True), Variable(y,requires_grad=True) # torch can only train on Variable

# define the network
net = Net(input_dim=1, output_dim=1, n_hid=10)     
net.weight_init(std)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

kfac = KFAC(net)
# train the network
for t in range(10000):
    prediction = net.forward(x)     # input x and predict based on x
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients  
    kfac.update(batch_size=1)

estimator = kfac
estimator.invert(std**2, N)


x_ = torch.unsqueeze(torch.linspace(-6, 6), dim=1)  # x data (tensor), shape=(100, 1)
y_ = x_.pow(3)      
x_ = Variable(x_)
y_ = Variable(y_)

std = []
for j,x_j in enumerate(x_):
    g = []
    pred_j = net.forward(x_j)  
    std_j = 0
    for layer in list(estimator.model.modules())[1:]:
        g = []
        if layer in estimator.state:
            Q_i = estimator.inv_state[layer][0]
            H_i = estimator.inv_state[layer][1] 
            for p in layer.parameters():    
                g.append(torch.flatten(jacobian(pred_j, p)))
            J_i = torch.cat(g, dim=0).unsqueeze(0) 
            H = torch.kron(Q_i,H_i)
            std_j += torch.abs(J_i @ H @ J_i.t()).item()
    std.append(std_j**0.5 + sigma)


pred_mean = net.forward(x_).data.numpy().squeeze(1)
pred_std = np.array(std, dtype=float) 


# view data
plt.figure(figsize=(6,5))
plt.fill_between(x_.data.numpy().squeeze(1), pred_mean - pred_std, pred_mean + pred_std, color='burlywood', alpha=.6, label='+/- 1 std')
plt.fill_between(x_.data.numpy().squeeze(1), pred_mean - 2*pred_std, pred_mean + 2*pred_std, color='burlywood', alpha=.5, label='+/- 2 std')
plt.fill_between(x_.data.numpy().squeeze(1), pred_mean - 3*pred_std, pred_mean + 3*pred_std, color='burlywood', alpha=.4, label='+/- 3 std')
plt.plot(x_.data.numpy(), y_.data.numpy(), c='black', label='ground truth', linewidth = 2)
plt.plot(x_.data.numpy(), pred_mean, c='cornflowerblue', label='mean pred', linewidth = 2)
plt.scatter(x.data.numpy(), y.data.numpy(), s=20, color = "black")
plt.title('Uncertainty with block diagonal Hessian', fontsize=20)
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$y$', fontsize=15)
plt.legend()
plt.xlim([-6, 6])
plt.ylim([-800, 800])
plt.gca().yaxis.grid(alpha=0.3)
plt.gca().xaxis.grid(alpha=0.3)
plt.tick_params(labelsize=10)
plt.savefig(result_path+'kfac.png', format='png', bbox_inches = 'tight')
#plt.savefig(result_path+'diagonal.eps', format='eps', bbox_inches = 'tight')

