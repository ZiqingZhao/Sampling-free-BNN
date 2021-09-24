from re import X
import sys
import os

from torch._C import _has_torch_function_variadic
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
from models.utilities import *
from models.plot import *
from models.wrapper import *
   
def tensor_to_image(tensor):
    min = tensor.min().item()
    max = tensor.max().item()
    norm = (tensor - min) / (max-min)
    image = Image.fromarray(np.uint8(255*torch.sqrt(norm).numpy())).convert('RGB')
    return image

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
test_loader = DataLoader(test_set, batch_size=256)

N = 200
std = 0.1

# Train the model
net = BaseNet_750()
net.weight_init(std)
if device == 'cuda': 
    net.to(torch.device('cuda'))
get_nb_parameters(net)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# train(net, device, train_loader, criterion, optimizer, epochs=10)
# save(net, model_path + 'BaseNet_750.dat')
load(net, model_path + 'BaseNet_750.dat')

# run on the testset
sgd_predictions, sgd_labels = eval(net, device, test_loader)
acc = accuracy(sgd_predictions, sgd_labels)

# update likelihood FIM
H = None
for images, labels in tqdm(train_loader):
    logits = net(images.to(device))
    dist = torch.distributions.Categorical(logits=logits)
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
            
H = H.cpu()/len(train_loader) 

diag = torch.diag(std * torch.ones(H.shape[0]))
H_inv = torch.linalg.pinv(N * H + diag)

H_diag = torch.diag(H)
H_inv_diag = torch.diag(torch.reciprocal(N * H_diag + std * torch.ones(H.shape[0])))


image_inv = tensor_to_image(H_inv.abs())
image_inv.save(result_path+'750/H_inv_750_dense.png')

image_inv_diag = tensor_to_image(H_inv_diag.abs())
image_inv_diag.save(result_path+'750/H_inv_750_diag.png')

image_error = tensor_to_image(torch.abs(H_inv-H_inv_diag))
image_error.save(result_path+'750/error_750.png')
