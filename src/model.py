import torch
from torch import nn
from src.kfac import sample_K_laplace_MN


# linear models
class Linear_2L_KFRA(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_2L_KFRA, self).__init__()

        self.n_hid = n_hid
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(self.n_hid, output_dim)

        # choose your non-linearity
        self.act = nn.ReLU(inplace=True)
        self.one = None
        self.a2 = None
        self.h2 = None
        self.a1 = None
        self.h1 = None
        self.a0 = None

    def forward(self, x):
        self.one = torch.ones(x.shape[0], 1)
        #self.one = x.new(x.shape[0], 1).fill_(1)
        # input
        a0 = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        self.a0 = torch.cat((a0.data, self.one), dim=1)

        # first layer: h1-fc & a1-activation
        h1 = self.fc1(a0)
        self.h1 = h1.data  # torch.cat((h1, self.one), dim=1)
        a1 = self.act(h1)
        self.a1 = torch.cat((a1.data, self.one), dim=1)

        # second layer: h2-fc & a2-activation
        h2 = self.fc2(a1)
        self.h2 = h2.data  # torch.cat((h2, self.one), dim=1)
        a2 = self.act(h2)
        self.a2 = torch.cat((a2.data, self.one), dim=1)

        # third layer: h3-fc
        h3 = self.fc3(a2)
        return h3

    def sample_predict(self, x, Nsamples, Qinv1, HHinv1, MAP1, Qinv2, HHinv2, MAP2, Qinv3, HHinv3, MAP3):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        x = x.view(-1, self.input_dim)
        for i in range(Nsamples):
            # -----------------
            w1, b1 = sample_K_laplace_MN(MAP1, Qinv1, HHinv1)
            a = torch.matmul(x, torch.t(w1)) + b1.unsqueeze(0)
            a = self.act(a)
            # -----------------
            w2, b2 = sample_K_laplace_MN(MAP2, Qinv2, HHinv2)
            a = torch.matmul(a, torch.t(w2)) + b2.unsqueeze(0)
            a = self.act(a)
            # -----------------
            w3, b3 = sample_K_laplace_MN(MAP3, Qinv3, HHinv3)
            y = torch.matmul(a, torch.t(w3)) + b3.unsqueeze(0)
            predictions[i] = y

        return predictions