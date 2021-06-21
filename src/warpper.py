import numpy as np
import torch
import torch.nn.functional as F

from src.model import Linear_2L_KFRA
from src.utilities import to_variable
from src.kfac import softmax_CE_preact_hessian
from src.kfac import layer_act_hessian_recurse
from src.kfac import sample_K_laplace_MN


class BaseNet(object):
    def __init__(self):
        print('Net:')

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
    def save(self, filename):
        print('Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        print('Reading %s\n' % filename)
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch


class KBayes_Net(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=False, classes=10, n_hid=1200, batch_size=128, prior_sig=0):
        super(KBayes_Net, self).__init__()
        print('Creating Net!!')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.n_hid=n_hid
        self.channels_in = channels_in
        self.prior_sig = prior_sig
        self.classes = classes
        self.batch_size = batch_size
        self.side_in=side_in
        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = Linear_2L_KFRA(input_dim=self.channels_in * self.side_in * self.side_in, output_dim=self.classes,
                                    n_hid=self.n_hid)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5,
                                         weight_decay=1 / self.prior_sig ** 2)

    def fit(self, x, y):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='sum')

        loss.backward()
        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
        out = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')
        probs = F.softmax(out, dim=1).data.cpu()
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def get_K_laplace_params(self, trainloader):
        self.model.eval()

        it_counter = 0
        cum_HH1 = self.model.fc1.weight.data.new(self.model.n_hid, self.model.n_hid).fill_(0)
        cum_HH2 = self.model.fc1.weight.data.new(self.model.n_hid, self.model.n_hid).fill_(0)
        cum_HH3 = self.model.fc1.weight.data.new(self.model.output_dim, self.model.output_dim).fill_(0)

        cum_Q1 = self.model.fc1.weight.data.new(self.model.input_dim + 1, self.model.input_dim + 1).fill_(0)
        cum_Q2 = self.model.fc1.weight.data.new(self.model.n_hid + 1, self.model.n_hid + 1).fill_(0)
        cum_Q3 = self.model.fc1.weight.data.new(self.model.n_hid + 1, self.model.n_hid + 1).fill_(0)

        # Forward pass

        for x, y in trainloader:
            x, y = to_variable(var=(x, y.long()), cuda=False)
            self.optimizer.zero_grad()
            out = self.model(x)
            out_act = F.softmax(out, dim=1)
            loss = F.cross_entropy(out, y, reduction='sum')

            loss.backward()

            #     ------------------------------------------------------------------
            HH3 = softmax_CE_preact_hessian(out_act.data)
            cum_HH3 += HH3.sum(dim=0)
            #     print(model.a2.data.shape)
            Q3 = torch.bmm(self.model.a2.data.unsqueeze(2), self.model.a2.data.unsqueeze(1))
            cum_Q3 += Q3.sum(dim=0)
            #     ------------------------------------------------------------------
            HH2 = layer_act_hessian_recurse(prev_hessian=HH3, prev_weights=self.model.fc3.weight.data,
                                            layer_pre_acts=self.model.h2.data)
            cum_HH2 += HH2.sum(dim=0)
            Q2 = torch.bmm(self.model.a1.data.unsqueeze(2), self.model.a1.data.unsqueeze(1))
            cum_Q2 += Q2.sum(dim=0)
            #     ------------------------------------------------------------------
            HH1 = layer_act_hessian_recurse(prev_hessian=HH2, prev_weights=self.model.fc2.weight.data,
                                            layer_pre_acts=self.model.h1.data)
            cum_HH1 += HH1.sum(dim=0)
            Q1 = torch.bmm(self.model.a0.data.unsqueeze(2), self.model.a0.data.unsqueeze(1))
            cum_Q1 += Q1.sum(dim=0)
            #     ------------------------------------------------------------------
            it_counter += x.shape[0]
            print(it_counter)

        EHH3 = cum_HH3 / it_counter
        EHH2 = cum_HH2 / it_counter
        EHH1 = cum_HH1 / it_counter

        EQ3 = cum_Q3 / it_counter
        EQ2 = cum_Q2 / it_counter
        EQ1 = cum_Q1 / it_counter

        MAP3 = torch.cat((self.model.fc3.weight.data, self.model.fc3.bias.data.unsqueeze(1)), dim=1)
        MAP2 = torch.cat((self.model.fc2.weight.data, self.model.fc2.bias.data.unsqueeze(1)), dim=1)
        MAP1 = torch.cat((self.model.fc1.weight.data, self.model.fc1.bias.data.unsqueeze(1)), dim=1)

        return EQ1, EHH1, MAP1, EQ2, EHH2, MAP2, EQ3, EHH3, MAP3

    def sample_eval(self, x, y, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2, MAP2,
                    scale_inv_EQ3, scale_inv_EHH3, MAP3, logits=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2,
                                        MAP2, scale_inv_EQ3, scale_inv_EHH3, MAP3)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2, MAP2,
                        scale_inv_EQ3, scale_inv_EHH3, MAP3):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2,
                                        MAP2, scale_inv_EQ3, scale_inv_EHH3, MAP3)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

    def get_weight_samples(self, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2, MAP2,
                           scale_inv_EQ3, scale_inv_EHH3, MAP3):
        weight_vec = []

        for i in range(Nsamples):

            w1, b1 = sample_K_laplace_MN(MAP1, scale_inv_EQ1, scale_inv_EHH1)
            w2, b2 = sample_K_laplace_MN(MAP2, scale_inv_EQ2, scale_inv_EHH2)
            w3, b3 = sample_K_laplace_MN(MAP3, scale_inv_EQ3, scale_inv_EHH3)

            for weight in w1.cpu().numpy().flatten():
                weight_vec.append(weight)
            for weight in w2.cpu().numpy().flatten():
                weight_vec.append(weight)
            for weight in w3.cpu().numpy().flatten():
                weight_vec.append(weight)

        return np.array(weight_vec)