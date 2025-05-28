#from imports import *
import helper_functions as hf
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
import re

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.utils.data as utils
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else print ('cpu')

class InfiniteLoader(utils.DataLoader):
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, num_epochs=None, *args, **kwargs):
        """Constructor.

        Args:
            dataset: A `Dataset` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """
        super().__init__(
            *args, **kwargs
        )
        self.finite_iterable = super().__iter__()
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = super().__iter__()
            return next(self.finite_iterable)

    def __iter__(self):
        return self

    def __len__(self):
        return None

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers,act=None,out_act=None,dropout=0.0):
        super().__init__()
        layers = []
        dcurr = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(dcurr, hidden_dim))
            layers.append(nn.ReLU() if act is None else act)
            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            dcurr = hidden_dim
        layers.append(nn.Linear(dcurr, output_dim))
        layers.append(nn.Tanh() if out_act is None else out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

loss_dict = {
    "MSE":nn.MSELoss(),
    "BCE":nn.BCELoss()
}
    
def train_nn(net,inputs,labels,val_inputs,val_labels,n_epoch=20,lr=1e-3,bs=256,lossfunc="BCE"):
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=lr)
    dataset = utils.TensorDataset(torch.tensor(inputs,dtype=torch.float32,device=device),torch.tensor(labels,dtype=torch.float32,device=device))
    val_dataset = utils.TensorDataset(torch.tensor(val_inputs,dtype=torch.float32,device=device),torch.tensor(val_labels,dtype=torch.float32,device=device))
    loader = InfiniteLoader(dataset=dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
    val_loader = utils.DataLoader(dataset=val_dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
    n_iter = n_epoch*inputs.shape[0]//bs
    n_iter_val = inputs.shape[0]//(bs)
    loss_fn = loss_dict[lossfunc]
    losses = []
    val_losses = []
    tbar = tqdm(range(n_iter))
    best_state = None
    min_loss = 1e+8
    patience_count = 0
    for i in tbar:
        x,y = next(loader)
        optimizer.zero_grad()
        loss = loss_fn(net(x),y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            patience_count=0
            best_state = net.state_dict()
        else:
            patience_count+=1
        s = 'Loss: {0}, p = {1}'.format(loss.item(),patience_count)
        tbar.set_description(s)
        if (i+1)%n_iter_val == 0:
            vlosses = []
            with torch.no_grad():
                for i, (x,y) in enumerate(val_loader):
                    vlosses.append(loss_fn(net(x),y).item())
            val_losses.append(np.mean(vlosses))

    net.load_state_dict(best_state)
    net.eval()
    torch.cuda.empty_cache()
    net = net.to('cpu')
    
    plt.figure(figsize=(8,6))
    #w = int(inputs.shape[0]/(bs))
    #smooth = np.convolve(np.ones(w),losses,mode='valid')/w
    #xvals = np.linspace(0,len(losses),len(smooth))
    #plt.plot(xvals,smooth)
    plt.plot(np.arange(len(losses)),losses)
    xvals = np.linspace(0,len(losses),len(val_losses))
    plt.plot(xvals,val_losses)
    
    return net

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        assert type(hidden_dims) == list
        self.latent_dim = latent_dim
        widths = [input_dim] + hidden_dims
        layers = []
        for i in range(len(widths)-1):
            layers.append(nn.Linear(widths[i],widths[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(widths[-1],2*latent_dim))
        self.layers = nn.ModuleList(layers)
        self.encoder = nn.Sequential(*self.layers)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        z = self.encoder(x)
        mu =  z[:,:self.latent_dim]
        sigma = torch.exp(z[:,self.latent_dim:])
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        assert type(hidden_dims) == list
        widths = [latent_dim] + hidden_dims
        layers = []
        for i in range(len(widths)-1):
            layers.append(nn.Linear(widths[i],widths[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(widths[-1],output_dim))
        self.layers = nn.ModuleList(layers)
        self.decoder = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.decoder(x)
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = VariationalEncoder(input_dim,hidden_dims,latent_dim)
        self.decoder = Decoder(latent_dim,hidden_dims[::-1],input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

mse = nn.MSELoss()
def trainVAE(autoencoder, loader, epochs=20,lr=5e-4):
    opt = torch.optim.Adam(autoencoder.parameters(),lr=lr)
    tbar = tqdm(range(epochs))
    losses = []
    best_dict = None
    min_loss = 99999
    for epoch in tbar:
        x, = next(loader)
        opt.zero_grad()
        x_hat = autoencoder(x)
        loss = mse(x,x_hat) + autoencoder.encoder.kl
        loss.backward()
        opt.step()
        losses.append(loss.item())
        s = "Loss: {0}".format(loss.item())
        tbar.set_description(s)
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_dict = autoencoder.state_dict()
    autoencoder.load_state_dict(best_dict)
    return autoencoder, losses