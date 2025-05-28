from argparse import ArgumentParser
import re
import os
import json
import time
import multiprocessing
import sys
import math
import time
import tqdm
import numpy as np
import scipy as sp
from scipy import stats
from scipy.spatial import ConvexHull
import itertools
import logging
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from sklearn import metrics
from pathlib import Path
from samples import sample_library_UL17, sample_library_UL18

import helper_functions as hf

#sys.path.append("/home/sambt/nflows/")
#sys.path.append("nflows/nflows/")
sys.path.append("/home/sambt/CASE_final/CASE_QUAK_final/")
from nflows.flows.base import Flow
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseQuadraticAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.utils.data as utils

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

class NFTrainer:
    def __init__(self,sample,mean_reference,variables,nTrain=100000,NF_kwargs={},dEta_cut=-1,sideband=False,modelName=None,rangeScale=3,minmax=False,outBase="combined_trainings"):
        self.sample = sample
        self.mean_reference = mean_reference
        self.variables = variables
        self.NF_kwargs = NF_kwargs
        self.nTrain = nTrain
        self.dEta_cut = dEta_cut
        self.sideband = sideband
        self.rangeScale = rangeScale
        self.minmax = minmax
        
        if type(self.sample) == list:
            if modelName:
                self.modelName = modelName
            else:
                self.modelName = "-and-".join(self.sample)
            self.outDir = f"{outBase}/{self.modelName}/"
            self.modelFile = self.outDir+self.modelName+".pt"
        else:
            if modelName:
                self.modelName = modelName
            else:
                self.modelName = self.sample
            self.outDir = f"{outBase}/{self.modelName}/"
            self.modelFile = self.outDir+self.modelName+".pt"
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        
        self.NF_kwargs['num_features'] = len(self.variables)
        self.NF_kwargs['variables'] = self.variables
        self.NF_kwargs['minmax'] = self.minmax
        self.NF_kwargs['rangeScale'] = self.rangeScale
    
    def loadMeans(self):
        return loadMeans(self.mean_reference,self.variables)
    
    def loadMinMax(self):
        return loadMinMax(self.variables)
    
    def loadData(self,nevts,mode="train"):
        return loadData(self.sample,self.variables,nevts,dEta_cut=self.dEta_cut,sideband=self.sideband,
                        mode=mode,mean_reference=self.mean_reference,minmax=self.minmax,rangeScale=self.rangeScale)
        
    def new_flow(self,num_features):
        return make_flow(num_features,self.NF_kwargs)
    
    def get_model(self):
        if not os.path.isfile(self.modelFile):
            print("No model file found! Need to train first")
            return
        flow = self.new_flow(len(self.variables))
        flow.load_state_dict(torch.load(self.modelFile))
        flow.eval()
        return flow
    
    def train_v2(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10,T_max=-1,eta_min=-1,scheduler=None,clip=None):
        trainData = self.loadData(self.nTrain,mode="train")
        print(f"{len(trainData)} training events")
        if clip is not None:
            trainData = trainData[np.all(np.abs(trainData)<clip,axis=1)]
        if bs < 1:
            bs = int(bs*trainData.shape[0])
        trainData = torch.tensor(trainData,device=device)
        loader = InfiniteLoader(dataset=trainData,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
        ntrain = trainData.shape[0]
        
        with open(self.outDir+"flow_kwargs.json","w") as f:
            json.dump(self.NF_kwargs,f,indent=4)
        
        flow = self.new_flow(len(self.variables))
        flow = flow.to(device)
        optimizer = optim.Adam(flow.parameters(),lr=learning_rate)
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        else:
            T_max = T_max if T_max!=-1 else n_epoch//2
            eta_min = eta_min if eta_min!=-1 else 0.01*learning_rate
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max,eta_min=eta_min,verbose=False)

        min_loss = 1e+8
        train_losses = []
        patience_count = 0
        saveName = self.modelFile
        
        print("Training flow {0}".format(self.modelName))
        tbar = tqdm(range(int(n_epoch)))
        for i in tbar:
            if patience_count == patience:
                break
            x = next(loader)
            optimizer.zero_grad()
            loss = -flow.log_prob(inputs=x)[0].mean()
            loss.backward()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(flow.state_dict(),saveName)
            train_losses.append(loss.item())
            if loss.item() == min_loss:
                patience_count = 0
            else:
                patience_count += 1
            s = 'Loss: {0}, p = {1}'.format(loss.item(),patience_count)
            tbar.set_description(s)
            scheduler.step()
            
        plt.figure(figsize=(8,6))
        w = int(ntrain/(5*bs))
        smooth = np.convolve(np.ones(w),train_losses,mode='valid')/w
        xvals = np.linspace(0,len(train_losses),len(smooth))
        plt.plot(xvals,smooth)
        #plt.plot(np.arange(len(train_losses)),train_losses)
        plt.title(self.modelName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(self.outDir+"trainCurve.pdf")
        
        del flow, loader, trainData, optimizer, scheduler
        torch.cuda.empty_cache()
    
    def train(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10,T_max=-1,eta_min=-1,scheduler=None,clip=None):
        trainData = self.loadData(self.nTrain,mode="train")
        print(f"{len(trainData)} training events")
        if clip is not None:
            trainData = trainData[np.all(np.abs(trainData)<clip,axis=1)]
        if bs < 1:
            bs = int(bs*trainData.shape[0])
        trainData = torch.tensor(trainData,device=device)
        loader = utils.DataLoader(trainData,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
        
        with open(self.outDir+"flow_kwargs.json","w") as f:
            json.dump(self.NF_kwargs,f,indent=4)
        
        flow = self.new_flow(len(self.variables))
        flow = flow.to(device)
        optimizer = optim.Adam(flow.parameters(),lr=learning_rate)
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        else:
            T_max = T_max if T_max!=-1 else n_epoch//2
            eta_min = eta_min if eta_min!=-1 else 0.01*learning_rate
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max,eta_min=eta_min,verbose=False)

        min_loss = 1e+8
        train_losses = []
        patience_count = 0
        saveName = self.modelFile
        
        print("Training flow {0}".format(self.modelName))
        tbar = tqdm(range(int(n_epoch)))
        for i in tbar:
            if patience_count == patience:
                break
            epoch_losses = []
            for batch_idx, x in enumerate(loader):
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=x)[0].mean()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    torch.save(flow.state_dict(),saveName)
            epoch_loss = np.min(epoch_losses)
            train_losses.append(epoch_loss)
            #if (i + 1) % print_interval == 0:
                #print('Iteration {} Complete'.format(i + 1))
                #print('Loss: ', epoch_loss)
            if epoch_loss == min_loss:
                patience_count = 0
            else:
                patience_count += 1
            s = 'Loss: {0}, p = {1}'.format(np.average(epoch_losses),patience_count)
            tbar.set_description(s)
            scheduler.step()
            
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(len(train_losses)),train_losses)
        plt.title(self.modelName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(self.outDir+"trainCurve.pdf")
        
        del flow, loader, trainData, optimizer, scheduler
        torch.cuda.empty_cache()
        
    def plotDensity(self,nSamp,log=False):
        data = self.loadData(nSamp,mode="test")
        flow = self.get_model()
        flow = flow.to(device)
        with torch.no_grad():
            samples = flow.sample(data.shape[0]).detach().cpu().numpy()
        plt.figure(figsize=(18,18))
        plt.rcParams['font.size'] = 10
        plt.rcParams['legend.fontsize'] = 8
        plt.rcParams['axes.labelsize'] = 8
        plt.suptitle("Training samples vs truth",fontsize=16,y=0.93)
        for index in range(len(self.variables)):
            plt.subplot(4,4,index+1)
            n, bins, patches = plt.hist(data[:, index], bins=50, histtype='step', label='Truth')
            plt.hist(samples[:, index], bins=bins, histtype='step', label='NF Density')
            if index == 0:
                plt.legend(loc='best')
            plt.title(self.variables[index])
            if log:
                plt.yscale('log')
        if log:
            plt.savefig(self.outDir+"density_estimation_log.pdf")
        else:
            plt.savefig(self.outDir+"density_estimation.pdf")
        plt.show()
        del data, samples, flow
        torch.cuda.empty_cache()
        
def make_flow(num_features,kwargs,perm=True):
    flow_type = kwargs['flow_type']
    base_dist = StandardNormal(shape=[num_features])
    transforms = []
    for i in range(kwargs['num_layers']):
        if flow_type == 'MAF':
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,
                                                                    hidden_features=kwargs['hidden_features'], 
                                                                    num_blocks=kwargs['num_blocks_per_layer']))
        elif flow_type == 'NSQUAD':
            transforms.append(MaskedPiecewiseQuadraticAutoregressiveTransform(features=num_features,
                                                                            hidden_features=num_features,
                                                                            num_bins=kwargs['num_bins'],
                                                                            num_blocks=kwargs['num_blocks_per_layer'],
                                                                            tail_bound=kwargs['tail_bound'], 
                                                                            tails='linear'))
        elif flow_type == 'NSRATQUAD':
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                hidden_features=kwargs['hidden_features'],
                                                                                num_bins=kwargs['num_bins'],
                                                                                num_blocks=kwargs['num_blocks_per_layer'],
                                                                                tail_bound=kwargs['tail_bound'], 
                                                                                tails=kwargs['tails']))
        if i < kwargs['num_layers'] - 1 and perm:
            transforms.append(ReversePermutation(features=num_features))

    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow


def loadMeans(mean_reference,variables):
    means = []
    stds = []    
    ref_file = "meanStd.h5" # no dEta cut!
    with h5py.File(f"/nobackup/users/sambt/CASE_training/slimmed_h5s/{mean_reference}/{ref_file}","r") as f:
        for v in variables:
            means.append(f[v+"_mean"][()])
            stds.append(f[v+"_std"][()])
    means = np.array(means)
    stds = np.array(stds)
    return means, stds

def loadMinMax(variables):
    mins = []
    maxes = []
    with open("/nobackup/users/sambt/CASE_training/slimmed_h5s/trainVarMinMaxes.json","r") as f:
        js = json.load(f)
    for v in variables:
        mins.append(js['mins'][v])
        maxes.append(js['maxes'][v])
    mins = np.array(mins,dtype='float32')
    maxes = np.array(maxes,dtype='float32')
    return mins, maxes

def loadData(sample,variables,nevts,dEta_cut=-1,sideband=False,mode="train",mean_reference="BKG",minmax=False,rangeScale=3):
    print(f"Loading {mode} sample")
    if type(sample) == list:
        datas = []
        nev_per = int(nevts/len(sample))
        for samp in sample:
            datas.append(hf.loadData(samp,variables,nMax=nev_per,mode=mode,
                                        dEta_cut=dEta_cut,sideband=sideband))
            print(f"loading {samp}, {len(datas[-1])} events")
        data = np.concatenate(datas,axis=0)
        np.random.shuffle(data)
    else:
        data = hf.loadData(sample,variables,nMax=nevts,mode=mode,dEta_cut=dEta_cut,sideband=sideband)
    means, stds = loadMeans(mean_reference,variables)
    mins, maxes = loadMinMax(variables)
    if minmax:
        data = data-means
        scale = np.where(np.abs(maxes-means)>np.abs(mins-means),np.abs(maxes-means),np.abs(mins-means))
        data = rangeScale*(data/scale)
    else:
        data = (data-means)/stds
    return data

def train(flow,trainData,NF_kwargs,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10,T_max=-1,eta_min=-1,scheduler=None,clip=None,outName="flow",outDir="flow_on_losses"):
    print(f"{len(trainData)} training events")
    if clip is not None:
        trainData = trainData[np.all(np.abs(trainData)<clip,axis=1)]
    if bs < 1:
        bs = int(bs*trainData.shape[0])
    trainData = torch.tensor(trainData,device=device)
    loader = utils.DataLoader(trainData,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))

    
    with open(f"{outDir}/{outName}.json","w") as f:
        json.dump(NF_kwargs,f,indent=4)

    flow = flow.to(device)
    optimizer = optim.Adam(flow.parameters(),lr=learning_rate)
    if scheduler is not None:
        scheduler = scheduler(optimizer)
    else:
        T_max = T_max if T_max!=-1 else n_epoch//2
        eta_min = eta_min if eta_min!=-1 else 0.01*learning_rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max,eta_min=eta_min,verbose=False)

    min_loss = 1e+8
    train_losses = []
    patience_count = 0
    saveName = f"{outDir}/{outName}.pt"
    for i in range(n_epoch):
        if patience_count == patience:
            break
        epoch_losses = []
        for batch_idx, x in enumerate(loader):
            optimizer.zero_grad()
            loss = -flow.log_prob(inputs=x)[0].mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(flow.state_dict(),saveName)
        epoch_loss = np.min(epoch_losses)
        train_losses.append(epoch_loss)
        if (i + 1) % print_interval == 0:
            print('Iteration {} Complete'.format(i + 1))
            print('Loss: ', epoch_loss)
        if epoch_loss == min_loss:
            patience_count = 0
        else:
            patience_count += 1
        scheduler.step()

    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(train_losses)),train_losses)
    plt.xlabel('Epoch',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{outDir}/{outName}-train.pdf")

    del loader, trainData, optimizer, scheduler
    torch.cuda.empty_cache()
    flow.load_state_dict(torch.load(f'flow_on_losses/{outName}.pt'))
    return flow

