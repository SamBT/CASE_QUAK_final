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

sys.path.append("/home/sambt/CASE_final/CASE_QUAK_final/")
from nflows.flows.base import Flow
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal, FullNormal
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

class NFTrainer:
    def __init__(self,sample,mean_reference,variables,nTrain=100000,NF_kwargs={},dEta_cut=1.3):
        self.sample = sample
        self.mean_reference = mean_reference
        self.variables = variables
        self.NF_kwargs = NF_kwargs
        self.nTrain = nTrain
        self.dEta_cut = dEta_cut
        
        if type(self.sample) == list:
            self.modelName = "-and-".join(self.sample)
            self.outDir = f"combined_trainings/{self.modelName}/"
            self.modelFile = self.outDir+self.modelName+".pt"
        else:
            self.modelName = self.sample
            self.outDir = f"combined_trainings/{self.modelName}/"
            self.modelFile = self.outDir+self.modelName+".pt"
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        
        self.NF_kwargs['num_features'] = len(self.variables)
        self.NF_kwargs['variables'] = self.variables
    
    def loadMeans(self):
        means = []
        stds = []
        if self.dEta_cut > 0:
            ref_file = f"meanStd_dEta{self.dEta_cut:.1f}.h5"
        else:
            ref_file = "meanStd.h5"
        with h5py.File(f"input_h5s/{self.mean_reference}/{ref_file}","r") as f:
            for v in self.variables:
                means.append(f[v+"_mean"][()])
                stds.append(f[v+"_std"][()])
        means = np.array(means)
        stds = np.array(stds)
        return means, stds
    
    def loadData(self,nevts,mode="train"):
        print(f"Loading {mode} sample")
        data = hf.loadData(self.sample,self.variables,nMax=nevts,mode=mode,dEta_cut=self.dEta_cut)
        means, stds = self.loadMeans()
        return (data-means)/stds
        
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
        
def make_flow(num_features,kwargs):
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
        if i < kwargs['num_layers'] - 1:
            transforms.append(ReversePermutation(features=num_features))

    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow


def get_model(modelFile,kwargs):
    if not os.path.isfile(modelFile):
        print("No model file found! Need to train first")
        return
    if not os.path.isfile(kwargs):
        print("No kwargs file found! Need to train first")
        return
    with open(kwargs) as f:
        NF_kwargs = json.load(f)
    if NF_kwargs['mean'] is not None:
        mean = torch.tensor(NF_kwargs['mean'])
        cov = torch.tensor(NF_kwargs['cov'])
    else:
        mean=None
        cov=None
    flow = make_flow(len(self.variables),mean,cov)
    flow.load_state_dict(torch.load(self.modelFile))
    flow.eval()
    return flow

def loadData(sample,year,variables,nTrain):
    smallYear = year % 2000
    if sample == "QCDBKG":
        dataFile = f"input_h5s/bkg_BB_UL{smallYear}/bkg_UL{smallYear}.h5"
    elif sample == "QCDBKG_mjjFlat":
        dataFile = f"input_h5s/bkg_BB_UL{smallYear}_mjjFlat/bkg_UL{smallYear}_mjjFlat.h5"
    elif sample == "DATA":
        dataFile = f"input_h5s/data_UL{smallYear}/data_UL{smallYear}.h5"
    else:
        dataFile = f"input_h5s/signal_UL{smallYear}/{sample}/{sample}.h5"
    f_data = h5py.File(dataFile,"r")
    data = []
    for n in variables:
        if n not in f_data.keys():
            print(f"var {n} not here")
        data.append(f_data[n][()][:min(nTrain,int(0.9*len(f_data[n])))])
    f_data.close()
    data = np.concatenate(data,axis=1)
    return data

def loadMeans(sample,year,variables):
    smallYear = year % 2000
    if sample == "QCDBKG":
        dataFile = f"input_h5s/bkg_BB_UL{smallYear}/bkg_UL{smallYear}.h5"
    elif sample == "QCDBKG_mjjFlat":
        dataFile = f"input_h5s/bkg_BB_UL{smallYear}_mjjFlat/bkg_UL{smallYear}_mjjFlat.h5"
    elif sample == "DATA":
        dataFile = f"input_h5s/data_UL{smallYear}/data_UL{smallYear}.h5"
    else:
        dataFile = f"input_h5s/signal_UL{smallYear}/{sample}/{sample}.h5"
    f_data = h5py.File(dataFile,"r")
    means = []
    stds = []
    for n in variables:
        if n not in f_data.keys():
            print(f"var {n} not here")
        means.append(f_data[n+"_mean"][()])
        stds.append(f_data[n+"_std"][()])
    f_data.close()
    means = np.concatenate(means,axis=1)
    stds = np.concatenate(stds,axis=1)
    return means, stds