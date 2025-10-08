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
from trainingTools import make_flow

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

from sklearn.decomposition import PCA
from sklearn.svm import SVC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else print ('cpu') 

class LossEval:
    """
    Evaluates losses on all available samples using all available trainings for a given year
    """
    def __init__(self,variables,mean_reference="BKG",extras=[],reDoAll=True,modelsToEval=[]):
        self.mean_reference = mean_reference
        self.reDoAll = reDoAll
        self.modelsToEval = modelsToEval
        self.extras = extras
        self.variables = variables
        print('extras = ',self.extras)
    
    def loadData(self,file,variables):
        return hf.loadVariables(file,variables,dEta_cut=-1,mode="all")
    
    def loadMeans(self,variables):
        means = []
        stds = []
        ref_file = "meanStd_dEta1.3.h5" # normalize to values with dEta cut; will be applied for final selection
        with h5py.File(f"input_h5s/{self.mean_reference}/{ref_file}","r") as f:
            for v in self.variables:
                means.append(f[v+"_mean"][()])
                stds.append(f[v+"_std"][()])
        means = np.array(means)
        stds = np.array(stds)
        return means, stds
        
    def evaluateLosses(self,toEval,chunkSize=50000):
        models = [f"combined_trainings/{samp}/{samp}.pt" for samp in \
                  os.listdir("combined_trainings/") if os.path.isdir(f"combined_trainings/{samp}")]
        modDirs = ["/".join(p.split("/")[:-1])+"/" for p in models]
        modNames = [mod.split("/")[1] for mod in models]
        print("Evaluating models : ",modNames)
        
        from samples import bkg_merged_dir, signal_merged_dir, data_merged_dir
        from samples import signal_map
        
        if toEval == "BKG":
            files = [bkg_merged_dir+f for f in os.listdir(bkg_merged_dir) if ".h5" in f]
        elif toEval == "SIG":
            files = [signal_merged_dir+signal_map[s] for s in signal_map.keys()]
        elif toEval == "DATA":
            files = [data_merged_dir+f for f in os.listdir(data_merged_dir) if ".h5" in f]
        elif toEval in siignal_map.keys():
            files = [signal_merged_dir+signal_map[toEval]]
        else:
            print("Evaluation file type not recognized!")
            exit()
        
        # loop through models & evaluate trainings on files
        for j,file in enumerate(files):
            print(f"File = {file}")
            data = self.loadData(file,self.variables)
            means,stds = self.loadMeans(self.variables)
            data = (data-means)/stds
            if data.shape[0] <= chunkSize:
                data = [data]
            else:
                data = np.array_split(data,data.shape[0]//chunkSize + 1)
            for i,mod in enumerate(tqdm(models)):
                modName = modNames[i]
                if len(self.modelsToEval) > 0 and modName not in self.modelsToEval:
                    continue
                #print(f"--- Model = {modName}")
                with open(modDirs[i]+"flow_kwargs.json") as kwf:
                    kwargs = json.load(kwf)
                model = make_flow(len(self.variables),kwargs).to(device)
                model.load_state_dict(torch.load(mod))
                model.eval()
                losses = []
                for chunk in data:
                    chunk_tens = torch.tensor(chunk).to(device)
                    with torch.no_grad():
                        losses.append(-model.eval_log_prob(chunk_tens)[0])
                    del chunk_tens
                    torch.cuda.empty_cache()
                losses = np.concatenate(losses,axis=0)
                with h5py.File(file,"r+") as fout:
                    if modName in fout.keys():
                        del fout[modName]
                    fout.create_dataset(modName,data=losses)
                del model
                torch.cuda.empty_cache()
            del data, means, stds
            
class LossTransform:
    def __init__(self,function,name):
        self.function = function
        self.name = name
    def transform(self,loss):
        return self.function(loss)

class QuakSpaceMaker:
    def __init__(self,bkg_axis,sig_axes,loss_transform=None,loss_reduction=None):
        self.bkg_axis = bkg_axis
        self.sig_axes = sig_axes
        self.loss_transform = loss_transform
        self.loss_reduction = loss_reduction
        if self.loss_transform is None:
            self.loss_transform = LossTransform(lambda x : x,"None")
        if self.loss_reduction is None:
            self.loss_reduction = LossTransform(lambda x : x, "None")
        self.outBase = f"quakSpaces/transform-{self.loss_transform.name}_reduce-{self.loss_reduction.name}/"
        if not os.path.isdir(self.outBase):
            os.makedirs(self.outBase)
        self.outDir = self.outBase + f"{self.bkg_axis}-" + "-".join(self.sig_axes) + "/"
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        
    def EvalBKGLosses_MakePCAs(self):
        # make the background quak space and derive the PCAs
        self.pcas = {}
        loss_axes = [self.bkg_axis] + self.sig_axes + ['mjj']
        losses = hf.loadData("BKG",loss_axes,dEta_cut=1.3,mode="all")
        mjj = losses[:,-1]
        corr_bkg_losses = []
        corr_sig_losses = []
        for i in range(losses.shape[1]-1):
            loss = losses[:,i]
            loss = self.loss_transform.transform(loss)
            pca = PCA()
            toFit = np.concatenate((mjj.reshape(-1,1),loss.reshape(-1,1)),axis=1)
            pca = pca.fit(toFit)
            corr_loss = np.dot(toFit,pca.components_.T)[:,1]
            if i==0:
                corr_bkg_losses.append(corr_loss.reshape(-1,1))
            else:
                corr_sig_losses.append(corr_loss.reshape(-1,1))
            self.pcas[loss_axes[i]] = pca
        corr_bkg_losses = np.concatenate(corr_bkg_losses,axis=1)
        corr_sig_losses = np.concatenate(corr_sig_losses,axis=1)
        corr_sig_losses = self.loss_reduction.transform(corr_sig_losses)
        if len(corr_sig_losses.shape)==1:
            corr_sig_losses = corr_sig_losses.reshape(-1,1)
        outFile = self.outDir+"BKG.npy"
        labels = np.zeros_like(mjj.reshape(-1,1))
        outArray = np.concatenate((mjj.reshape(-1,1),labels,corr_bkg_losses,corr_sig_losses),axis=1)
        np.save(outFile,outArray)
        del losses,mjj,corr_bkg_losses,corr_sig_losses,labels,outArray
        
    def makeQuakSpaces(self):
        from samples import signal_map
        signals = list(signal_map.keys())
        for signal in tqdm(signals):
            axes = [self.bkg_axis] + self.sig_axes + ['mjj']
            losses = hf.loadData(signal,axes,dEta_cut=1.3,mode="all")
            mjj = losses[:,-1]
            bkg_losses = []
            sig_losses = []
            for i in range(losses.shape[1]-1):
                ax = axes[i]
                loss = self.loss_transform.transform(losses[:,i])
                pca = self.pcas[ax]
                new_loss = np.dot(np.concatenate((mjj.reshape(-1,1),loss.reshape(-1,1)),axis=1),pca.components_.T)[:,1]
                if i==0:
                    bkg_losses.append(new_loss.reshape(-1,1))
                else:
                    sig_losses.append(new_loss.reshape(-1,1))
            bkg_losses = np.concatenate(bkg_losses,axis=1)
            sig_losses = self.loss_reduction.transform(np.concatenate(sig_losses,axis=1))
            if len(sig_losses.shape) == 1:
                sig_losses = sig_losses.reshape(-1,1)
            outFile = self.outDir+f"{signal}.npy"
            labels = np.ones_like(mjj.reshape(-1,1))
            outArray = np.concatenate((mjj.reshape(-1,1),labels,bkg_losses,sig_losses),axis=1)
            np.save(outFile,outArray)
            del losses,mjj,bkg_losses,sig_losses,labels,outArray
    
    def run(self):
        self.EvalBKGLosses_MakePCAs()
        self.makeQuakSpaces()
                
def evalKernelSVM(base_dir,sig_name,sig_frac_train=0.8):
    bkg = np.load(f"{base_dir}/QCDBKG.npy")[:,2:4]
    sig = np.load(f"{base_dir}/{sig_name}.npy")[:,2:4]
    total = np.concatenate((bkg,sig),axis=0)
    sig = (sig-total.mean(axis=0))/total.std(axis=0)
    bkg = (bkg-total.mean(axis=0))/total.std(axis=0)
    ntrain = int(sig_frac_train*len(sig))
    sig_train, sig_test = sig[:ntrain], sig[ntrain:]
    bkg_train, bkg_test = bkg[:ntrain], bkg[ntrain:len(sig)]
    print(f"--Train on {ntrain} sig events, {ntrain} bkg events")

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.scatter(bkg_train[:,0],bkg_train[:,1],s=2,label="bkg")
    plt.scatter(sig_train[:,0],sig_train[:,1],s=2,label="sig")
    plt.xlim([-2,5])
    plt.ylim([-2,5])

    train = np.concatenate((sig_train,bkg_train),axis=0)
    labels = np.concatenate((np.ones(sig_train.shape[0]),-1*np.ones(bkg_train.shape[0])),axis=0)
    shuf = np.random.permutation(len(train))
    train = train[shuf]
    labels = labels[shuf]
    svm = SVC(kernel='rbf')
    svm = svm.fit(train,labels)

    plt.subplot(122)
    step=0.05
    x,y = np.meshgrid(np.arange(-2,3+step,step),np.arange(-2,3+step,step))
    inp = np.stack((x,y)).reshape(2,-1).transpose(1,0)
    out = svm.decision_function(inp).reshape(x.shape)
    plt.imshow(out,origin='lower',extent=(-2,3,-2,3),cmap='binary')
    plt.scatter(bkg_test[:,0],bkg_test[:,1],s=2,label="bkg")
    plt.scatter(sig_test[:,0],sig_test[:,1],s=2,label="sig")
    plt.xlim([-2,3])
    plt.ylim([-2,3])

    test = np.concatenate((sig_test,bkg_test),axis=0)
    test_labels = np.concatenate((np.ones(sig_test.shape[0]),-1*np.ones(bkg_test.shape[0])),axis=0)
    shuf = np.random.permutation(len(test))
    test = test[shuf]
    test_labels = test_labels[shuf]
    mean_acc = svm.score(test,test_labels)
    plt.suptitle(f"{sig_name} test acc : {mean_acc:.4f} ({len(sig_test)} sig, {len(bkg_test)} bkg)")
    plt.tight_layout()
    return mean_acc
                