from argparse import ArgumentParser
import re
import os
import json
import time
import multiprocessing
import sys
import math
import time
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
from samples import bkg_merged_dir, signal_merged_dir, data_merged_dir, signal_map
from tqdm import tqdm
import shutil

import helper_functions as hf
from trainingTools import make_flow
from condTrainingTools import make_flow as make_cond_flow

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

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from scipy.stats import boxcox

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else print ('cpu') 

class LossEval:
    """
    Evaluates losses on all available samples using all available trainings for a given year
    """
    def __init__(self,mean_reference="BKG",extras=[],reDoAll=True,modelsToEval=[],overwrite=True):
        self.mean_reference = mean_reference
        self.reDoAll = reDoAll
        self.modelsToEval = modelsToEval
        self.extras = extras
        self.JME_systs = ["JES_up","JES_down","JER_up","JER_down","JMS_up","JMS_down","JMR_up","JMR_down"]
        self.overwrite = overwrite
        if len(self.modelsToEval) > 0 and self.overwrite:
            print("Evaluating specific models but also overwriting, please don't do this")
        print('extras = ',self.extras)
        self.sideband = False
    
    def loadMeans(self,variables):
        means = []
        stds = []
        ref_file = "meanStd.h5" # no dEta cut
        with h5py.File(f"input_h5s/{self.mean_reference}/{ref_file}","r") as f:
            for v in variables:
                means.append(f[v+"_mean"][()])
                stds.append(f[v+"_std"][()])
        means = np.array(means)
        stds = np.array(stds)
        return means, stds
    
    def loadMinMax(self,variables):
        mins = []
        maxes = []
        with open("input_h5s/trainVarMinMaxes.json","r") as f:
            js = json.load(f)
        for v in variables:
            mins.append(js['mins'][v])
            maxes.append(js['maxes'][v])
        mins = np.array(mins,dtype='float32')
        maxes = np.array(maxes,dtype='float32')
        return mins, maxes
        
    def evaluateLosses(self,toEval,chunkSize=100000):
        # some insurance
        if len(self.modelsToEval) > 0 and self.overwrite:
            print("Evaluating specific models but also overwriting, please don't do this")
            exit()
        # find all trained models
        if self.modelsToEval == []:
            models = [f"combined_trainings/{samp}/{samp}.pt" for samp in \
                      os.listdir("combined_trainings/") if os.path.isdir(f"combined_trainings/{samp}") and os.path.exists(f"combined_trainings/{samp}/{samp}.pt")]
            modDirs = ["/".join(p.split("/")[:-1])+"/" for p in models]
            modNames = [mod.split("/")[1] for mod in models]
        else:
            models = [f"combined_trainings/{mod}/{mod}.pt" for mod in self.modelsToEval]
            modDirs = ["/".join(p.split("/")[:-1])+"/" for p in models]
            modNames = [mod.split("/")[1] for mod in models]
        
        from samples import bkg_merged_dir, signal_merged_dir, data_merged_dir
        from samples import signal_map
        
        # make list of files to run over
        isSignal = False
        if toEval == "BKG":
            files = [bkg_merged_dir+f for f in os.listdir(bkg_merged_dir) if ".h5" in f]
            fname = "BKG"
        elif toEval == "BKG_SB":
            files = [bkg_merged_dir+f for f in os.listdir(bkg_merged_dir) if ".h5" in f]
            fname = "BKG_SB"
            self.sideband = True
        elif toEval == "SIG":
            isSignal = True
            files = []
            fnames = []
            for s in signal_map.keys():
                files.append(signal_merged_dir+signal_map[s])
                fnames.append(s)
        elif type(toEval) == list:
            isSignal = True
            files = []
            fnames = []
            for s in toEval:
                files.append(signal_merged_dir+signal_map[s])
                fnames.append(s)
        elif toEval == "DATA":
            files = [data_merged_dir+f for f in os.listdir(data_merged_dir) if ".h5" in f]
            fname = "DATA"
        elif toEval == "DATA_SB":
            files = files = [data_merged_dir+f for f in os.listdir(data_merged_dir) if ".h5" in f]
            fname = "DATA_SB"
            self.sideband = True
        elif toEval in signal_map.keys():
            isSignal = True
            files = [signal_merged_dir+signal_map[toEval]]
            fname = toEval
        else:
            print("Evaluation file type not recognized!")
            exit()
        
        # clean up existing files
        existing_keys = []
        if isSignal:
            for f in fnames:
                if not os.path.exists(f"lossEvals/{f}.h5") and not self.overwrite:
                    print(f"Error: you opted not to overwrite losses but target file lossEvals/{f}.h5 does not exist")
                    exit()
                elif os.path.exists(f"lossEvals/{f}.h5") and not self.overwrite:
                    with h5py.File(f"lossEvals/{f}.h5","r+") as fin:
                        existing_keys = list(fin.keys())
                        if self.modelsToEval != []:
                            for m in self.modelsToEval:
                                if m in existing_keys:
                                    del fin[m]
                elif os.path.exists(f"lossEvals/{f}.h5") and self.overwrite:
                    os.remove(f"lossEvals/{f}.h5")
        else:
            if not os.path.exists(f"lossEvals/{fname}.h5") and not self.overwrite:
                print(f"Error: you opted not to overwrite losses but target file lossEvals/{fname}.h5 does not exist")
                exit()
            elif os.path.exists(f"lossEvals/{fname}.h5") and not self.overwrite:
                with h5py.File(f"lossEvals/{fname}.h5","r+") as fin:
                    existing_keys = list(fin.keys())
                    if self.modelsToEval != []:
                        for m in self.modelsToEval:
                            if m in existing_keys:
                                del fin[m]
            elif os.path.exists(f"lossEvals/{fname}.h5") and self.overwrite:
                os.remove(f"lossEvals/{fname}.h5")
        
        # remove already-evaluated models if we're not overwriting and we just want to run over new trainings
        if not self.overwrite and self.modelsToEval == []: # just want to run over new trainings
            keep = []
            for i,name in enumerate(modNames):
                if name not in existing_keys:
                    keep.append(i)
            models = [models[k] for k in keep]
            modNames = [modNames[k] for k in keep]
            modDirs = [modDirs[k] for k in keep]
            
        
        # pre-loading which variables we'll need
        all_vars = []
        for md in modDirs:
            with open(f"{md}/flow_kwargs.json","r") as f:
                kwargs = json.load(f)
            for v in kwargs['variables']:
                all_vars.append(v)
                if "rho" in v and isSignal:
                    for sys in self.JME_systs:
                        all_vars.append(v.replace("rho",f"rho_{sys}"))
        all_vars = list(set(all_vars))
        all_vars += ["mjj"]
                
        # write non-loss outputs to file (only if we're overwriting)
        if self.overwrite:
            print("writing mass & systematics")
            for j,file in enumerate(tqdm(files)):
                outName = fnames[j] if isSignal else fname
                ## load from input
                with h5py.File(file,"r") as inFile:
                    if self.sideband:
                        mjj = hf.loadVariables(file,["mjj"],dEta_cut=-1,sideband=True,mode="all")[:,0]
                    else:
                        mjj = hf.loadVariables(file,["mjj"],dEta_cut=1.3,mode="all")[:,0]
                    if isSignal:
                        dEta = hf.getVariable(inFile,"dEta")
                        cut = np.abs(dEta) < 1.3
                        sys_weights = inFile['sys_weights'][()][cut]
                ## write to output
                with h5py.File(f"lossEvals/{outName}.h5","a") as fout:
                    # write mjj
                    if "mjj" in fout.keys():
                        prev_size = fout["mjj"].shape[0]
                        new_size = prev_size + mjj.shape[0]
                        fout["mjj"].resize(new_size,axis=0)
                        fout["mjj"][prev_size:] = mjj
                    else:
                        fout.create_dataset("mjj",data=mjj,chunks=True,maxshape=(None,))
                    # write sys_weights if signal
                    if isSignal:
                        if "sys_weights" in fout.keys():
                            prev_size = fout["sys_weights"].shape[0]
                            new_size = sys_weights.shape[0] + prev_size
                            fout["sys_weights"].resize(new_size,axis=0)
                            fout["sys_weights"][prev_size:] = sys_weights
                        else:
                            fout.create_dataset("sys_weights",data=sys_weights,chunks=True,maxshape=(None,21))
        
        # loop over files
        for j,file in enumerate(files):
            # load all the data we'll need
            if self.sideband:
                alldata = hf.loadVariables(file,all_vars,dEta_cut=-1,sideband=True,mode="all")
            else:
                alldata = hf.loadVariables(file,all_vars,dEta_cut=1.3,mode="all")
            alldata = {all_vars[i]:alldata[:,i] for i in range(len(all_vars))}
            # loop over models
            for i,mod in enumerate(tqdm(models)):
                modName = modNames[i]
                with open(modDirs[i]+"flow_kwargs.json") as kwf:
                    kwargs = json.load(kwf)
                variables = kwargs['variables']
                conditional = False
                if 'context_vars' in kwargs.keys():
                    conditional = True
                    context_vars = kwargs['context_vars']
                    model = make_cond_flow(len(variables),len(context_vars),kwargs)
                else:
                    model = make_flow(len(variables),kwargs).to(device)
                model.load_state_dict(torch.load(mod))
                model.eval()
                # prepare output file and sets of vars to evaluate on
                variableSets = [variables]
                variableSetNames = [""]
                if isSignal:
                    outName = fnames[j]
                    # need to evaluate for each JME systematic variation
                    for syst in self.JME_systs:
                        syst_vars = [v.replace("rho",f"rho_{syst}") for v in variables]
                        variableSets.append(syst_vars)
                        variableSetNames.append(f"_{syst}")
                else:
                    outName = fname
                for varset, sysname in zip(variableSets,variableSetNames):
                    # load and normalize
                    if conditional:
                        data = np.concatenate([alldata[vname].reshape(-1,1) for vname in varset+context_vars],axis=1)
                        means,stds = self.loadMeans(variables+context_vars)
                        mins, maxes = self.loadMinMax(variables+context_vars)
                    else:
                        data = np.concatenate([alldata[vname].reshape(-1,1) for vname in varset],axis=1)
                        means,stds = self.loadMeans(variables)
                        mins, maxes = self.loadMinMax(variables)
                    if kwargs['minmax']:
                        data = data-means
                        scale = np.where(np.abs(maxes-means)>np.abs(mins-means),np.abs(maxes-means),np.abs(mins-means))
                        data = kwargs['rangeScale']*(data/scale)
                    else:
                        data = (data-means)/stds
                    # split into chunks for passing through network
                    if data.shape[0] <= chunkSize:
                        data = [data]
                    else:
                        data = np.array_split(data,data.shape[0]//chunkSize + 1)
                    
                    # evaluate loss on chunks
                    losses = []
                    for chunk in data:
                        chunk_tens = torch.tensor(chunk).to(device)
                        with torch.no_grad():
                            if conditional:
                                losses.append(-model.eval_log_prob(inputs=chunk_tens[:,:-len(context_vars)],context=chunk_tens[:,-len(context_vars):])[0])
                            else:
                                losses.append(-model.eval_log_prob(chunk_tens)[0])
                        del chunk_tens
                        torch.cuda.empty_cache()
                    losses = np.concatenate(losses,axis=0)
                    # write to output
                    with h5py.File(f"lossEvals/{outName}.h5","a") as fout:
                        bName = modName+sysname
                        if bName in fout.keys():
                            prev_size = fout[bName].shape[0]
                            new_size = prev_size + losses.shape[0]
                            fout[bName].resize(new_size,axis=0)
                            fout[bName][prev_size:] = losses
                        else:
                            fout.create_dataset(f"{bName}",data=losses,chunks=True,maxshape=(None,))
                    del data, means, stds
                del model
                torch.cuda.empty_cache()
            del alldata
            torch.cuda.empty_cache()
                
def normalizeLosses(decorr_type,doBoxCox=True,overwrite=True,toEval=[]):
    if decorr_type == "pca":
        outDir = "lossEvals_pcaDecorr"
    elif decorr_type == "linear":
        outDir = "lossEvals_linearDecorr"
    elif decorr_type == "none":
        outDir = "lossEvals_noDecorr"
    else:
        print("invalid mass decorrelation choice")
        exit()
    
    if doBoxCox:
        outDir += "_boxCox"
        
    if not os.path.isdir(outDir):
        print(f"need to setup the directory {outDir} first")
        exit()

    bkg_file = "lossEvals/BKG.h5"
    sig_files = ["lossEvals/"+f for f in os.listdir("lossEvals/") if f != "BKG.h5" and ".h5" in f]
    
    existing_losses = []
    if not overwrite and os.path.exists(f"{outDir}/BKG.h5"):
        with h5py.File(f"{outDir}/BKG.h5","r") as fin:
            existing_losses = list(fin.keys())
    if not overwrite and not os.path.exists(f"{outDir}/BKG.h5"):
        print("not overwriting, but target files don't exist")
        exit()

    # list of signal losses
    with h5py.File(bkg_file,"r") as f:
        if toEval != []:
            losses = toEval
        else:
            losses = [k for k in list(f.keys()) if k != "mjj" and k!="truth_label" and k!='jet1_mass' and k!='jet2_mass' and k not in existing_losses]

    # fit PCAs and linears
    if decorr_type != "none":
        print("Fitting PCAs and linear fits")
        transforms = {}
        with h5py.File(bkg_file,"r") as f:
            mjj = f["mjj"][()]
            for l in tqdm(losses):
                if decorr_type == "pca":
                    transforms[l] = PCA().fit(np.concatenate((mjj.reshape(-1,1),f[l][()].reshape(-1,1)),axis=1))
                elif decorr_type == "linear":
                    transforms[l] = LinearRegression().fit(mjj.reshape(-1,1),f[l][()].reshape(-1,1))


    print("Applying transformations, recording min/max loss values")
    maxes = {l:-99999 for l in losses}
    mins = {l:99999 for l in losses}
    systematics = ["JES_up","JES_down","JER_up","JER_down","JMS_up","JMS_down","JMR_up","JMR_down"]
    for f in tqdm([bkg_file] + sig_files):
        fname = f.split("/")[-1]
        outf = f"{outDir}/{fname}"
        if overwrite:
            if os.path.exists(outf):
                os.remove(outf)
            shutil.copy(f,outf)
        with h5py.File(f,"r") as fin:
            print(f)
            mjj = fin["mjj"][()]
            for l in losses:
                possible = [l] + [f"{l}_{s}" for s in systematics]
                relevant = [k for k in list(fin.keys()) if k in possible]
                for b in relevant:
                    loss = fin[b][()]
                    if decorr_type == "pca":
                        new = np.dot(np.concatenate((mjj.reshape(-1,1),loss.reshape(-1,1)),axis=1),transforms[l].components_.T)[:,1]
                    elif decorr_type == "linear":
                        new = loss - transforms[l].predict(mjj.reshape(-1,1))[:,0] + transforms[l].intercept_
                    else:
                        new = loss
                    lmax = new.max()
                    lmin = new.min()
                    if lmax > maxes[l]:
                        maxes[l] = lmax
                    if lmin < mins[l]:
                        mins[l] = lmin

                    with h5py.File(outf,"r+") as fout:
                        if b not in fout.keys():
                            fout.create_dataset(b,data=new)
                        else:
                            fout[b][...] = new

    if doBoxCox:
        print("Applying box-cox transformed losses & centering background losses on 0")
    else:
        print("Centering background losses on 0")
    lambdas = {}
    means = {}
    sds = {}
    bkg_file = f"{outDir}/BKG.h5"
    sig_files = [f"{outDir}/"+f for f in os.listdir(f"{outDir}/") if f != "BKG.h5" and ".h5" in f]
    with h5py.File(bkg_file,"r+") as fout:
        for l in tqdm(losses):
            x = fout[l][()]
            if doBoxCox:
                y = 1+x-mins[l] # shift to make positive
                y,lam = boxcox(y)
                lambdas[l] = lam
                means[l] = y.mean()
                sds[l] = y.std()
                y = (y-means[l])/sds[l]
            else:
                means[l] = x.mean()
                sds[l] = x.std()
                y = (x-means[l])/sds[l]
            fout[l][...] = y
    # apply boxcox to signal samples
    for f in tqdm(sig_files):
        fname = f.split("/")[-1]
        outf = f"{outDir}/{fname}"
        with h5py.File(outf,"r+") as fout:
            for l in losses:
                possible = [l] + [f"{l}_{s}" for s in systematics]
                relevant = [k for k in list(fout.keys()) if k in possible]
                for b in relevant:
                    x = fout[b][()]
                    if doBoxCox:
                        lam = lambdas[l]
                        y = boxcox(1 + x - mins[l],lmbda=lam)
                        y = (y-means[l])/sds[l]
                    else:
                        y = (x-means[l])/sds[l]
                    fout[b][...] = y
                    
def lossPlots(bkg_axis,sig_axes,sig_alias,sample,reducs=[],decorr=True,base="lossEvals_pcaDecorr_boxCox",negPenalty=False):
    from sklearn.linear_model import LinearRegression
    if decorr:
        outDir = f"plots/lossPlots/{base}/BKG-{bkg_axis}_SIG-{sig_alias}/withDecorr/"
    else:
        outDir = f"plots/lossPlots/{base}/BKG-{bkg_axis}_SIG-{sig_alias}/noDecorr/"
    if not os.path.isdir(outDir):
        os.makedirs(outDir)

    from quakSpaceMaker import reductions
    sig_losses_bkg = []
    with h5py.File(f"{base}/BKG.h5","r") as f:
        bkg_loss_bkg = f["BKG_mjjFlat"][()]
        for ax in sig_axes:
            sig_losses_bkg.append(f[ax][()].reshape(-1,1))
    sig_losses_bkg = np.concatenate(sig_losses_bkg,axis=1)
    bad_bkg = sig_losses_bkg < 0
    
    sig_losses_sig = []
    with h5py.File(f"{base}/{sample}.h5","r") as f:
        bkg_loss_sig = f["BKG_mjjFlat"][()]
        for ax in sig_axes:
            sig_losses_sig.append(f[ax][()].reshape(-1,1))
    sig_losses_sig = np.concatenate(sig_losses_sig,axis=1)
    bad_sig = sig_losses_sig < 0
    
    if decorr:
        fit_bkg = LinearRegression().fit(bkg_loss_bkg.reshape(-1,1),sig_losses_bkg)
        sig_losses_bkg = sig_losses_bkg - fit_bkg.predict(bkg_loss_bkg.reshape(-1,1))
        sig_losses_sig = sig_losses_sig - fit_bkg.predict(bkg_loss_sig.reshape(-1,1))
        
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.title('BKG correlations')
    corr_bkg = np.corrcoef(np.concatenate((bkg_loss_bkg.reshape(-1,1),sig_losses_bkg),axis=1),rowvar=False)
    plt.imshow(corr_bkg)
    plt.xticks(np.arange(len(sig_axes)+1))
    plt.gca().set_xticklabels([bkg_axis]+sig_axes,rotation=45)
    plt.subplot(122)
    plt.title(f'{sample} correlations')
    corr_bkg = np.corrcoef(np.concatenate((bkg_loss_sig.reshape(-1,1),sig_losses_sig),axis=1),rowvar=False)
    plt.imshow(corr_bkg)
    plt.xticks(np.arange(len(sig_axes)+1))
    plt.gca().set_xticklabels([bkg_axis]+sig_axes,rotation=45)
    plt.savefig(f"{outDir}/{sample}_correlations.png")
    
    if negPenalty:
        sig_losses_bkg[bad_bkg] = np.max(sig_losses_bkg)
        sig_losses_sig[bad_sig] = np.max(sig_losses_sig)
    
    reduc_losses_bkg = [reductions[r].transform(sig_losses_bkg) for r in reducs]
    reduc_losses_sig = [reductions[r].transform(sig_losses_sig) for r in reducs]

    plt.figure(figsize=(16,6))
    plt.subplot(121)
    minx = np.min(np.quantile(np.concatenate((bkg_loss_bkg.reshape(-1,1),sig_losses_bkg),axis=1),0.0001,axis=0))
    maxx = np.max(np.quantile(np.concatenate((bkg_loss_bkg.reshape(-1,1),sig_losses_bkg),axis=1),0.9999,axis=0))
    binsx = np.linspace(minx,maxx,100)
    h = plt.hist(bkg_loss_bkg,bins=binsx,histtype='step',label=bkg_axis)
    for i,ax in enumerate(sig_axes):
        h = plt.hist(sig_losses_bkg[:,i],bins=binsx,histtype='step',label=ax)
    plt.yscale('log')
    plt.title("BKG; linear decorr + box cox")
    plt.legend(loc='best')
    plt.subplot(122)
    plt.boxplot(np.concatenate((bkg_loss_bkg.reshape(-1,1),sig_losses_bkg),axis=1))
    labels = [bkg_axis]+sig_axes
    plt.xticks(np.arange(1,len(sig_axes)+1+1))
    plt.gca().set_xticklabels(labels,rotation=45)
    plt.ylim([-10,10])
    plt.grid(axis='y')
    plt.savefig(f"{outDir}/bkg_losses.png")
    

    plt.figure(figsize=(16,6))
    plt.subplot(121)
    minx = np.min(np.quantile(np.concatenate((bkg_loss_sig.reshape(-1,1),sig_losses_sig),axis=1),0.00001,axis=0))
    maxx = np.max(np.quantile(np.concatenate((bkg_loss_sig.reshape(-1,1),sig_losses_sig),axis=1),0.99999,axis=0))
    binsx = np.linspace(minx,maxx,100)
    h = plt.hist(bkg_loss_sig,bins=binsx,histtype='step',label=bkg_axis)
    for i,ax in enumerate(sig_axes):
        h = plt.hist(sig_losses_sig[:,i],bins=binsx,histtype='step',label=ax)
    plt.yscale('log')
    plt.title(f"{sample}; linear decorr + box cox")
    plt.legend(loc='best')
    plt.subplot(122)
    plt.boxplot(np.concatenate((bkg_loss_sig.reshape(-1,1),sig_losses_sig),axis=1))
    labels = [bkg_axis]+sig_axes
    plt.xticks(np.arange(1,len(sig_axes)+1+1))
    plt.gca().set_xticklabels(labels,rotation=45)
    plt.ylim([-10,10])
    plt.grid(axis='y')
    plt.savefig(f"{outDir}/{sample}_losses.png")
        
    from matplotlib.colors import LogNorm
    for i,r in enumerate(reducs):
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        minx = min(np.quantile(reduc_losses_bkg[i],0.0001),np.quantile(reduc_losses_sig[i],0.0001))
        maxx = max(np.quantile(reduc_losses_bkg[i],0.9999),np.quantile(reduc_losses_sig[i],0.9999))
        bins = np.linspace(minx,maxx,100)
        ax = plt.gca()
        ax2 = ax.twinx()
        hb,bins,_ = ax.hist(reduc_losses_bkg[i],color="C0",bins=bins,histtype='step',label="bkg",density=True)
        hs = ax.hist(reduc_losses_sig[i],color="C1",bins=bins,histtype='step',label=sample,density=True)
        hb,bins,_ = ax2.hist(reduc_losses_bkg[i],color="C0",bins=bins,histtype='step',density=True,linestyle=':')
        hs = ax2.hist(reduc_losses_sig[i],color="C1",bins=bins,histtype='step',density=True,linestyle=':')
        ax2.set_yscale('log')
        #plt.legend(loc='best')
        plt.title(r)
        plt.subplot(122)
        minx = min(np.quantile(bkg_loss_bkg,0.0001),np.quantile(bkg_loss_sig,0.0001))
        maxx = max(np.quantile(bkg_loss_bkg,0.9999),np.quantile(bkg_loss_sig,0.9999))
        miny = min(np.quantile(reduc_losses_bkg[i],0.0001),np.quantile(reduc_losses_sig[i],0.0001))
        maxy = max(np.quantile(reduc_losses_bkg[i],0.9999),np.quantile(reduc_losses_sig[i],0.9999))
        binsx = np.linspace(minx,maxx,100)
        binsy = np.linspace(miny,maxy,100)
        nb,xb,yb = np.histogram2d(bkg_loss_bkg,reduc_losses_bkg[i],bins=(binsx,binsy),density=True)
        ns,xs,ys = np.histogram2d(bkg_loss_sig,reduc_losses_sig[i],bins=(binsx,binsy),density=True)
        norm=LogNorm(vmin=min(np.min(ns[ns>0]),np.min(nb[nb>0])),vmax=max(nb.max(),ns.max()))
        extent = [xb[0],xb[-1],yb[0],yb[-1]]
        plt.imshow(ns.T,extent=extent,origin='lower',cmap="Oranges",alpha=1,norm=norm,aspect='auto')
        plt.imshow(nb.T,extent=extent,origin='lower',cmap="Blues",alpha=1,norm=norm,aspect='auto')
        plt.savefig(f"{outDir}/{sample}_{r}_quakSpace.png")


def roc_effs(sig,den_sig,bkg,den_bkg):
    left, right = min(sig.min(),bkg.min()), max(sig.max(),bkg.max())
    cuts = np.linspace(left,right,1000)
    sig_eff = []
    bkg_eff = []
    for c in cuts:
        sig_eff.append(np.count_nonzero(sig<c)/den_sig)
        bkg_eff.append(np.count_nonzero(bkg<c)/den_bkg)
    sig_eff = np.array(sig_eff)
    bkg_eff = np.array(bkg_eff)
    return sig_eff, bkg_eff
        
def load_losses(fs,fb,bkg_axis,sig_axes,decorr=True):
    sig_losses_bkg = []
    with h5py.File(fb,"r") as f:
        bkg_loss_bkg = f[bkg_axis][()]
        for ax in sig_axes:
            sig_losses_bkg.append(f[ax][()].reshape(-1,1))
    sig_losses_bkg = np.concatenate(sig_losses_bkg,axis=1)
    
    sig_losses_sig = []
    with h5py.File(fs,"r") as f:
            bkg_loss_sig = f[bkg_axis][()]
            for ax in sig_axes:
                sig_losses_sig.append(f[ax][()].reshape(-1,1))
    sig_losses_sig = np.concatenate(sig_losses_sig,axis=1)
    
    if decorr:
        fit_bkg = LinearRegression().fit(bkg_loss_bkg.reshape(-1,1),sig_losses_bkg)
        sig_losses_bkg = sig_losses_bkg - fit_bkg.predict(bkg_loss_bkg.reshape(-1,1))
        sig_losses_sig = sig_losses_sig - fit_bkg.predict(bkg_loss_sig.reshape(-1,1))
    
    return bkg_loss_bkg, sig_losses_bkg, bkg_loss_sig, sig_losses_sig
    
def sic_curve(bkg_axis,sig_axes,sig_alias,reduction,sample,decorr=True,base="lossEvals_pcaDecorr_boxCox"):
    from sklearn.linear_model import LinearRegression
    from quakSpaceMaker import reductions
    base_old = "/home/sambt/CASE_final/CASE_QUAK_final/finalTraining_nodEtaCut_old/"+base
    base_new = "/home/sambt/CASE_final/CASE_QUAK_final/finalTraining_nodEtaCut/"+base
    outDir = f"plots/sics/{base}/{bkg_axis}-{sig_alias}/"
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    outFile = outDir+f"{sample}_{reduction}_decorr" if decorr else outDir+f"{sample}_{reduction}_noDecorr"
    
    fig,axes = plt.subplots(1,2,figsize=(7,5),gridspec_kw={'width_ratios':[5,2]})
    plt.sca(axes[0])
    colors = [f'C{i}' for i in range(10)]
    lines = ['-','--']
    labels = ["old","new"]
    for j,fbase in enumerate([base_old,base_new]):
        bkg_loss_bkg, sig_losses_bkg, bkg_loss_sig, sig_losses_sig = load_losses(f"{fbase}/{sample}.h5",f"{fbase}/BKG.h5",bkg_axis,sig_axes,decorr=decorr)
        bkg = reductions[reduction].transform(sig_losses_bkg)
        sig = reductions[reduction].transform(sig_losses_sig)
        quantiles = [0,0.25,0.5,0.7,0.9,1]
        qvals = np.quantile(bkg_loss_sig,quantiles)
        for i in range(len(quantiles)-1):
            lo = qvals[i]
            hi = qvals[i+1]
            cut_s = (bkg_loss_sig > lo) & (bkg_loss_sig < hi)
            cut_b = (bkg_loss_bkg > lo) & (bkg_loss_bkg < hi)
            s = sig[cut_s]
            b = bkg[cut_b]        
            ns = s.shape[0]
            nb = b.shape[0]
            sig_eff, bkg_eff = roc_effs(s,ns,b,nb)
            srt = np.argsort(sig_eff)
            sig_eff = sig_eff[srt]
            bkg_eff = bkg_eff[srt]
            sig_eff = sig_eff[bkg_eff>0]
            bkg_eff = bkg_eff[bkg_eff>0]
            yval = sig_eff/np.sqrt(bkg_eff)
            plt.plot(sig_eff,yval,label=f"[{quantiles[i]:.2f},{quantiles[i+1]:.2f}], {labels[j]}",color=colors[i],linestyle=lines[j])
    plt.ylim([0.002,20])
    title = f"{base.split('/')[-1]}"
    plt.yscale('log')
    leg = plt.legend(loc='lower right',ncol=2)
    plt.xlabel("$\epsilon_\mathrm{sig}$")
    plt.ylabel("$\epsilon_\mathrm{sig}/\sqrt{\epsilon_\mathrm{bkg}}$")
    
    if decorr:
        title += ", decorrelated"
    plt.title(title)
    axes_label = "\n-"+"\n-".join(sig_axes)
    plt.sca(axes[1])
    plt.axis('off')
    plt.text(0,0.6,f"A0 : {bkg_axis}" "\n" f"A1 : {reduction}" "\n" f"{sample}",transform=plt.gca().transAxes)
    plt.text(0,0.3,f"Axes:{axes_label}",transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(outFile+".png")
    
def sic_fixedEff(bkg_axis,sig_axes,sig_alias,reducs,sample,eff=0.2,decorr=True,base="lossEvals_pcaDecorr_boxCox"):
    from sklearn.linear_model import LinearRegression
    from quakSpaceMaker import reductions
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    base_old = "/home/sambt/CASE_final/CASE_QUAK_final/finalTraining_nodEtaCut_old/"+base
    base_new = "/home/sambt/CASE_final/CASE_QUAK_final/finalTraining_nodEtaCut/"+base
    outDir = f"plots/sic_fixedEff/{base}/{bkg_axis}-{sig_alias}/"
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    outFile = outDir+f"{sample}_eff{eff}_decorr" if decorr else outDir+f"{sample}_eff{eff}_noDecorr"
    
    fig,axes = plt.subplots(1,2,figsize=(7,5),gridspec_kw={'width_ratios':[5,2]})
    plt.sca(axes[0])
    colors = [f'C{i}' for i in range(10)]
    markers = ['.','^']
    labels = ["old","new"]
    xvals = np.arange(len(reducs))+0.5
    quantiles = [0,0.25,0.5,0.7,0.9,1]
    offsets = np.linspace(-0.4,0.4,len(quantiles)-1)
    for j,fbase in enumerate([base_old,base_new]):
        bkg_loss_bkg, sig_losses_bkg, bkg_loss_sig, sig_losses_sig = load_losses(f"{fbase}/{sample}.h5",f"{fbase}/BKG.h5",bkg_axis,sig_axes,decorr=decorr)
        qvals = np.quantile(bkg_loss_sig,quantiles)
        for k,reduc in enumerate(reducs):
            bkg = reductions[reduc].transform(sig_losses_bkg)
            sig = reductions[reduc].transform(sig_losses_sig)
            for i in range(len(quantiles)-1):
                lo = qvals[i]
                hi = qvals[i+1]
                cut_s = (bkg_loss_sig > lo) & (bkg_loss_sig < hi)
                cut_b = (bkg_loss_bkg > lo) & (bkg_loss_bkg < hi)
                s = sig[cut_s]
                b = bkg[cut_b]        
                ns = s.shape[0]
                nb = b.shape[0]
                sig_eff, bkg_eff = roc_effs(s,ns,b,nb)
                ind = np.argmin(np.abs(sig_eff-eff))
                num = sig_eff[ind]
                den = bkg_eff[ind]
                yval = num/np.sqrt(den)
                x_plot = xvals[k] + offsets[i]
                plt.scatter(x_plot,yval,color=colors[i],marker=markers[j],s=50)
    # getting legend stuff right
    l1 = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                          markersize=6, label='Old')
    l2 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=6, label='New')
    leg1 = plt.gca().legend(handles=[l1,l2],loc='upper center')
    plt.gca().add_artist(leg1)
    color_patches = []
    for i in range(len(quantiles)-1):
        color_patches.append(mpatches.Patch(color=colors[i],label=f"[{quantiles[i]},{quantiles[i+1]}]"))
    plt.gca().legend(handles=color_patches,loc='upper right')            
    plt.xticks(ticks=xvals,labels=reducs,rotation=45)
    plt.ylabel("$\epsilon_\mathrm{sig}/\sqrt{\epsilon_\mathrm{bkg}}$ at $\epsilon_\mathrm{sig} = "+f"{eff}"+"$")
    
    plt.xlim([0,len(reducs)])
    plt.ylim([0,15])
    # Major ticks
    ticks2 = np.arange(len(reducs)+1)
    plt.gca().set_xticks(ticks2)
    plt.gca().set_xticklabels([])
    plt.grid(axis='x')

    # Minor ticks
    ticks1 = xvals
    labs = reducs
    plt.gca().set_xticks(ticks1, minor=True)
    plt.gca().set_xticklabels(labs, minor=True)
    plt.gca().tick_params(axis='x', which="minor",length=0)
    
    title = f"{base.split('/')[-1]}"
    if decorr:
        title += ", decorrelated"
    plt.title(title)
    plt.sca(axes[1])
    plt.axis('off')
    axes_label = "\n-"+"\n-".join(sig_axes)
    plt.text(0,0.6,f"A0 : {bkg_axis}" "\n" f"{sample}",transform=plt.gca().transAxes)
    plt.text(0,0.3,f"Axes:{axes_label}",transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(outFile+".png")

class LossTransform:
    def __init__(self,function,name):
        self.function = function
        self.name = name
    def transform(self,loss):
        return self.function(loss)

# Defining loss reductions (add yours here!)
def normLoss(order,name,offset=0):
    f = lambda x : np.linalg.norm(x+offset,ord=order,axis=1)
    return LossTransform(f,name)

minLoss = LossTransform(lambda x : np.min(x,axis=1),"Min")
sumLoss = LossTransform(lambda x : np.sum(x,axis=1)/x.shape[1],"Sum")
sumProbLoss = LossTransform(lambda x : -np.log(np.sum(np.exp(-x),axis=1)), "SumProb")
Identity = LossTransform(lambda x : x, "None")

reductions = {
    "inv2norm":normLoss(-2,"inv2norm"),
    "inv5norm":normLoss(-5,"inv5norm"),
    "inv10norm":normLoss(-10,"inv10norm"),
    "inv2norm_o10":normLoss(-2,"inv2norm_o10",offset=10),
    "inv5norm_o10":normLoss(-5,"inv5norm_o10",offset=10),
    "inv10norm_o10":normLoss(-10,"inv10norm_o10",offset=10),
    "Min":minLoss,
    "Sum":sumLoss,
    "SumProb":sumProbLoss,
    "None":Identity
}

# Defining loss transformations (add yours here!)
transformations = {
    "None":Identity
}


class QuakSpaceMaker:
    def __init__(self,bkg_axis,sig_axes,loss_reduction="None",loss_transform="None",decorrelation="linear",normalized_losses=False):
        """
        bkg_axis = background training to use for the background axis
        sig_axes = signal training(s) to use for signal axis/axes
        loss_transform = transformation to apply to all losses before making QUAK space
        loss_reduction = reduction to apply to signal losses (e.g. sum, norm, etc.)
        """
        assert type(bkg_axis) == str
        assert type(sig_axes) == list
        self.bkg_axis = bkg_axis
        self.sig_axes = sig_axes
        self.axes = [bkg_axis] + sig_axes
        self.decorrelation = decorrelation
        self.loss_transform = transformations[loss_transform]
        self.loss_reduction = reductions[loss_reduction]
        self.normalized_losses = normalized_losses
        
        if self.normalized_losses:
            self.outBase = f"quakSpaces_normalizedLosses/transform-{self.loss_transform.name}_reduce-{self.loss_reduction.name}/"
        else:
            self.outBase = f"quakSpaces/transform-{self.loss_transform.name}_reduce-{self.loss_reduction.name}/"
        if not os.path.isdir(self.outBase):
            os.makedirs(self.outBase)
        
        self.outDir = self.outBase + f"{self.bkg_axis}-" + "-".join(self.sig_axes) + "/"
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
            
        print(f"Bkg axis : {self.bkg_axis}")
        print(f"Sig axes : {self.sig_axes}")
        print(f"Sig Reduction : {loss_reduction}")
        print(f"Mjj decorrelation : {decorrelation}")
        
    def run(self):    
        signals = list(signal_map.keys())
        if self.normalized_losses:
            h5dir = "lossEvals_normalized"
        else:
            h5dir = "lossEvals"
            
        #pre-compute the PCAs or regressions
        bkg_mjj = None
        self.regs = {}
        self.pcas = {}
        with h5py.File(f"{h5dir}/BKG.h5","r") as f:
            bkg_losses = {}
            bkg_mjj = f["mjj"][()]
            for ax in self.axes:
                bkg_losses[ax] = self.loss_transform.transform(f[ax][()])
                if self.decorrelation == "linear":
                    self.regs[ax] = LinearRegression()
                    self.regs[ax] = self.regs[ax].fit(bkg_mjj.reshape(-1,1),bkg_losses[ax].reshape(-1,1))
                elif self.decorrelation == "pca":
                    self.pcas[ax] = PCA()
                    self.pcas[ax] = self.pcas[ax].fit(np.concatenate((bkg_mjj.reshape(-1,1),bkg_losses[ax].reshape(-1,1)),axis=1))
                elif self.decorrelation == "None":
                    pass
                else:
                    print("Invalid decorrelation specifier")
                    exit()
            del bkg_losses, bkg_mjj
        
        # loop over signals & bkg and make the quak spaces
        for sample in tqdm(["BKG"] + signals):
            # load injection losses
            losses = {}
            mjj = None
            with h5py.File(f"{h5dir}/{sample}.h5","r") as f:
                mjj = f["mjj"][()]
                for ax in self.axes:
                    losses[ax] = self.loss_transform.transform(f[ax][()])

            # perform (mjj,loss) decorrelation for each loss
            for ax in self.axes:
                if self.decorrelation == "linear":
                    losses[ax] = losses[ax] - self.regs[ax].predict(mjj.reshape(-1,1))[:,0] + self.regs[ax].intercept_
                elif self.decorrelation == "pca":
                    losses[ax] = np.dot(np.concatenate((mjj.reshape(-1,1),losses[ax].reshape(-1,1)),axis=1),self.pcas[ax].components_.T)[:,1]
                elif self.decorrelation == "None":
                    pass
                else:
                    print("Invalid decorrelation specifier")
                    exit()
                    
            bkg_loss = losses[self.bkg_axis]
            sig_loss = self.loss_reduction.transform(np.concatenate([losses[ax].reshape(-1,1) for ax in self.sig_axes],axis=1))
            
            outFile = self.outDir+f"{sample}.npy"
            labels = np.zeros(mjj.shape[0]) if sample=="BKG" else np.ones(mjj.shape[0])
            outArr = np.concatenate((mjj.reshape(-1,1),labels.reshape(-1,1),bkg_loss.reshape(-1,1),sig_loss.reshape(-1,1)),axis=1)
            np.save(outFile,outArr)
                
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