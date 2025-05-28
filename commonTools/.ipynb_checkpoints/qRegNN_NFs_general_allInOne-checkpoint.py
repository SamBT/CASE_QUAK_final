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

sys.path.append("/home/sambt/CASE_final/CASE_QUAK_final/")
from nflows.flows.base import Flow
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseQuadraticAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform, ConditionalRationalQuadraticAutoregressiveTransform, IndependentRQS, AutoregressiveRQS
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

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers,act=None,out_act=None):
        super().__init__()
        layers = []
        dcurr = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(dcurr, hidden_dim))
            layers.append(nn.ReLU() if act is None else act)
            dcurr = hidden_dim
        layers.append(nn.Linear(dcurr, output_dim))
        layers.append(nn.Tanh() if out_act is None else out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class chainedNFTrainer:
    def __init__(self,projName,bkg_train,bkg_test,data_train,data_test,varNames,control=[],NF_kwargs={},outDir="NF_models_QR_oneShot/",rangeScale=3):
        assert len(varNames) == bkg_train.shape[1] and len(varNames) == data_train.shape[1]
        self.projName = projName
        self.varNames = varNames
        self.control = control
        self.fit = [i for i in range(len(varNames)) if i not in control]
        self.controlVars = [varNames[k] for k in control]
        self.fitVars = [varNames[i] for i in range(len(varNames)) if i not in control]
        self.NF_kwargs = NF_kwargs
        self.outDir = f"{outDir}/{projName}/"
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        self.rangeScale = rangeScale
        
        self.varDict = {v:v for v in varNames}
        
        # saving means, stds, mins, maxes for scaling
        bkg_all = np.concatenate((bkg_train,bkg_test),axis=0)
        self.bkg_maxes = bkg_all.max(axis=0)
        self.bkg_mins = bkg_all.min(axis=0)
        #self.bkg_maxes = np.quantile(bkg_all,0.9999,axis=0)
        #self.bkg_mins = np.quantile(bkg_all,0.0001,axis=0)
        self.bkg_means = bkg_all.mean(axis=0)
        self.bkg_stds = bkg_all.std(axis=0)
        data_all = np.concatenate((data_train,data_test),axis=0)
        self.data_maxes = data_all.max(axis=0)
        self.data_mins = data_all.min(axis=0)
        #self.data_maxes = np.quantile(data_all,0.9999,axis=0)
        #self.data_mins = np.quantile(data_all,0.0001,axis=0)
        self.data_means = data_all.mean(axis=0)
        self.data_stds = data_all.std(axis=0)
        
        # write basic info to json in base directory
        out_js = {}
        out_js['projName'] = self.projName
        out_js['varNames'] = self.varNames
        out_js['control'] = control
        out_js['controlVars'] = self.controlVars
        out_js['fitVars'] = self.fitVars
        out_js['baseDir'] = self.outDir
        out_js['rangeScale'] = self.rangeScale
        out_js['bkg_maxes'] = self.bkg_maxes.tolist()
        out_js['bkg_mins'] = self.bkg_mins.tolist()
        out_js['bkg_means'] = self.bkg_means.tolist()
        out_js['bkg_stds'] = self.bkg_stds.tolist()
        out_js['data_maxes'] = self.data_maxes.tolist()
        out_js['data_mins'] = self.data_mins.tolist()
        out_js['data_means'] = self.data_means.tolist()
        out_js['data_stds'] = self.data_stds.tolist()
        with open(f"{self.outDir}/info.json","w") as info_out:
            json.dump(out_js,info_out,indent=4)
        
        
        # normalize data for trainings
        bkg_train = self.scale(bkg_train,"bkg",forward=True)
        bkg_test = self.scale(bkg_test,"bkg",forward=True)
        data_train = self.scale(data_train,"data",forward=True)
        data_test = self.scale(data_test,"data",forward=True)
        
        # set up dictionaries of train/test data
        self.data_train = {self.varNames[i]:data_train[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        self.data_test = {self.varNames[i]:data_test[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        self.bkg_train = {self.varNames[i]:bkg_train[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        self.bkg_test = {self.varNames[i]:bkg_test[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        
        # variables to track training -- keep *raw* values in the 'corrected' collection b/c data and MC have different preprocessing scaling
        self.correctedBkg_train = {self.varNames[i]:self.scale(bkg_train[:,i],"bkg",forward=False,idx=i).reshape(-1,1) for i in range(len(self.varNames))}
        self.correctedBkg_test = {self.varNames[i]:self.scale(bkg_test[:,i],"bkg",forward=False,idx=i).reshape(-1,1) for i in range(len(self.varNames))}
        self.data_models = None
        self.bkg_models = None
        self.data_model_loc = None
        self.bkg_model_locs = None
        
    @property
    def correctedBkg(self,mode):
        if mode == "train":
            return np.concatenate([self.correctedBkg_train[v] for v in self.fitVars],axis=1)
        else:
            return np.concatenate([self.correctedBkg_test[v] for v in self.fitVars],axis=1)
        
    @property
    def data(self,mode):
        if mode == "train":
            return np.concatenate([self.data_train[v] for v in self.fitVars],axis=1)
        else:
            return np.concatenate([self.data_test[v] for v in self.fitVars],axis=1)
        
    def scale(self,inputs,mode,forward=True,idx=None):
        #if mode == "bkg":
        #    mins,maxes = self.bkg_mins,self.bkg_maxes
        #    means,stds = self.bkg_means,self.bkg_stds
        #elif mode == "data":
        #    mins,maxes = self.data_mins,self.data_maxes
        #    #means,stds = self.bkg_means,self.bkg_stds
        #    means,stds = self.data_means,self.data_stds
        #else:
        #    print("Can't scale, unrecognized mode!")
        #    return inputs
        mins = np.minimum(self.bkg_mins,self.data_mins)
        maxes = np.maximum(self.bkg_maxes,self.data_maxes)
        means = self.bkg_means
        stds = self.bkg_stds
        scale = np.where(np.abs(maxes-means)>np.abs(mins-means),np.abs(maxes-means),np.abs(mins-means))
        #scale = np.where(np.abs(maxes)>np.abs(mins),np.abs(maxes),np.abs(mins))
        # if scaling a single variable
        if idx is not None:
            mins,maxes,means,stds = mins[idx],maxes[idx],means[idx],stds[idx]
            scale = scale[idx]
        if forward:
            #inputs = inputs-means
            #inputs = self.rangeScale*(inputs/scale)
            
            inputs = 2*self.rangeScale*((inputs-mins)/(maxes-mins)-0.5)
            
            #inputs = (inputs-means)/stds
        else:
            #inputs = scale*inputs/self.rangeScale
            #inputs = inputs + means
            
            inputs = (maxes-mins)*(inputs/(2*self.rangeScale) + 0.5) + mins            
            
            #inputs = stds*inputs + means
        return inputs
            
    def new_flow(self,num_features,num_context,kwargs):
        return make_flow(num_features,num_context,kwargs)
    
    def get_flow(self,n_features,n_context,loc,kwargs):
        flow = self.new_flow(n_features,n_context,kwargs)
        flow.load_state_dict(torch.load(loc))
        flow.eval()
        return flow
    
    def train_flow(self,flow,loader,name,kwargs):
        flow = flow.to(device)
        if kwargs['wd']>0:
            optimizer = optim.Adam(flow.parameters(),lr=kwargs['learning_rate'],weight_decay=kwargs['wd'])
        else:
            optimizer = optim.Adam(flow.parameters(),lr=kwargs['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=kwargs['n_epoch'],eta_min=0.01*kwargs['learning_rate'],verbose=False)

        min_loss = 1e+8
        train_losses = []
        patience_count = 0
        saveName = f"{self.outDir}/{name}.pt"
        
        print("Training flow {0}".format(name))
    
        tbar = tqdm(range(int(kwargs['n_epoch'])))
        for i in tbar:
            if patience_count == kwargs['patience']:
                break
            epoch_losses = []
            x = next(loader)
            #for batch_idx, x in enumerate(loader):
            inputs,context = x
            optimizer.zero_grad()
            if torch.all(context==0):
                loss = -flow.log_prob(inputs=inputs)[0].mean()
            else:
                loss = -flow.log_prob(inputs=inputs,context=context)[0].mean()
            loss.backward()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(flow.state_dict(),saveName)
            train_losses.append(loss.item())
            if loss.item() == min_loss:
                patience_count = 0
                s = 'Loss: {0}, p = {1}'.format(loss.item(),patience_count)
                tbar.set_description(s)
            else:
                patience_count += 1
                s = 'Loss: {0}, p = {1}'.format(min_loss,patience_count)
                tbar.set_description(s)
            scheduler.step()
        
        flow.load_state_dict(torch.load(saveName))
        flow.eval()
        torch.cuda.empty_cache()
        flow = flow.to('cpu')

        return flow, train_losses
    
    def trainCurrentBkg(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,wd=0):
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'wd':wd}
        contextVars = self.controlVars
        contextIdx = self.control
        
        # train bkg flow
        bkg_train = torch.tensor(np.concatenate([self.bkg_train[v] for v in self.fitVars],axis=1),dtype=torch.float32,device=device)
        if len(contextVars) > 0:
            bkg_train_context = torch.tensor(self.scale(
                np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),
                "bkg",forward=True,idx=contextIdx),dtype=torch.float32,device=device)
        else:
            bkg_train_context = torch.zeros((bkg_train.shape[0],1),dtype=torch.float32,device=device)
        bkg_train_dataset = utils.TensorDataset(bkg_train,bkg_train_context)
        bkg_loader = InfiniteLoader(dataset=bkg_train_dataset,batch_size=bs,shuffle=True,
                                    generator=torch.Generator(device='cuda'))
        bkg_flow = self.new_flow(len(self.fitVars),len(contextVars),self.NF_kwargs)
        bkg_flowName = "bkgFlow"
        bkg_flow, bkg_trainLosses = self.train_flow(bkg_flow,bkg_loader,bkg_flowName,train_kwargs)
        
        plt.figure(figsize=(8,6))
        w = int(self.bkg_train[self.fitVars[0]].shape[0]/(5*bs))
        smooth = np.convolve(np.ones(w),bkg_trainLosses,mode='valid')/w
        xvals = np.linspace(0,len(bkg_trainLosses),len(smooth))
        plt.plot(xvals,smooth)
        #plt.plot(np.arange(len(bkg_trainLosses)),bkg_trainLosses)
        plt.title(bkg_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{self.outDir}/trainCurve_bkg.pdf")
        del bkg_train_context, bkg_train, bkg_train_dataset, bkg_loader, bkg_flow
        torch.cuda.empty_cache()
        
    def trainCurrentData(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,wd=0):
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'wd':wd}
        contextVars = self.controlVars
        contextIdx = self.control
        
        # train bkg flow
        data_train = torch.tensor(np.concatenate([self.data_train[v] for v in self.fitVars],axis=1),dtype=torch.float32,device=device)
        if len(contextVars) > 0:
            data_train_context = torch.tensor(self.scale(
                np.concatenate([self.data_train[n] for n in contextVars],axis=1),
                "data",forward=True,idx=contextIdx),dtype=torch.float32,device=device)
        else:
            data_train_context = torch.zeros((data_train.shape[0],1),dtype=torch.float32,device=device)
        data_train_dataset = utils.TensorDataset(data_train,data_train_context)
        data_loader = InfiniteLoader(dataset=data_train_dataset,batch_size=bs,shuffle=True,
                                    generator=torch.Generator(device='cuda'))
        data_flow = self.new_flow(len(self.fitVars),len(contextVars),self.NF_kwargs)
        data_flowName = "dataFlow"
        data_flow, data_trainLosses = self.train_flow(data_flow,data_loader,data_flowName,train_kwargs)
        
        plt.figure(figsize=(8,6))
        w = int(self.data_train[self.fitVars[0]].shape[0]/(5*bs))
        smooth = np.convolve(np.ones(w),data_trainLosses,mode='valid')/w
        xvals = np.linspace(0,len(data_trainLosses),len(smooth))
        plt.plot(xvals,smooth)
        #plt.plot(np.arange(len(data_trainLosses)),data_trainLosses)
        plt.title(data_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{self.outDir}/trainCurve_data.pdf")
        del data_train_context, data_train, data_train_dataset, data_loader, data_flow
        torch.cuda.empty_cache()
        
    def trainCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,wd=0):
        self.trainCurrentBkg(bs=bs,n_epoch=n_epoch,patience=patience,
                              learning_rate=learning_rate,wd=wd)
        self.trainCurrentData(bs=bs,n_epoch=n_epoch,patience=patience,
                              learning_rate=learning_rate,wd=wd)
        self.plotDensity(bkg=True,data=False)
        self.plotDensity(bkg=False,data=True)
        
    def plotDensity(self,bkg=True,data=True):
        contextVars = self.controlVars
        contextIdx = self.control
        fitIdx = self.fit

        if bkg:
            bkg_flow = self.get_flow(len(self.fitVars),len(self.controlVars),
                                     f"{self.outDir}/bkgFlow.pt",self.NF_kwargs).to(device)
            bkg_flow.eval()
            with torch.no_grad():
                bkg_test_var = np.concatenate([self.bkg_test[v] for v in self.fitVars],axis=1)
                nTest = bkg_test_var.shape[0]
                n_per = 20000
                split = np.array_split(np.arange(nTest),nTest//n_per + 1)
                samples = []
                if len(contextVars) > 0:
                    bkg_test_context = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                    for k in split:
                        samples.append(bkg_flow.sample(1,context=torch.tensor(bkg_test_context[k],dtype=torch.float32,
                                        device=device)).detach().cpu().numpy().reshape(bkg_test_context.shape[0],-1))
                        torch.cuda.empty_cache()
                    del bkg_test_context
                else:
                    for k in split:
                        samples.append(bkg_flow.sample(len(k)).detach().cpu().numpy())
            bkg_samples = np.concatenate(samples,axis=0)
            for i,v in enumerate(self.fitVars):
                plotName = f"{self.outDir}/bkg_fitDensity_{v}.pdf"
                self.plotPair(bkg_test_var[:,i],bkg_samples[:,i],'bkg',plotName,log=False,xlabel=v)
                plotName = f"{self.outDir}/bkg_fitDensity_{v}_log.pdf"
                self.plotPair(bkg_test_var[:,i],bkg_samples[:,i],'bkg',plotName,log=True,xlabel=v)
            del bkg_flow, bkg_samples, bkg_test_var
            torch.cuda.empty_cache()
        
        if data:
            data_flow = self.get_flow(len(self.fitVars),len(self.controlVars),
                                     f"{self.outDir}/dataFlow.pt",self.NF_kwargs).to(device)
            data_flow.eval()
            with torch.no_grad():
                data_test_var = np.concatenate([self.data_test[v] for v in self.fitVars],axis=1)
                nTest = data_test_var.shape[0]
                n_per = 20000
                split = np.array_split(np.arange(nTest),nTest//n_per + 1)
                samples = []
                if len(contextVars) > 0:
                    data_test_context = self.scale(np.concatenate([self.data_test[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
                    for k in split:
                        samples.append(data_flow.sample(1,context=torch.tensor(data_test_context[k],dtype=torch.float32,
                                       device=device)).detach().cpu().numpy().reshape(data_test_context.shape[0],-1))
                    del data_test_context
                else:
                    for k in split:
                        samples.append(data_flow.sample(len(k)).detach().cpu().numpy())
            data_samples = np.concatenate(samples,axis=0)
            for i,v in enumerate(self.fitVars):
                plotName = f"{self.outDir}/data_fitDensity_{v}.pdf"
                self.plotPair(data_test_var[:,i],data_samples[:,i],'data',plotName,log=False,xlabel=v)
                plotName = f"{self.outDir}/data_fitDensity_{v}_log.pdf"
                self.plotPair(data_test_var[:,i],data_samples[:,i],'data',plotName,log=True,xlabel=v)
            del data_flow, data_samples, data_test_var
            torch.cuda.empty_cache()
            
    def plotPair(self,ref,samples,sampName,saveName,log=False,xlabel=""):
        plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
        plt.subplot(211)
        h1,bins,_ = plt.hist(ref,bins=np.linspace(-self.rangeScale,self.rangeScale,50),
                             density=True,histtype='step',label=f'{sampName} test')
        h2,bins,_ = plt.hist(samples,bins=np.linspace(-self.rangeScale,self.rangeScale,50),
                            density=True,histtype='step',label=f'{sampName} samples')
        if log:
            plt.yscale('log')
        plt.legend()
        plt.subplot(212)
        w = bins[1]-bins[0]
        c = (bins[1:]+bins[:-1])/2
        h = np.divide(h2,h1,where=h1>0)
        plt.bar(x=c,height=h,width=w,align='center')
        plt.ylim([0,2])
        plt.yticks(np.arange(0,2.5,0.5))
        plt.grid(axis='y')
        plt.ylabel("Ratio")
        plt.xlabel(xlabel)
        plt.savefig(saveName)
        
    def correct(self,n_per=10000,pad=0.1):
        contextVars = self.controlVars
        contextIdx = self.control
        fitIdx = self.fit
        
        bkg_flow = self.get_flow(len(self.fitVars),len(contextVars),f"{self.outDir}/bkgFlow.pt",self.NF_kwargs).to(device)
        data_flow = self.get_flow(len(self.fitVars),len(contextVars),f"{self.outDir}/dataFlow.pt",self.NF_kwargs).to(device)
        
        # correct training set
        bkg_train_var = np.concatenate([self.bkg_train[v] for v in self.fitVars],axis=1)
        if len(contextVars) > 0:
            bkg_train_context_bkg = self.scale(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
            bkg_train_context_data = self.scale(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
        else:
            bkg_train_context_bkg = np.zeros((bkg_train_var.shape[0],1))
            bkg_train_context_data = np.zeros((bkg_train_var.shape[0],1))
        nTrain = bkg_train_context_bkg.shape[0]
        split = np.array_split(np.arange(nTrain),nTrain//n_per + 1)
        bkg_train_context_bkg = [bkg_train_context_bkg[k] for k in split]
        bkg_train_context_data = [bkg_train_context_data[k] for k in split]
        bkg_train_var = [bkg_train_var[k] for k in split]
        bkg_train_corr = []
        for i in tqdm(range(len(split))):
            inputs = torch.tensor(bkg_train_var[i],dtype=torch.float32,device=device)
            context_bkg = torch.tensor(bkg_train_context_bkg[i],dtype=torch.float32,device=device)
            context_data = torch.tensor(bkg_train_context_data[i],dtype=torch.float32,device=device)
            if torch.all(context_bkg==0):
                noise = bkg_flow.transform_to_noise(inputs=inputs)
                corrected = data_flow._transform.inverse(noise)[0]
            else:
                noise = bkg_flow.transform_to_noise(inputs=inputs,context=context_bkg)
                corrected = data_flow._transform.inverse(noise,context=context_data)[0]
            bkg_train_corr.append(corrected.detach().cpu().numpy())
            del inputs,context_bkg,context_data,noise,corrected
            torch.cuda.empty_cache()
        bkg_train_corr = self.scale(np.concatenate(bkg_train_corr,axis=0),"data",forward=False,idx=fitIdx)
        for i,v in enumerate(self.fitVars):
            self.correctedBkg_train[v] = np.copy(bkg_train_corr[:,i]).reshape(-1,1)
        del bkg_train_corr
        
         # correct test set
        bkg_test_var = np.concatenate([self.bkg_test[v] for v in self.fitVars],axis=1)
        if len(contextVars) > 0:
            bkg_test_context_bkg = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
            bkg_test_context_data = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
        else:
            bkg_test_context_bkg = np.zeros((bkg_test_var.shape[0],1))
            bkg_test_context_data = np.zeros((bkg_test_var.shape[0],1))
        nTest = bkg_test_context_bkg.shape[0]
        split = np.array_split(np.arange(nTest),nTest//n_per + 1)
        bkg_test_context_bkg = [bkg_test_context_bkg[k] for k in split]
        bkg_test_context_data = [bkg_test_context_data[k] for k in split]
        bkg_test_var = [bkg_test_var[k] for k in split]
        bkg_test_corr = []
        for i in tqdm(range(len(split))):
            inputs = torch.tensor(bkg_test_var[i],dtype=torch.float32,device=device)
            context_bkg = torch.tensor(bkg_test_context_bkg[i],dtype=torch.float32,device=device)
            context_data = torch.tensor(bkg_test_context_data[i],dtype=torch.float32,device=device)
            if torch.all(context_bkg==0):
                noise = bkg_flow.transform_to_noise(inputs=inputs)
                corrected = data_flow._transform.inverse(noise)[0]
            else:
                noise = bkg_flow.transform_to_noise(inputs=inputs,context=context_bkg)
                corrected = data_flow._transform.inverse(noise,context=context_data)[0]
            bkg_test_corr.append(corrected.detach().cpu().numpy())
            del inputs,context_bkg,context_data,noise,corrected
            torch.cuda.empty_cache()
        bkg_test_corr = self.scale(np.concatenate(bkg_test_corr,axis=0),"data",forward=False,idx=fitIdx)
        for i,v in enumerate(self.fitVars):
            self.correctedBkg_test[v] = np.copy(bkg_test_corr[:,i].reshape(-1,1))
        del bkg_test_corr
        
        del bkg_flow, data_flow
        torch.cuda.empty_cache()
        
        
    def correctFull(self,data,bkg,n_per=10000):
        bkg_corr = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        data = self.scale(data,"data",forward=True)
        bkg = self.scale(bkg,"bkg",forward=True)
        data = {self.varNames[i]:data[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        bkg = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        
        for j,v in enumerate(self.fitVars):
            currentVar = v
            contextVars = self.controlVars+self.fitVars[:j]
            contextIdx = [self.varNames.index(v) for v in contextVars]
            
            bkg_flow = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
            data_flow = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)

            # correct training set
            if len(contextVars) > 0:
                context_bkg = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                context_data = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
            else:
                context_bkg = np.zeros(bkg[currentVar].shape)
                context_data = np.zeros(bkg[currentVar].shape)
            inputs = bkg[currentVar]
            nEvts = inputs.shape[0]
            split = np.array_split(np.arange(nEvts),nEvts//n_per + 1)
            context_bkg = [context_bkg[k] for k in split]
            context_data = [context_data[k] for k in split]
            inputs = [inputs[k] for k in split]
            corr = []
            for i in tqdm(range(len(split))):
                inputs_i = torch.tensor(inputs[i],dtype=torch.float32,device=device)
                context_i_bkg = torch.tensor(context_bkg[i],dtype=torch.float32,device=device)
                context_i_data = torch.tensor(context_data[i],dtype=torch.float32,device=device)
                with torch.no_grad():
                    if torch.all(context_i_bkg==0):
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i)
                        corrected = data_flow._transform.inverse(noise)[0]
                    else:
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i,context=context_i_bkg)
                        corrected = data_flow._transform.inverse(noise,context=context_i_data)[0]
                corr.append(corrected.detach().cpu().numpy())
                del inputs_i,context_i_bkg,context_i_data,noise,corrected
                torch.cuda.empty_cache()
            corr = self.scale(np.concatenate(corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
            bkg_corr[currentVar] = np.copy(corr)
            del corr
        
        for i,v in enumerate(self.varNames):
            data[v] = self.scale(data[v],"data",forward=False,idx=i)
            #bkg_corr[v] = self.scale(bkg_corr[v],"data",forward=False,idx=i)
        return data,bkg_corr,self.varNames
    
    def plotTriplet(self,ref,corr,uncorr,bins,xlabel,title,saveName,log=False):
        plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
        plt.subplot(211)
        h1,bins,_ = plt.hist(ref,bins=bins,histtype='step',label="Data",density=True,color='gray',fill=True,alpha=0.5)
        h2,bins,_ = plt.hist(corr,bins=bins,histtype='step',label="Corr Bkg",density=True,color='C0',linewidth=2)
        h3,bins,_ = plt.hist(uncorr,bins=bins,histtype='step',label='Uncorr Bkg',density=True,color='red',linewidth=2)
        plt.title(title)
        if log:
            plt.yscale('log')
        plt.legend(loc='best')
        plt.subplot(212)
        w = bins[1]-bins[0]
        c = (bins[1:]+bins[:-1])/2
        h = np.divide(h2,h1,where=h1>0)
        plt.bar(x=c,height=h,width=w,align='center',color="C0")
        h = np.divide(h3,h1,where=h1>0)
        plt.step(x=bins[:-1],y=h,where='post',color="red",linewidth=2)
        plt.ylim([0,2])
        plt.yticks(np.arange(0,2.5,0.5))
        plt.grid(axis='y')
        plt.ylabel("Ratio")
        plt.xlabel(xlabel)
        plt.savefig(saveName)
    
    def plotVar(self,var,bins):
        idx = self.varNames.index(var)
        data_train = self.scale(self.data_train[var],"data",forward=False,idx=idx)
        data_test = self.scale(self.data_test[var],"data",forward=False,idx=idx)
        bkg_train = self.scale(self.bkg_train[var],"bkg",forward=False,idx=idx)
        bkg_test = self.scale(self.bkg_test[var],"bkg",forward=False,idx=idx)
        #corrBkg_train = self.scale(self.correctedBkg_train[var],"data",forward=False,idx=idx)
        #corrBkg_test = self.scale(self.correctedBkg_test[var],"data",forward=False,idx=idx)
        corrBkg_train = self.correctedBkg_train[var]
        corrBkg_test = self.correctedBkg_test[var]
        
        saveName = f"{self.outDir}/{var}_trainSet_beforeAfter.pdf"
        self.plotTriplet(data_train,corrBkg_train,bkg_train,bins,self.varDict[var],"Train Set",saveName)
        saveName = f"{self.outDir}/{var}_trainSet_beforeAfter_log.pdf"
        self.plotTriplet(data_train,corrBkg_train,bkg_train,bins,self.varDict[var],"Train Set",saveName,log=True)
        
        saveName = f"{self.outDir}/{var}_testSet_beforeAfter.pdf"
        self.plotTriplet(data_test,corrBkg_test,bkg_test,bins,self.varDict[var],"Test Set",saveName)
        saveName = f"{self.outDir}/{var}_testSet_beforeAfter_log.pdf"
        self.plotTriplet(data_test,corrBkg_test,bkg_test,bins,self.varDict[var],"Test Set",saveName,log=True)
        
    def plotAll(self,bins):
        for i,v in enumerate(self.controlVars+self.fitVars):
            b = bins[i] if type(bins)==list else bins
            self.plotVar(v,b)
        
    def run(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,bins=50,wd=0):
        print("RUNNING TRAININGS")
        self.trainCurrent(bs=bs,n_epoch=n_epoch,patience=patience,learning_rate=learning_rate,wd=wd)
        print("CORRECTING BKG")
        self.correct(n_per=10000)
        print("PLOTTING CORRECTIONS")
        self.plotAll(bins)
        
class chainedNFCorrector:
    def __init__(self,baseDir):
        self.baseDir = baseDir
        self.dirs = [f"{baseDir}/{d}" for d in os.listdir(baseDir) if os.path.isdir(f"{baseDir}/{d}") and 'step' in d]
        self.dirs.sort(key=lambda x: int(re.search('step(\d+)',x).group(1)))
        self.steps = [f.split("/")[-1] for f in self.dirs]
        with open(f"{self.baseDir}/info.json") as f:
            self.info = json.load(f)
        for k in self.info.keys():
            setattr(self,k,self.info[k])
        self.bkg_means = np.array(self.bkg_means)
        self.bkg_stds = np.array(self.bkg_stds)
        self.bkg_mins = np.array(self.bkg_mins)
        self.bkg_maxes = np.array(self.bkg_maxes)
        self.data_means = np.array(self.data_means)
        self.data_stds = np.array(self.data_stds)
        self.data_mins = np.array(self.data_mins)
        self.data_maxes = np.array(self.data_maxes)
            
    def scale(self,inputs,mode,forward=True,idx=None):
        #if mode == "bkg":
        #    mins,maxes = self.bkg_mins,self.bkg_maxes
        #    means,stds = self.bkg_means,self.bkg_stds
        #elif mode == "data":
        #    mins,maxes = self.data_mins,self.data_maxes
        #    #means,stds = self.bkg_means,self.bkg_stds
        #    means,stds = self.data_means,self.data_stds
        #else:
        #    print("Can't scale, unrecognized mode!")
        #    return inputs
        mins = np.minimum(self.bkg_mins,self.data_mins)
        maxes = np.maximum(self.bkg_maxes,self.data_maxes)
        means = self.bkg_means
        stds = self.bkg_stds
        scale = np.where(np.abs(maxes-means)>np.abs(mins-means),np.abs(maxes-means),np.abs(mins-means))
        #scale = np.where(np.abs(maxes)>np.abs(mins),np.abs(maxes),np.abs(mins))
        # if scaling a single variable
        if idx is not None:
            mins,maxes,means,stds = mins[idx],maxes[idx],means[idx],stds[idx]
            scale = scale[idx]
        if forward:
            #inputs = inputs-means
            #inputs = self.rangeScale*(inputs/scale)
            
            inputs = 2*self.rangeScale*((inputs-mins)/(maxes-mins)-0.5)
        else:
            #inputs = scale*inputs/self.rangeScale
            #inputs = inputs + means
            
            inputs = (maxes-mins)*(inputs/(2*self.rangeScale) + 0.5) + mins            
        return inputs
    
    def new_flow(self,num_features,num_context,kwargs):
        return make_flow(num_features,num_context,kwargs)
    
    def get_flow(self,n_features,n_context,loc,kwargs):
        flow = self.new_flow(n_features,n_context,kwargs)
        flow.load_state_dict(torch.load(loc))
        flow.eval()
        return flow
        
    def correctFull(self,bkg,n_per=10000):
        bkg_corr = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        bkg = self.scale(bkg,"bkg",forward=True)
        bkg = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        
        for j,v in enumerate(self.fitVars):
            currentVar = v
            contextVars = self.controlVars+self.fitVars[:j]
            contextIdx = [self.varNames.index(v) for v in contextVars]
            print(contextIdx)
            
            flowDir = f"{self.baseDir}/{self.dirs[j]}/"
            with open(f"{self.dirs[j]}/dataFlowConfig_{self.steps[j]}.json") as fj:
                dataFlowJson = json.load(fj)
            dataFlowLoc = f"{self.dirs[j]}/dataFlow_{self.steps[j]}.pt"
            with open(f"{self.dirs[j]}/bkgFlowConfig_{self.steps[j]}.json") as fj:
                bkgFlowJson = json.load(fj)
            bkgFlowLoc = f"{self.dirs[j]}/bkgFlow_{self.steps[j]}.pt"
            
            bkg_flow = self.get_flow(1,len(bkgFlowJson['contextVars']),bkgFlowLoc,bkgFlowJson['NF_kwargs']).to(device)
            data_flow = self.get_flow(1,len(dataFlowJson['contextVars']),dataFlowLoc,dataFlowJson['NF_kwargs']).to(device)

            # correct training set
            if len(contextVars) > 0:
                context_bkg = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                context_data = self.scale(np.concatenate([bkg_corr[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
            else:
                context_bkg = np.zeros(bkg[currentVar].shape)
                context_data = np.zeros(bkg[currentVar].shape)
            inputs = bkg[currentVar]
            nEvts = inputs.shape[0]
            split = np.array_split(np.arange(nEvts),nEvts//n_per + 1)
            context_bkg = [context_bkg[k] for k in split]
            context_data = [context_data[k] for k in split]
            inputs = [inputs[k] for k in split]
            corr = []
            for i in tqdm(range(len(split))):
                inputs_i = torch.tensor(inputs[i],dtype=torch.float32,device=device)
                context_i_bkg = torch.tensor(context_bkg[i],dtype=torch.float32,device=device)
                context_i_data = torch.tensor(context_data[i],dtype=torch.float32,device=device)
                with torch.no_grad():
                    if torch.all(context_i_bkg==0):
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i)
                        corrected = data_flow._transform.inverse(noise)[0]
                    else:
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i,context=context_i_bkg)
                        corrected = data_flow._transform.inverse(noise,context=context_i_data)[0]
                corr.append(corrected.detach().cpu().numpy())
                del inputs_i,context_i_bkg,context_i_data,noise,corrected
                torch.cuda.empty_cache()
            corr = self.scale(np.concatenate(corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
            bkg_corr[currentVar] = np.copy(corr)
            del corr
        
        return bkg_corr
    
def plotTriplet(ref,corr,uncorr,bins,xlabel,title,saveName,save=False,log=False):
    plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
    plt.subplot(211)
    h1,bins,_ = plt.hist(ref,bins=bins,histtype='step',label="Data",density=True,color='gray',fill=True,alpha=0.5)
    h2,bins,_ = plt.hist(corr,bins=bins,histtype='step',label="Corr Bkg",density=True,color='C0',linewidth=2)
    h3,bins,_ = plt.hist(uncorr,bins=bins,histtype='step',label='Uncorr Bkg',density=True,color='red',linewidth=2)
    plt.title(title)
    if log:
        plt.yscale('log')
    plt.legend(loc='best')
    plt.subplot(212)
    w = bins[1]-bins[0]
    c = (bins[1:]+bins[:-1])/2
    h = np.divide(h2,h1,where=h1>0)
    plt.bar(x=c,height=h,width=w,align='center',color="C0")
    h = np.divide(h3,h1,where=h1>0)
    plt.step(x=bins[:-1],y=h,where='post',color="red",linewidth=2)
    plt.ylim([0,2])
    plt.yticks(np.arange(0,2.5,0.5))
    plt.grid(axis='y')
    plt.ylabel("Ratio")
    plt.xlabel(xlabel)
    if save:
        plt.savefig(saveName)

def plotPair(ref,samples,bins=50,saveName="",save=False,log=False):
    plt.subplots(figsize=(8,6),nrows=2,ncols=1,gridspec_kw={'height_ratios':[3,1]},sharex=True)
    plt.subplot(211)
    h1,bins,_ = plt.hist(ref,bins=bins,
                         density=True,histtype='step',label=f'reference')
    h2,bins,_ = plt.hist(samples,bins=bins,
                        density=True,histtype='step',label=f'samples')
    if log:
        plt.yscale('log')
    plt.legend()
    plt.subplot(212)
    w = bins[1]-bins[0]
    c = (bins[1:]+bins[:-1])/2
    h = np.divide(h2,h1,where=h1>0)
    plt.bar(x=c,height=h,width=w,align='center')
    plt.ylim([0,2])
    plt.yticks(np.arange(0,2.5,0.5))
    plt.grid(axis='y')
    plt.ylabel("Ratio")
    if save:
        plt.savefig(saveName)
        

def make_flow(num_features,num_context,kwargs,perm=True):
    flow_type = kwargs['flow_type']
    #if num_context == 0:
    #    base_dist = StandardNormal(shape=[num_features])
    #else:
    #    encoder = NeuralNet(num_context, 20, 2, 3)
    #    base_dist = ConditionalDiagonalNormal(shape=[num_features],context_encoder=encoder)
    base_dist = StandardNormal(shape=[num_features])
    transforms = []
    if num_context == 0:
        num_context = None
    for i in range(kwargs['num_layers']):
        if flow_type == 'MAF':
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,
                                                                    hidden_features=kwargs['hidden_features'], 
                                                                    num_blocks=kwargs['num_blocks_per_layer']))
        elif flow_type == 'NSQUAD':
            transforms.append(MaskedPiecewiseQuadraticAutoregressiveTransform(features=num_features,
                                                                              context_features=num_context,
                                                                            hidden_features=num_features,
                                                                            num_bins=kwargs['num_bins'],
                                                                            num_blocks=kwargs['num_blocks_per_layer'],
                                                                            tail_bound=kwargs['tail_bound'], 
                                                                            tails='linear'))
        elif flow_type == 'NSRATQUAD':
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                      context_features=num_context,
                                                                                hidden_features=kwargs['hidden_features'],
                                                                                num_bins=kwargs['num_bins'],
                                                                                num_blocks=kwargs['num_blocks_per_layer'],
                                                                                tail_bound=kwargs['tail_bound'], 
                                                                                tails=kwargs['tails'],
                                                                                dropout_probability=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                                                                use_batch_norm=kwargs['batchnorm'] if 'batchnorm' in kwargs.keys() else False))
        elif flow_type == "IRQS":
            transforms.append(IndependentRQS(features=num_features,
                                             context=num_context,
                                             hidden=kwargs['hidden_features'],
                                             num_hidden=kwargs['num_blocks_per_layer'],
                                                num_bins=kwargs['num_bins'],
                                                tails=kwargs['tails'],
                                                tail_bound=kwargs['tail_bound'],
                                            dropout=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                            residual=kwargs['residual'] if 'residual' in kwargs.keys() else False))
        elif flow_type == "ARQS":
            transforms.append(AutoregressiveRQS(features=num_features,
                                             context=num_context,
                                             hidden=kwargs['hidden_features'],
                                             num_hidden=kwargs['num_blocks_per_layer'],
                                                num_bins=kwargs['num_bins'],
                                                tails=kwargs['tails'],
                                                tail_bound=kwargs['tail_bound'],
                                            dropout=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0,
                                            residual=kwargs['residual'] if 'residual' in kwargs.keys() else False))
            """if num_context is None:
                transforms.append(IndependentRQS(features=num_features,
                                                num_bins=kwargs['num_bins'],
                                                tails=kwargs['tails'],
                                                tail_bound=kwargs['tail_bound']))
            else:
                transforms.append(ConditionalRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                      context_features=num_context,
                                                                                hidden_features=kwargs['hidden_features'],
                                                                                num_bins=kwargs['num_bins'],
                                                                                num_blocks=kwargs['num_blocks_per_layer'],
                                                                                tail_bound=kwargs['tail_bound'], 
                                                                                tails=kwargs['tails']))"""
        #if i < kwargs['num_layers'] - 1 and perm:
        #    transforms.append(ReversePermutation(features=num_features))

    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow