#from imports import *
import helper_functions as hf
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os

sys.path.append("/home/sambt/CASE_final/CASE_QUAK_final/")
from nflows.flows.base import Flow
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseQuadraticAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform, ConditionalRationalQuadraticAutoregressiveTransform, IndependentRQS
from nflows.transforms.permutations import ReversePermutation

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.utils.data as utils

from tqdm import tqdm
import os

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
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(NeuralNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# numba for fast binning
from numba import vectorize, guvectorize, float64, int64, float32, int32, jit
@guvectorize([(float32[:,:],float32[:,:],int32[:,:])], '(n,m),(n,k)->(n,m)')
def nbinAll(XX,YY,res):
    N,M = XX.shape
    K = YY.shape[1]
    for i in range(N):
        X = XX[i]
        Y = YY[i]
        for j in range(M):
            x = X[j]
            if x < Y[0]:
                ind = 0
            elif x>=Y[-1]:
                ind = len(Y)
            else:
                for l in range(K-1):
                    if x >= Y[l] and x < Y[l+1]:
                        ind = l+1
                        break
            res[i,j] = ind

# numba for applying corrections
@guvectorize([(float32[:],float32[:],float32[:,:],float32[:,:],float32[:])], '(n),(k),(n,k),(n,k)->(n)')
def correctAll(Y,x,qbkg,qdata,Yout):
    N,M = qbkg.shape
    bad = 0
    bad_bkg = 0
    bad_data = 0
    underflow = 0
    overflow = 0
    good = 0
    for i in range(N):
        qb = qbkg[i]
        qd = qdata[i]
        y = Y[i]
        if y < x[0] or y > x[-1]:
            Yout[i] = y
        else:
            for j in range(M-1):
                if y >= x[j] and y < x[j+1]:
                    m = (qb[j+1]-qb[j])/(x[j+1]-x[j])
                    p = qb[j] + m*(y-x[j])
                    for k in range(M-1):
                        if p >= qd[k] and p < qd[k+1]:
                            ycorr = x[k] + (p-qd[k])*(x[k+1]-x[k])/(qd[k+1]-qd[k])
                    break
        Yout[i] = ycorr

class chainedNFTrainer:
    def __init__(self,projName,bkg_train,bkg_test,data_train,data_test,varNames,name="TEST",control=[],NF_kwargs={},outDir="NF_models_QR_general_simultaneous/",rangeScale=3):
        assert len(varNames) == bkg_train.shape[1] and len(varNames) == data_train.shape[1]
        self.varNames = varNames
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
        self.bkg_means = bkg_all.mean(axis=0)
        self.bkg_stds = bkg_all.std(axis=0)
        data_all = np.concatenate((data_train,data_test),axis=0)
        self.data_maxes = data_all.max(axis=0)
        self.data_mins = data_all.min(axis=0)
        self.data_means = data_all.mean(axis=0)
        self.data_stds = data_all.std(axis=0)
        
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
        
        # variables to track training
        self.correctedBkg_train = {self.varNames[i]:self.scale(bkg_train[:,i],"bkg",forward=False,idx=i).reshape(-1,1) for i in range(len(self.varNames))}
        self.correctedBkg_test = {self.varNames[i]:self.scale(bkg_test[:,i],"bkg",forward=False,idx=i).reshape(-1,1) for i in range(len(self.varNames))}
        self.data_models = [None for _ in range(len(self.fitVars))]
        self.bkg_models = [None for _ in range(len(self.fitVars))]
        self.data_model_locs = [None for _ in range(len(self.fitVars))]
        self.bkg_model_locs = [None for _ in range(len(self.fitVars))]
        self.current = 0 # index of variable currently being corrected (index in self.varOrder)
        self.bkg_trainings = {n:{} for n in self.fitVars}
        self.data_trainings = {n:{} for n in self.fitVars}
        
    @property
    def currentDir(self):
        currDir = f"{self.outDir}/step{self.current}_{self.fitVars[self.current]}/"
        if not os.path.isdir(currDir):
            os.makedirs(currDir)
        return currDir
    
    def scale(self,inputs,mode,forward=True,idx=None):
        #if mode == "bkg":
        #    mins,maxes,means,stds = self.bkg_mins,self.bkg_maxes,self.bkg_means,self.bkg_stds
        #elif mode == "data":
        #    mins,maxes,means,stds = self.data_mins,self.data_maxes,self.data_means,self.data_stds
        #else:
        #    print("Can't scale, unrecognized mode!")
        #    return inputs
        mins,maxes,means,stds = np.minimum(self.bkg_mins,self.data_mins),np.maximum(self.bkg_maxes,self.data_maxes),self.bkg_means,self.bkg_stds
        scale = np.where(np.abs(maxes-means)>np.abs(mins-means),np.abs(maxes-means),np.abs(mins-means))
        # if scaling a single variable
        if idx is not None:
            mins,maxes,means,stds = mins[idx],maxes[idx],means[idx],stds[idx]
            scale = scale[idx]
        if forward:
            inputs = inputs-means
            inputs = self.rangeScale*(inputs/scale)
        else:
            inputs = scale*inputs/self.rangeScale
            inputs += means
        return inputs
            
    def new_flow(self,num_features,num_context,kwargs):
        return make_flow(num_features,num_context,kwargs)
    
    def get_flow(self,n_features,n_context,loc,kwargs):
        transforms = self.new_flow(n_features,n_context,kwargs)
        #flow.load_state_dict(torch.load(loc))
        #flow.eval()
        return transforms
    
    def train_flow(self,bkg_flow,bkg_name,data_flow,data_name,loader,kwargs):
        bkg_flow = bkg_flow.to(device)
        data_flow = data_flow.to(device)
        optimizer = optim.Adam(list(bkg_flow.parameters())+list(data_flow.parameters()),lr=kwargs['learning_rate'],weight_decay=kwargs['wd'])
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=kwargs['n_epoch'],eta_min=0,verbose=False)

        min_bkg_loss = 1e+8
        min_data_loss = 1e+8
        bkg_train_losses = []
        data_train_losses = []
        patience_count = 0
        saveName_bkg = f"{self.currentDir}/{bkg_name}.pt"
        saveName_data = f"{self.currentDir}/{data_name}.pt"
        
        print("Training flows {0}, {1}".format(bkg_name,data_name))
    
        tbar = tqdm(range(int(kwargs['n_epoch'])))
        bkg_avg = 0
        data_avg = 0
        for i in tbar:
            if patience_count == kwargs['patience']:
                break
            epoch_losses = []
            x = next(loader)
            inputs_bkg,context_bkg,inputs_data,context_data = x
            optimizer.zero_grad()
            if torch.all(context_bkg==0):
                loss_bkg = -bkg_flow.log_prob(inputs=inputs_bkg)[0].mean() 
                loss_data = -data_flow.log_prob(inputs=inputs_data)[0].mean()
                loss = loss_bkg+loss_data
            else:
                loss_bkg = -bkg_flow.log_prob(inputs=inputs_bkg,context=context_bkg)[0].mean() 
                loss_data = -data_flow.log_prob(inputs=inputs_data,context=context_data)[0].mean()
                loss = loss_bkg+loss_data
            loss.backward()
            optimizer.step()
            if loss_bkg.item() < min_bkg_loss:
                min_bkg_loss = loss_bkg.item()
                torch.save(bkg_flow.state_dict(),saveName_bkg)
            if loss_data.item() < min_data_loss:
                min_data_loss = loss_data.item()
                torch.save(data_flow.state_dict(),saveName_data)
            bkg_train_losses.append(loss_bkg.item())
            data_train_losses.append(loss_data.item())
            if loss_bkg.item() == min_bkg_loss or loss_data.item() == min_data_loss:
                patience_count = 0
            else:
                patience_count += 1
            if i < 100:
                bkg_avg += loss_bkg.item()
                data_avg += loss_data.item()
            else:
                bkg_avg += (loss_bkg.item()-bkg_train_losses[i-100])
                data_avg += (loss_data.item()-data_train_losses[i-100])
            s = 'Bkg: {0:.4f}, Data: {1:.4f}, p ={2}'.format(bkg_avg/100,data_avg/100,patience_count)
            tbar.set_description(s)
            #scheduler.step()
        
        bkg_flow.load_state_dict(torch.load(saveName_bkg))
        bkg_flow.eval()
        data_flow.load_state_dict(torch.load(saveName_data))
        data_flow.eval()
        torch.cuda.empty_cache()
        bkg_flow = bkg_flow.to('cpu')
        data_flow = data_flow.to('cpu')

        return bkg_flow, saveName_bkg, bkg_train_losses, data_flow, saveName_data, data_train_losses
        
    def trainCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,wd=0):    
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'wd':wd}
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        contextIdx = [self.varNames.index(v) for v in contextVars]
        
        # set up flows
        if len(contextVars) > 0:
            bkg_train_context = torch.tensor(self.scale(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx),dtype=torch.float32,device=device)
            data_train_context = torch.tensor(np.concatenate([self.data_train[n] for n in contextVars],axis=1),dtype=torch.float32,device=device)
        else:
            bkg_train_context = torch.zeros(self.bkg_train[currentVar].shape,dtype=torch.float32,device=device)
            data_train_context = torch.zeros(self.data_train[currentVar].shape,dtype=torch.float32,device=device)
            
        bkg_train_var = torch.tensor(self.bkg_train[currentVar],dtype=torch.float32,device=device)
        data_train_var = torch.tensor(self.data_train[currentVar],dtype=torch.float32,device=device)
        
        train_dataset = utils.TensorDataset(bkg_train_var,bkg_train_context,data_train_var,data_train_context)
        
        loader = InfiniteLoader(dataset=train_dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
        
        bkg_transforms = self.new_flow(1,len(contextVars),self.NF_kwargs)
        data_transforms = self.new_flow(1,len(contextVars),self.NF_kwargs)
        
        if len(contextVars) == 0:
            base_dist_bkg = StandardNormal(shape=[1])
            base_dist_data = StandardNormal(shape=[1])
            bkg_flow = Flow(bkg_transforms,base_dist_bkg)
            data_flow = Flow(data_transforms,base_dist_data)
        else:
            encoder = NeuralNet(len(contextVars), 20, 2, 3)
            base_dist = ConditionalDiagonalNormal(shape=[1],context_encoder=encoder)
            # bkg and data share conditional base distribution
            bkg_flow = Flow(bkg_transforms,base_dist)
            data_flow = Flow(data_transforms,base_dist)
                
        bkg_flowName = "bkgFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        self.bkg_trainings[currentVar]['flowName'] = bkg_flowName
        self.bkg_trainings[currentVar]['contextVars'] = contextVars
        self.bkg_trainings[currentVar]['NF_kwargs'] = self.NF_kwargs
        self.bkg_trainings[currentVar]['train_kwargs'] = train_kwargs
        
        data_flowName = "dataFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        self.data_trainings[currentVar]['flowName'] = data_flowName
        self.data_trainings[currentVar]['contextVars'] = contextVars
        self.data_trainings[currentVar]['NF_kwargs'] = self.NF_kwargs
        self.data_trainings[currentVar]['train_kwargs'] = train_kwargs
        
        # train flows
        bkg_flow, bkg_flowLoc, bkg_trainLosses, data_flow, data_flowLoc, data_trainLosses = self.train_flow(bkg_flow,bkg_flowName,data_flow,data_flowName,loader,train_kwargs)
        
        self.bkg_trainings[currentVar]['losses'] = bkg_trainLosses
        self.bkg_trainings[currentVar]['flowLoc'] = bkg_flowLoc
        self.data_trainings[currentVar]['losses'] = data_trainLosses
        self.data_trainings[currentVar]['flowLoc'] = data_flowLoc
        
        bkg_flowConfig = "bkgFlowConfig_step{0}_{1}".format(self.current,self.fitVars[self.current])
        with open(f"{self.currentDir}/{bkg_flowConfig}.json","w") as cfg_out:
            json.dump(self.bkg_trainings[currentVar],cfg_out,indent=4)
            
        data_flowConfig = "dataFlowConfig_step{0}_{1}".format(self.current,self.fitVars[self.current])
        with open(f"{self.currentDir}/{data_flowConfig}.json","w") as cfg_out:
            json.dump(self.data_trainings[currentVar],cfg_out,indent=4)
        
        del train_dataset, loader
        
        plt.figure(figsize=(14,5))
        plt.subplot(121)
        w = int(self.bkg_train[currentVar].shape[0]/(5*bs))
        smooth = np.convolve(np.ones(w),bkg_trainLosses,mode='valid')/w
        xvals = np.linspace(0,len(bkg_trainLosses),len(smooth))
        #plt.plot(np.arange(len(bkg_trainLosses)),bkg_trainLosses)
        plt.plot(xvals,smooth)
        plt.title(bkg_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Bkg Train Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        torch.cuda.empty_cache()
        plt.subplot(122)
        w = int(self.data_train[currentVar].shape[0]/(5*bs))
        smooth = np.convolve(np.ones(w),data_trainLosses,mode='valid')/w
        xvals = np.linspace(0,len(data_trainLosses),len(smooth))
        #plt.plot(np.arange(len(data_trainLosses)),data_trainLosses)
        plt.plot(xvals,smooth)
        plt.title(data_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Data Train Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{self.currentDir}/trainCurves.pdf")
        del data_train_context, data_train_var, data_flow
        torch.cuda.empty_cache()
        
        self.plotDensity(bkg=True,data=False)
        self.plotDensity(bkg=False,data=True)
        
    def plotDensity(self,bkg=True,data=True):
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        contextIdx = [self.varNames.index(v) for v in contextVars]
        
        if bkg:
            bkg_transforms = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
            if len(self.bkg_trainings[currentVar]['contextVars']) == 0:
                base_dist = StandardNormal(shape=[1])
                bkg_flow = Flow(bkg_transforms,base_dist)
                bkg_flow.load_state_dict(torch.load(self.bkg_trainings[currentVar]['flowLoc']))
                bkg_flow.eval()    
            else:
                encoder = NeuralNet(len(self.bkg_trainings[currentVar]['contextVars']), 20, 2, 3)
                base_dist = ConditionalDiagonalNormal(shape=[1],context_encoder=encoder)
                bkg_flow = Flow(bkg_transforms,base_dist)
                bkg_flow.load_state_dict(torch.load(self.bkg_trainings[currentVar]['flowLoc']))
                bkg_flow.eval()  
            bkg_test_var = self.bkg_test[currentVar]
            if len(contextVars) > 0:
                bkg_test_context = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
                bkg_samples = bkg_flow.sample(1,context=torch.tensor(bkg_test_context,dtype=torch.float32,device=device)).detach().cpu().numpy().reshape(bkg_test_context.shape[0],-1)
                del bkg_test_context
            else:
                bkg_samples = bkg_flow.sample(bkg_test_var.shape[0]).detach().cpu().numpy()
            plotName = f"{self.currentDir}/bkg_fitDensity_step{self.current}_{self.fitVars[self.current]}.pdf"
            self.plotPair(bkg_test_var[:,0],bkg_samples[:,0],'bkg',plotName,log=False)
            del bkg_flow, bkg_samples, bkg_test_var
            torch.cuda.empty_cache()
        
        if data:
            data_transforms = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)
            if len(self.data_trainings[currentVar]['contextVars']) == 0:
                base_dist = StandardNormal(shape=[1])
                data_flow = Flow(data_transforms,base_dist)
                data_flow.load_state_dict(torch.load(self.data_trainings[currentVar]['flowLoc']))
                data_flow.eval()    
            else:
                encoder = NeuralNet(len(self.data_trainings[currentVar]['contextVars']), 20, 2, 3)
                base_dist = ConditionalDiagonalNormal(shape=[1],context_encoder=encoder)
                data_flow = Flow(data_transforms,base_dist)
                data_flow.load_state_dict(torch.load(self.data_trainings[currentVar]['flowLoc']))
                data_flow.eval()  
            data_test_var = self.data_test[currentVar]
            if len(contextVars) > 0:
                data_test_context = np.concatenate([self.data_test[n] for n in contextVars],axis=1)
                data_samples = data_flow.sample(1,context=torch.tensor(data_test_context,dtype=torch.float32,device=device)).detach().cpu().numpy().reshape(data_test_context.shape[0],-1)
                del data_test_context
            else:
                data_samples = data_flow.sample(data_test_var.shape[0]).detach().cpu().numpy()
            plotName = f"{self.currentDir}/data_fitDensity_step{self.current}_{self.fitVars[self.current]}.pdf"
            self.plotPair(data_test_var[:,0],data_samples[:,0],'data',plotName,log=False)
            del data_flow, data_samples, data_test_var
            torch.cuda.empty_cache()
        
    def plotPair(ref,samples,sampName,saveName,log=False):
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
        plt.ylim([0.8,1.2])
        plt.yticks(np.arange(0.8,1.3,0.1))
        plt.grid(axis='y')
        plt.ylabel("Ratio")
        plt.xlabel(self.fitVars[self.current])
        plt.savefig(saveName)
    
    def correctCurrent(self,n_per=10000,pad=0.1):
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        contextIdx = [self.varNames.index(v) for v in contextVars]
        
        bkg_transforms = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
        data_transforms = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)
        if len(contextVars) == 0:
            base_dist_bkg = StandardNormal(shape=[1])
            base_dist_data = StandardNormal(shape=[1])
            bkg_flow = Flow(bkg_transforms,base_dist_bkg)
            bkg_flow.load_state_dict(torch.load(self.bkg_trainings[currentVar]['flowLoc']))
            bkg_flow.eval()
            data_flow = Flow(data_transforms,base_dist_data)
            data_flow.load_state_dict(torch.load(self.data_trainings[currentVar]['flowLoc']))
            data_flow.eval()
        else:
            encoder = NeuralNet(len(contextVars), 20, 2, 3)
            base_dist = ConditionalDiagonalNormal(shape=[1],context_encoder=encoder)
            # bkg and data share conditional base distribution
            bkg_flow = Flow(bkg_transforms,base_dist)
            bkg_flow.load_state_dict(torch.load(self.bkg_trainings[currentVar]['flowLoc']))
            bkg_flow.eval()
            data_flow = Flow(data_transforms,base_dist)
            data_flow.load_state_dict(torch.load(self.data_trainings[currentVar]['flowLoc']))
            data_flow.eval()
        
        # correct training set
        if len(contextVars) > 0:
            bkg_train_context_bkg = self.scale(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
            bkg_train_context_data = self.scale(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
        else:
            bkg_train_context_bkg = np.zeros(self.bkg_train[currentVar].shape)
            bkg_train_context_data = np.zeros(self.bkg_train[currentVar].shape)
        bkg_train_var = self.bkg_train[currentVar]
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
        bkg_train_corr = self.scale(np.concatenate(bkg_train_corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
        self.correctedBkg_train[currentVar] = np.copy(bkg_train_corr)
        del bkg_train_corr
        
        # correct test set
        if len(contextVars) > 0:
            bkg_test_context_bkg = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"bkg",forward=True,idx=contextIdx)
            bkg_test_context_data = self.scale(np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1),"data",forward=True,idx=contextIdx)
        else:
            bkg_test_context_bkg = np.zeros(self.bkg_test[currentVar].shape)
            bkg_test_context_data = np.zeros(self.bkg_test[currentVar].shape)
        bkg_test_var = self.bkg_test[currentVar]
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
        bkg_test_corr = self.scale(np.concatenate(bkg_test_corr,axis=0),"data",forward=False,idx=self.varNames.index(currentVar))
        self.correctedBkg_test[currentVar] = np.copy(bkg_test_corr)
        del bkg_test_corr
        
        del bkg_flow, data_flow
        torch.cuda.empty_cache()
        
    def correctFull(self,data,bkg,n_per=10000):
        data = self.scale(data,"data",forward=True)
        bkg = self.scale(bkg,"bkg",forward=True)
        data = {self.varNames[i]:data[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        bkg = {self.varNames[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.varNames))}
        bkg_corr = {v:bkg[v] for v in self.controlVars}
        
        for j,v in enumerate(self.fitVars):
            currentVar = v
            contextVars = self.controlVars+self.fitVars[:j]
            contextIdx = [self.varNames.index(v) for v in contextVars]
            
            bkg_transforms = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
            data_transforms = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)
            if len(contextVars) == 0:
                base_dist_bkg = StandardNormal(shape=[1])
                base_dist_data = StandardNormal(shape=[1])
                bkg_flow = Flow(bkg_transforms,base_dist_bkg)
                bkg_flow.load_state_dict(torch.load(self.bkg_trainings[currentVar]['flowLoc']))
                bkg_flow.eval()
                data_flow = Flow(data_transforms,base_dist_data)
                data_flow.load_state_dict(torch.load(self.data_trainings[currentVar]['flowLoc']))
                data_flow.eval()
            else:
                encoder = NeuralNet(len(contextVars), 20, 2, 3)
                base_dist = ConditionalDiagonalNormal(shape=[1],context_encoder=encoder)
                # bkg and data share conditional base distribution
                bkg_flow = Flow(bkg_transforms,base_dist)
                bkg_flow.load_state_dict(torch.load(self.bkg_trainings[currentVar]['flowLoc']))
                bkg_flow.eval()
                data_flow = Flow(data_transforms,base_dist)
                data_flow.load_state_dict(torch.load(self.data_trainings[currentVar]['flowLoc']))
                data_flow.eval()

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
    
    def plotTriplet(ref,corr,uncorr,xlabel,title,saveName,log=False):
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
        plt.ylim([0.8,1.2])
        plt.yticks(np.arange(0.8,1.3,0.1))
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
        self.plotTriplet(data_train,corrBkg_train,bkg_train,self.varDict[var],"Train Set",saveName)
        saveName = f"{self.outDir}/{var}_trainSet_beforeAfter_log.pdf"
        self.plotTriplet(data_train,corrBkg_train,bkg_train,self.varDict[var],"Train Set",saveName,log=True)
        
        saveName = f"{self.outDir}/{var}_testSet_beforeAfter.pdf"
        self.plotTriplet(data_test,corrBkg_test,bkg_test,self.varDict[var],"Test Set",saveName)
        saveName = f"{self.outDir}/{var}_testSet_beforeAfter_log.pdf"
        self.plotTriplet(data_test,corrBkg_test,bkg_test,self.varDict[var],"Test Set",saveName,log=True)
        
    def plotCurrent(self,bins):
        currentVar = self.fitVars[self.current]
        self.plotVar(currentVar,bins)
        
    def plotAll(self,bins):
        for i,v in enumerate(self.controlVars+self.fitVars):
            b = bins[i] if type(bins)==list else bins
            self.plotVar(v,b)
        
    def runCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,bins=50,wd=0):
        print("RUNNING TRAININGS")
        self.trainCurrent(bs=bs,n_epoch=n_epoch,patience=patience,learning_rate=learning_rate,wd=wd)
        print("CORRECTING BKG")
        self.correctCurrent(n_per=10000)
        print("PLOTTING CORRECTIONS")
        self.plotCurrent(bins)
    
    def runAll(self,n_epoch=100,patience=20,learning_rate=1e-3,num_pt=100,bs=10000,wd=0):
        for i in range(len(self.fitVars)):
            nep = n_epoch[i] if type(n_epoch)==list else n_epoch
            pat = patience[i] if type(patience)==list else patience
            lr = learning_rate[i] if type(learning_rate)==list else learning_rate
            print("RUNNING TRAININGS")
            self.trainCurrent(bs=bs,n_epoch=nep,patience=pat,learning_rate=lr,wd=wd)
            print("CORRECTING BKG")
            self.correctCurrent(n_per=10000)
            self.stepForward()
        
    def stepForward(self):
        self.current += 1
        
    def stepBack(self):
        self.current -= 1
        
    def stepTo(self,step):
        self.current = step

def make_flow(num_features,num_context,kwargs,perm=True):
    flow_type = kwargs['flow_type']
    #if num_context == 0:
    #    base_dist = StandardNormal(shape=[num_features])
    #else:
    #    encoder = NeuralNet(num_context, 20, 2, 3)
    #    base_dist = ConditionalDiagonalNormal(shape=[num_features],context_encoder=encoder)
    #base_dist = StandardNormal(shape=[num_features])
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
            """"transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                      context_features=num_context,
                                                                                hidden_features=kwargs['hidden_features'],
                                                                                num_bins=kwargs['num_bins'],
                                                                                num_blocks=kwargs['num_blocks_per_layer'],
                                                                                tail_bound=kwargs['tail_bound'], 
                                                                                tails=kwargs['tails']))"""
            transforms.append(IndependentRQS(features=num_features,
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
    #flow = Flow(transform, base_dist)
    return transform