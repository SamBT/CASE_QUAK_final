#from imports import *
import helper_functions as hf
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py

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
    def __init__(self,controlVars=[],fitVars=[],nTrain=100000,nTest=100000,NF_kwargs={},outDir="NF_models/",
                 dEta_cut=-1,sideband=True,rangeScale=3,minmax=True,mean_reference="BKG"):
        self.controlVars = controlVars
        self.fitVars = fitVars
        self.variables = controlVars+fitVars
        self.nTrain = nTrain
        self.nTest = nTest
        self.NF_kwargs = NF_kwargs
        self.outDir = outDir
        self.sideband = sideband
        self.dEta_cut = dEta_cut
        self.minmax = minmax
        self.rangeScale = rangeScale
        self.mean_reference = mean_reference
        
        self.varDict = hf.varTitle_map
        
        # load train data
        data = self.loadData("DATA",self.nTrain,mode='train')
        bkg = self.loadData("BKG",self.nTrain,mode='train')
        self.data_train = {self.variables[i]:data[:,i].reshape(-1,1) for i in range(len(self.variables))}
        self.bkg_train = {self.variables[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.variables))}
        del data, bkg
        
        # load test data
        data = self.loadData("DATA",self.nTrain,mode='test')
        bkg = self.loadData("BKG",self.nTrain,mode='test')
        self.data_test = {self.variables[i]:data[:,i].reshape(-1,1) for i in range(len(self.variables))}
        self.bkg_test = {self.variables[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.variables))}
        del data, bkg
        
        # variables to track training
        self.correctedBkg_train = self.bkg_train.copy()
        self.correctedBkg_test = self.bkg_test.copy()
        self.data_models = [None for _ in range(len(fitVars))]
        self.bkg_models = [None for _ in range(len(fitVars))]
        self.data_model_locs = [None for _ in range(len(fitVars))]
        self.bkg_model_locs = [None for _ in range(len(fitVars))]
        self.current = 0 # index of variable currently being corrected (index in self.varOrder)
        self.bkg_trainings = {n:{} for n in fitVars}
        self.data_trainings = {n:{} for n in fitVars}
    
    def loadMeans(self):
        means = []
        stds = []    
        ref_file = "meanStd.h5" # no dEta cut!
        with h5py.File(f"/nobackup/users/sambt/CASE_training/slimmed_h5s/{self.mean_reference}/{ref_file}","r") as f:
            for v in self.variables:
                means.append(f[v+"_mean"][()])
                stds.append(f[v+"_std"][()])
        means = np.array(means)
        stds = np.array(stds)
        return means, stds
    
    def loadMinMax(self):
        mins = []
        maxes = []
        with open("/nobackup/users/sambt/CASE_training/slimmed_h5s/trainVarMinMaxes_bkgData.json","r") as f:
            js = json.load(f)
        for v in self.variables:
            mins.append(js['mins'][v])
            maxes.append(js['maxes'][v])
        mins = np.array(mins,dtype='float32')
        maxes = np.array(maxes,dtype='float32')
        return mins, maxes
    
    def loadData(self,sample,nevts,mode="train",sideband=None,dEta_cut=None):
        print(f"Loading {mode} sample")
        sideband_set = self.sideband if sideband is None else sideband
        dEta_cut_set = self.dEta_cut if dEta_cut is None else dEta_cut
        if type(sample) == list:
            datas = []
            nev_per = int(nevts/len(sample)) if nevts != -1 else -1
            for samp in sample:
                datas.append(hf.loadData(samp,self.variables,nMax=nev_per,mode=mode,
                                         dEta_cut=dEta_cut_set,sideband=sideband_set))
                print(f"loading {samp}, {len(datas[-1])} events")
            data = np.concatenate(datas,axis=0)
            np.random.shuffle(data)
        else:
            data = hf.loadData(sample,self.variables,nMax=nevts,mode=mode,dEta_cut=dEta_cut_set,sideband=sideband_set)
        mins, maxes = self.loadMinMax()
        means, stds = self.loadMeans()
        if self.minmax:
            data = data-means
            scale = np.where(np.abs(maxes-means)>np.abs(mins-means),np.abs(maxes-means),np.abs(mins-means))
            data = self.rangeScale*(data/scale)
            #data = 2*self.rangeScale*((data-mins)/(maxes-mins)-0.5)
        else:
            data = (data-means)/stds
        return data
        
    def new_flow(self,num_features,num_context,kwargs):
        return make_flow(num_features,num_context,kwargs,perm=False)
    
    def get_flow(self,n_features,n_context,loc,kwargs):
        flow = self.new_flow(n_features,n_context,kwargs)
        flow.load_state_dict(torch.load(loc))
        flow.eval()
        return flow
    
    def train_flow(self,flow,loader,name,kwargs):
        flow = flow.to(device)
        optimizer = optim.Adam(flow.parameters(),lr=kwargs['learning_rate'])
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-5,verbose=False)

        min_loss = 1e+8
        train_losses = []
        patience_count = 0
        saveName = self.outDir+name+".pt"
        
        print("Training flow {0}".format(name))
    
        for i in range(kwargs['n_epoch']):
            if patience_count == kwargs['patience']:
                break
            epoch_losses = []
            for batch_idx, x in enumerate(loader):
                inputs,context = x
                optimizer.zero_grad()
                if torch.all(context==0):
                    loss = -flow.log_prob(inputs=inputs)[0].mean()
                else:
                    loss = -flow.log_prob(inputs=inputs,context=context)[0].mean()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    torch.save(flow.state_dict(),saveName)
            epoch_loss = np.min(epoch_losses)
            train_losses.append(epoch_loss)
            if (i + 1) % kwargs['print_interval'] == 0:
                print('Iteration {} Complete'.format(i + 1))
                print('Loss: ', epoch_loss)
            if epoch_loss == min_loss:
                patience_count = 0
            else:
                patience_count += 1
            #scheduler.step()
        
        flow.load_state_dict(torch.load(saveName))
        flow.eval()
        torch.cuda.empty_cache()
        flow = flow.to('cpu')

        return flow, saveName, train_losses
    
    def trainCurrentBkg(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10):
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'print_interval':print_interval}
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        
        # train bkg flow
        if len(contextVars) > 0:
            bkg_train_context = torch.tensor(np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1),dtype=torch.float32,device=device)
        else:
            bkg_train_context = torch.zeros(self.bkg_train[currentVar].shape,dtype=torch.float32,device=device)
        bkg_train_var = torch.tensor(self.bkg_train[currentVar],dtype=torch.float32,device=device)
        bkg_train_dataset = utils.TensorDataset(bkg_train_var,bkg_train_context)
        bkg_loader = utils.DataLoader(bkg_train_dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
        bkg_flow = self.new_flow(1,len(contextVars),self.NF_kwargs)
        bkg_flowName = "bkgFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        bkg_flow, bkg_flowLoc, bkg_trainLosses = self.train_flow(bkg_flow,bkg_loader,bkg_flowName,train_kwargs)
        self.bkg_trainings[currentVar]['flowLoc'] = bkg_flowLoc
        self.bkg_trainings[currentVar]['flowName'] = bkg_flowName
        self.bkg_trainings[currentVar]['contextVars'] = contextVars
        self.bkg_trainings[currentVar]['NF_kwargs'] = self.NF_kwargs
        self.bkg_trainings[currentVar]['train_kwargs'] = train_kwargs
        self.bkg_trainings[currentVar]['losses'] = bkg_trainLosses
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(len(bkg_trainLosses)),bkg_trainLosses)
        plt.title(bkg_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(self.outDir+"trainCurve_{0}.pdf".format(bkg_flowName))
        del bkg_train_context, bkg_train_var, bkg_train_dataset, bkg_loader, bkg_flow
        torch.cuda.empty_cache()
        
    def trainCurrentData(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10):
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'print_interval':print_interval}
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        
        # train data flow
        if len(contextVars) > 0:
            data_train_context = torch.tensor(np.concatenate([self.data_train[n] for n in contextVars],axis=1),dtype=torch.float32,device=device)
        else:
            data_train_context = torch.zeros(self.data_train[currentVar].shape,dtype=torch.float32,device=device)
        data_train_var = torch.tensor(self.data_train[currentVar],dtype=torch.float32,device=device)
        data_train_dataset = utils.TensorDataset(data_train_var,data_train_context)
        data_loader = utils.DataLoader(data_train_dataset,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
        data_flow = self.new_flow(1,len(contextVars),self.NF_kwargs)
        data_flowName = "dataFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        data_flow, data_flowLoc, data_trainLosses = self.train_flow(data_flow,data_loader,data_flowName,train_kwargs)
        self.data_trainings[currentVar]['flowLoc'] = data_flowLoc
        self.data_trainings[currentVar]['flowName'] = data_flowName
        self.data_trainings[currentVar]['contextVars'] = contextVars
        self.data_trainings[currentVar]['NF_kwargs'] = self.NF_kwargs
        self.data_trainings[currentVar]['train_kwargs'] = train_kwargs
        self.data_trainings[currentVar]['losses'] = data_trainLosses
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(len(data_trainLosses)),data_trainLosses)
        plt.title(data_flowName)
        plt.xlabel('Epoch',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(self.outDir+"trainCurve_{0}.pdf".format(data_flowName))
        del data_train_context, data_train_var, data_train_dataset, data_loader, data_flow
        torch.cuda.empty_cache()
        
    def trainCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10):
        self.trainCurrentBkg(bs=bs,n_epoch=n_epoch,patience=patience,learning_rate=learning_rate,print_interval=print_interval)
        self.plotDensity(bkg=True,data=False)
        self.trainCurrentData(bs=bs,n_epoch=n_epoch,patience=patience,learning_rate=learning_rate,print_interval=print_interval)
        self.plotDensity(bkg=False,data=True)
        
    def plotDensity(self,bkg=True,data=True):
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        
        if bkg:
            bkg_flow = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
            bkg_test_var = self.bkg_test[currentVar]
            if len(contextVars) > 0:
                bkg_test_context = np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1)
                bkg_samples = bkg_flow.sample(1,context=torch.tensor(bkg_test_context,dtype=torch.float32,device=device)).detach().cpu().numpy().reshape(bkg_test_context.shape[0],-1)
                del bkg_test_context
            else:
                bkg_samples = bkg_flow.sample(bkg_test_var.shape[0]).detach().cpu().numpy()
            plt.figure(figsize=(8,6))
            h1 = plt.hist(bkg_test_var[:,0],bins=np.linspace(-self.rangeScale,self.rangeScale,50),density=True,histtype='step',label='bkg test')
            h1 = plt.hist(bkg_samples[:,0],bins=np.linspace(-self.rangeScale,self.rangeScale,50),density=True,histtype='step',label='bkg samples')
            plt.legend()
            del bkg_flow, bkg_samples, bkg_test_var
            torch.cuda.empty_cache()
        
        if data:
            data_flow = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)
            data_test_var = self.data_test[currentVar]
            if len(contextVars) > 0:
                data_test_context = np.concatenate([self.data_test[n] for n in contextVars],axis=1)
                data_samples = data_flow.sample(1,context=torch.tensor(data_test_context,dtype=torch.float32,device=device)).detach().cpu().numpy().reshape(data_test_context.shape[0],-1)
                del data_test_context
            else:
                data_samples = data_flow.sample(data_test_var.shape[0]).detach().cpu().numpy()
            plt.figure(figsize=(8,6))
            h1 = plt.hist(data_test_var[:,0],bins=np.linspace(-self.rangeScale,self.rangeScale,50),density=True,histtype='step',label='data test')
            h1 = plt.hist(data_samples[:,0],bins=np.linspace(-self.rangeScale,self.rangeScale,50),density=True,histtype='step',label='data samples')
            plt.legend()
            del data_flow, data_samples, data_test_var
            torch.cuda.empty_cache()
        
    def correctCurrent(self,npt=100,n_per=10000,pad=0.1):
        currentVar = self.fitVars[self.current]
        contextVars = self.controlVars+self.fitVars[:self.current]
        bkg_flow = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
        data_flow = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)
        
        # correct training set
        if len(contextVars) > 0:
            bkg_train_context = np.concatenate([self.correctedBkg_train[n] for n in contextVars],axis=1)
        else:
            bkg_train_context = torch.zeros(self.bkg_train[currentVar].shape,dtype=torch.float32,device=device)
        bkg_train_var = self.bkg_train[currentVar]
        nTrain = bkg_train_context.shape[0]
        split = np.array_split(np.arange(nTrain),nTrain//n_per + 1)
        bkg_train_context = [bkg_train_context[k] for k in split]
        bkg_train_var = [bkg_train_var[k] for k in split]
        bkg_train_corr = []
        for i in tqdm(range(len(split))):
            inputs = torch.tensor(bkg_train_var[i],dtype=torch.float32,device=device)
            context = torch.tensor(bkg_train_context[i],dtype=torch.float32,device=device)
            if torch.all(context==0):
                noise = bkg_flow.transform_to_noise(inputs=inputs)
                corrected = data_flow._transform.inverse(noise)[0]
            else:
                noise = bkg_flow.transform_to_noise(inputs=inputs,context=context)
                corrected = data_flow._transform.inverse(noise,context=context)[0]
            bkg_train_corr.append(corrected.detach().cpu().numpy())
            del inputs,context,noise,corrected
            torch.cuda.empty_cache()
        bkg_train_corr = np.concatenate(bkg_train_corr,axis=0)
        self.correctedBkg_train[currentVar] = np.copy(bkg_train_corr)
        del bkg_train_corr
        
        # correct test set
        if len(contextVars) > 0:
            bkg_test_context = np.concatenate([self.correctedBkg_test[n] for n in contextVars],axis=1)
        else:
            bkg_test_context = torch.zeros(self.bkg_test[currentVar].shape,dtype=torch.float32,device=device)
        bkg_test_var = self.bkg_test[currentVar]
        nTest = bkg_test_context.shape[0]
        split = np.array_split(np.arange(nTest),nTest//n_per + 1)
        bkg_test_context = [bkg_test_context[k] for k in split]
        bkg_test_var = [bkg_test_var[k] for k in split]
        bkg_test_corr = []
        for i in tqdm(range(len(split))):
            inputs = torch.tensor(bkg_test_var[i],dtype=torch.float32,device=device)
            context = torch.tensor(bkg_test_context[i],dtype=torch.float32,device=device)
            if torch.all(context==0):
                noise = bkg_flow.transform_to_noise(inputs=inputs)
                corrected = data_flow._transform.inverse(noise)[0]
            else:
                noise = bkg_flow.transform_to_noise(inputs=inputs,context=context)
                corrected = data_flow._transform.inverse(noise,context=context)[0]
            bkg_test_corr.append(corrected.detach().cpu().numpy())
            del inputs,context,noise,corrected
            torch.cuda.empty_cache()
        bkg_test_corr = np.concatenate(bkg_test_corr,axis=0)
        self.correctedBkg_test[currentVar] = np.copy(bkg_test_corr)
        del bkg_test_corr
        
        del bkg_flow, data_flow
        torch.cuda.empty_cache()
        
        """
        nTrain = self.bkg_train[currentVar].shape[0]
        nTest = self.bkg_test[currentVar].shape[0]
        train_split = nTrain//n_per
        test_split = nTest//n_per
        
        # CDF integral points
        bmin, bmax = np.min(self.bkg_train[currentVar]), np.max(self.bkg_train[currentVar])
        dmin, dmax = np.min(self.data_train[currentVar]), np.max(self.data_train[currentVar])
        xmin = min(bmin,dmin)-pad
        xmax = max(bmax,dmax)+pad
        x = np.linspace(xmin,xmax,npt)
        print(f"xmin = {xmin}, xmax={xmax}")
        bs = x[1]-x[0]
        
        # correct training bkg
        bkg_train = np.array_split(np.concatenate([self.correctedBkg_train[n] for n in self.controlVars+self.fitVars[:self.current]],axis=1),train_split)
        pbs = []
        pds = []
        print("Correcting train data")
        for i,bc in tqdm(enumerate(bkg_train)):
            ntest = bc.shape[0]
            xi = np.tile(x,ntest).reshape(-1,1)
            bc = bc.repeat(npt,axis=0)
            bc = np.concatenate((bc,xi),axis=1)
            bc = torch.tensor(bc,dtype=torch.float32).to(device)
            pbs.append(np.exp(bkg_flow.eval_log_prob(bc)[0]).reshape(ntest,npt))
            pds.append(np.exp(data_flow.eval_log_prob(bc)[0]).reshape(ntest,npt))
            del bc,xi
            torch.cuda.empty_cache()
        pb = np.concatenate(pbs,axis=0)
        pd = np.concatenate(pds,axis=0)
        qb = bs*np.cumsum(pb,axis=1)/np.tile(np.sum(pb,axis=1).reshape(-1,1),(1,pb.shape[1]))
        qd = bs*np.cumsum(pd,axis=1)/np.tile(np.sum(pd,axis=1).reshape(-1,1),(1,pd.shape[1]))
        toCorrect = np.copy(self.correctedBkg_train[currentVar])[:,0]
        Ycorr = -999*np.ones_like(toCorrect)
        print(toCorrect.shape)
        print(x.shape)
        print(qb.shape)
        print(qd.shape)
        correctAll(toCorrect,x.astype('float32'),qb,qd,Ycorr)
        self.correctedBkg_train[currentVar] = np.copy(Ycorr).reshape(-1,1)
        del pbs,pds,pb,pd,qb,qd,toCorrect,Ycorr
        
        # correct test bkg
        bkg_test = np.array_split(np.concatenate([self.correctedBkg_test[n] for n in self.controlVars+self.fitVars[:self.current]],axis=1),test_split)
        pbs = []
        pds = []
        print("Correcting test data")
        for i,bc in tqdm(enumerate(bkg_test)):
            ntest = bc.shape[0]
            xi = np.tile(x,ntest).reshape(-1,1)
            bc = bc.repeat(npt,axis=0)
            bc = np.concatenate((bc,xi),axis=1)
            bc = torch.tensor(bc,dtype=torch.float32).to(device)
            pbs.append(np.exp(bkg_flow.eval_log_prob(bc)[0]).reshape(ntest,npt))
            pds.append(np.exp(data_flow.eval_log_prob(bc)[0]).reshape(ntest,npt))
            del bc,xi
            torch.cuda.empty_cache()
        pb = np.concatenate(pbs,axis=0)
        pd = np.concatenate(pds,axis=0)
        qb = bs*np.cumsum(pb,axis=1)/np.tile(np.sum(pb,axis=1).reshape(-1,1),(1,pb.shape[1]))
        qd = bs*np.cumsum(pd,axis=1)/np.tile(np.sum(pd,axis=1).reshape(-1,1),(1,pd.shape[1]))
        toCorrect = np.copy(self.correctedBkg_test[currentVar])[:,0]
        Ycorr = -999*np.ones_like(toCorrect)
        correctAll(toCorrect,x.astype('float32'),qb,qd,Ycorr)
        self.correctedBkg_test[currentVar] = np.copy(Ycorr).reshape(-1,1)
        del pbs,pds,pb,pd,qb,qd,toCorrect,Ycorr
        
        bkg_flow = bkg_flow.to('cpu')
        data_flow = data_flow.to('cpu')
        del bkg_flow, data_flow
        torch.cuda.empty_cache()"""
        
    def correctFull(self,n_per=10000,nevts=1000000):
        data = self.loadData("DATA",nevts,mode='all',sideband=False,dEta_cut=-1)
        bkg = self.loadData("BKG",nevts,mode='all',sideband=False,dEta_cut=-1)
        data = {self.variables[i]:data[:,i].reshape(-1,1) for i in range(len(self.variables))}
        bkg = {self.variables[i]:bkg[:,i].reshape(-1,1) for i in range(len(self.variables))}
        bkg_corr = {v:bkg[v] for v in self.controlVars}
        
        for j,v in enumerate(self.fitVars):
            currentVar = v
            contextVars = self.controlVars+self.fitVars[:j]
            bkg_flow = self.get_flow(1,len(self.bkg_trainings[currentVar]['contextVars']),self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
            data_flow = self.get_flow(1,len(self.data_trainings[currentVar]['contextVars']),self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)

            # correct training set
            if len(contextVars) > 0:
                context = np.concatenate([bkg_corr[n] for n in contextVars],axis=1)
            else:
                context = np.zeros(bkg[currentVar].shape)
            inputs = bkg[currentVar]
            nEvts = inputs.shape[0]
            split = np.array_split(np.arange(nEvts),nEvts//n_per + 1)
            context = [context[k] for k in split]
            inputs = [inputs[k] for k in split]
            corr = []
            for i in tqdm(range(len(split))):
                inputs_i = torch.tensor(inputs[i],dtype=torch.float32,device=device)
                context_i = torch.tensor(context[i],dtype=torch.float32,device=device)
                with torch.no_grad():
                    if torch.all(context_i==0):
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i)
                        corrected = data_flow._transform.inverse(noise)[0]
                    else:
                        noise = bkg_flow.transform_to_noise(inputs=inputs_i,context=context_i)
                        corrected = data_flow._transform.inverse(noise,context=context_i)[0]
                corr.append(corrected.detach().cpu().numpy())
                del inputs_i,context_i,noise,corrected
                torch.cuda.empty_cache()
            corr = np.concatenate(corr,axis=0)
            bkg_corr[currentVar] = np.copy(corr)
            del corr
        
        return data,bkg_corr,self.variables
    
    def plotVar(self,var,bins):
        data_train = self.data_train[var]
        data_test = self.data_test[var]
        bkg_train = self.bkg_train[var]
        bkg_test = self.bkg_test[var]
        corrBkg_train = self.correctedBkg_train[var]
        corrBkg_test = self.correctedBkg_test[var]
        
        plt.figure(figsize=(16,6))
        plt.subplot(121)
        h1 = plt.hist(data_train,bins=bins,histtype='step',label="Data")
        #h2 = plt.hist(bkg_train,bins=bins,histtype='step',label="Raw Bkg")
        h3 = plt.hist(corrBkg_train,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TRAIN SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.subplot(122)
        h1 = plt.hist(data_train,bins=bins,histtype='step',label="Data")
        #h2 = plt.hist(bkg_train,bins=bins,histtype='step',label="Raw Bkg")
        h3 = plt.hist(corrBkg_train,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TRAIN SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.yscale('log')
        plt.savefig(self.outDir+"train_beforeAfter_{0}.pdf".format(var))
        
        plt.figure(figsize=(16,6))
        plt.subplot(121)
        h1 = plt.hist(data_test,bins=bins,histtype='step',label="Data")
        #h2 = plt.hist(bkg_test,bins=bins,histtype='step',label="Raw Bkg")
        h3= plt.hist(corrBkg_test,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TEST SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.subplot(122)
        h1 = plt.hist(data_test,bins=bins,histtype='step',label="Data")
        #h2 = plt.hist(bkg_test,bins=bins,histtype='step',label="Raw Bkg")
        h3 = plt.hist(corrBkg_test,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TEST SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.yscale('log')
        plt.savefig(self.outDir+"test_beforeAfter_{0}.pdf".format(var))
        
    def plotCurrent(self,bins):
        currentVar = self.fitVars[self.current]
        self.plotVar(currentVar,bins)
        
    def plotAll(self,bins):
        for v in self.controlVars+self.fitVars:
            self.plotVar(v,bins)
        
    def runCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10,npt=100,bins=np.linspace(-8,8,num=100)):
        print("RUNNING TRAININGS")
        self.trainCurrent(bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10)
        print("CORRECTING BKG")
        self.correctCurrent(npt=npt,n_per=10000)
        print("PLOTTING CORRECTIONS")
        self.plotCurrent(bins)
    
    def runAll(self,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10,num_pt=100):
        for i in range(len(self.fitVars)):
            nep = n_epoch[i] if type(n_epoch)==list else n_epoch
            pat = patience[i] if type(patience)==list else patience
            lr = learning_rate[i] if type(learning_rate)==list else learning_rate
            pi = print_interval[i] if type(print_interval)==list else print_interval
            npt = num_pt[i] if type(num_pt)==list else num_pt
            print("RUNNING TRAININGS")
            self.trainCurrent(bs=10000,n_epoch=nep,patience=pat,learning_rate=lr,print_interval=pi)
            print("CORRECTING BKG")
            self.correctCurrent(npt=npt,n_per=10000)
            self.stepForward()
        
    def stepForward(self):
        self.current += 1
        
    def stepBack(self):
        self.current -= 1
        
    def stepTo(self,step):
        self.current = step

def make_flow(num_features,num_context,kwargs,perm=True):
    flow_type = kwargs['flow_type']
    base_dist = StandardNormal(shape=[num_features])
    #base_dist = ConditionalDiagonalNormal(shape=[num_features],context_encoder=nn.Linear(num_context,2*num_features))
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
                                                                                tails=kwargs['tails']))
        if i < kwargs['num_layers'] - 1 and perm:
            transforms.append(ReversePermutation(features=num_features))

    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow