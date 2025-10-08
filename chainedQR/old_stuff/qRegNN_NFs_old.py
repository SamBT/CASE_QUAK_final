from imports import *
import helper_functions as hf
import time

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
    def __init__(self,data,bkg,controlVars=[],fitVars=[],nTrain=1000000,nTest=100000,NF_kwargs={},outDir="NF_models/"):
        self.controlVars = controlVars
        self.fitVars = fitVars
        self.varDict = hf.varTitle_map
        self.data_loc = data
        self.bkg_loc = bkg
        self.NF_kwargs = NF_kwargs
        self.outDir = outDir
        
        # load data
        f_data = h5py.File(data,"r")
        self.data_train = {n:f_data[n][()][:nTrain] for n in controlVars+fitVars}
        self.data_means = {n:np.mean(self.data_train[n]) for n in controlVars+fitVars}
        self.data_stds = {n:np.std(self.data_train[n]) for n in controlVars+fitVars}
        for n in controlVars+fitVars:
            self.data_train[n] = (self.data_train[n] - self.data_means[n])/self.data_stds[n]
        self.data_test = {n:(f_data[n][()][nTrain:nTrain+nTest]-self.data_means[n])/self.data_stds[n] for n in controlVars+fitVars}
        f_data.close()
        del f_data
        
        # load bkg
        f_bkg = h5py.File(bkg,"r")
        self.bkg_train = {n:(f_bkg[n][()][:nTrain]-self.data_means[n])/self.data_stds[n] for n in controlVars+fitVars}
        self.bkg_test = {n:(f_bkg[n][()][nTrain:nTrain+nTest]-self.data_means[n])/self.data_stds[n] for n in controlVars+fitVars}
        f_bkg.close()
        del f_bkg
        
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
        
    def new_flow(self,num_features,kwargs=None):
        if kwargs is None:
            kwargs = self.NF_kwargs.copy()
            kwargs['num_features'] = num_features
        flow_type = kwargs['flow_type']
        
        base_dist = StandardNormal(shape=[kwargs['num_features']])
        transforms = []
        for i in range(kwargs['num_layers']):
            if flow_type == 'MAF':
                transforms.append(MaskedAffineAutoregressiveTransform(features=kwargs['num_features'],
                                                                      hidden_features=kwargs['hidden_features'], 
                                                                      num_blocks=kwargs['num_blocks_per_layer']))
            elif flow_type == 'NSQUAD':
                transforms.append(MaskedPiecewiseQuadraticAutoregressiveTransform(features=kwargs['num_features'],
                                                                                  hidden_features=kwargs['num_features'],
                                                                                  num_bins=kwargs['num_bins'],
                                                                                  num_blocks=kwargs['num_blocks_per_layer'],
                                                                                  tail_bound=kwargs['tail_bound'], 
                                                                                  tails='linear'))
            elif flow_type == 'NSRATQUAD':
                transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=kwargs['num_features'],
                                                                                          hidden_features=kwargs['hidden_features'],
                                                                                          num_bins=kwargs['num_bins'],
                                                                                          num_blocks=kwargs['num_blocks_per_layer'],
                                                                                          tail_bound=kwargs['tail_bound'], 
                                                                                          tails='linear'))
            if i < kwargs['num_layers'] - 1:
                transforms.append(ReversePermutation(features=kwargs['num_features']))

        transform = CompositeTransform(transforms)
        flow = Flow(transform, base_dist)
        return flow, kwargs
    
    def get_flow(self,loc,kwargs):
        flow,kwargs = self.new_flow(-1,kwargs)
        flow.load_state_dict(torch.load(loc))
        flow.eval()
        return flow
    
    def train_flow(self,flow,loader,name,kwargs):
        flow = flow.to(device)
        optimizer = optim.Adam(flow.parameters(),lr=kwargs['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-5,verbose=False)

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
            if (i + 1) % kwargs['print_interval'] == 0:
                print('Iteration {} Complete'.format(i + 1))
                print('Loss: ', epoch_loss)
            if epoch_loss == min_loss:
                patience_count = 0
            else:
                patience_count += 1
            scheduler.step()
        
        flow.load_state_dict(torch.load(saveName))
        flow.eval()
        torch.cuda.empty_cache()
        flow = flow.to('cpu')

        return flow, saveName, train_losses
        
    def trainCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10):
        train_kwargs = {'n_epoch':n_epoch,'patience':patience,'learning_rate':learning_rate,'print_interval':print_interval}
        currentVar = self.fitVars[self.current]
        
        # train bkg flow
        bkg_train = [self.correctedBkg_train[n] for n in self.controlVars+self.fitVars[:self.current]] + [self.bkg_train[self.fitVars[self.current]]]
        bkg_train = torch.tensor(np.concatenate(bkg_train,axis=1),device=device)
        bkg_loader = utils.DataLoader(bkg_train,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
        bkg_flow, bkg_NFkwargs = self.new_flow(bkg_train.shape[1])
        bkg_flowName = "bkgFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        bkg_flow, bkg_flowLoc, bkg_trainLosses = self.train_flow(bkg_flow,bkg_loader,bkg_flowName,train_kwargs)
        self.bkg_trainings[currentVar]['flowLoc'] = bkg_flowLoc
        self.bkg_trainings[currentVar]['flowName'] = bkg_flowName
        self.bkg_trainings[currentVar]['NF_kwargs'] = bkg_NFkwargs
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
        del bkg_train, bkg_loader, bkg_flow
        torch.cuda.empty_cache()
        
        # train data flow
        data_train = [self.data_train[n] for n in self.controlVars+self.fitVars[:self.current+1]]
        data_train = torch.tensor(np.concatenate(data_train,axis=1),device=device)
        data_loader = utils.DataLoader(data_train,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))
        data_flow, data_NFkwargs = self.new_flow(data_train.shape[1])
        data_flowName = "dataFlow_step{0}_{1}".format(self.current,self.fitVars[self.current])
        data_flow = data_flow.to(device)
        data_flow, data_flowLoc, data_trainLosses = self.train_flow(data_flow,data_loader,data_flowName,train_kwargs)
        data_flow = data_flow.to('cpu')
        self.data_trainings[currentVar]['flowLoc'] = data_flowLoc
        self.data_trainings[currentVar]['flowName'] = data_flowName
        self.data_trainings[currentVar]['NF_kwargs'] = data_NFkwargs
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
        del data_train, data_loader, data_flow
        torch.cuda.empty_cache()
        
    def correctCurrent(self,npt=100,n_per=1000,pad=0.1):
        currentVar = self.fitVars[self.current]
        bkg_flow = self.get_flow(self.bkg_trainings[currentVar]['flowLoc'],self.bkg_trainings[currentVar]['NF_kwargs']).to(device)
        data_flow = self.get_flow(self.data_trainings[currentVar]['flowLoc'],self.data_trainings[currentVar]['NF_kwargs']).to(device)
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
        torch.cuda.empty_cache()
        
    def plotVar(self,var,bins):
        data_train = self.data_train[var]
        data_test = self.data_test[var]
        bkg_train = self.bkg_train[var]
        bkg_test = self.bkg_test[var]
        corrBkg_train = self.correctedBkg_train[var]
        corrBkg_test = self.correctedBkg_test[var]
        
        plt.figure(figsize=(16,16))
        plt.subplot(221)
        h1 = plt.hist(data_train,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(bkg_train,bins=bins,histtype='step',label="Raw Bkg")
        plt.title("TRAIN SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.subplot(222)
        h1 = plt.hist(data_train,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(corrBkg_train,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TRAIN SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.subplot(223)
        h1 = plt.hist(data_train,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(bkg_train,bins=bins,histtype='step',label="Raw Bkg")
        plt.title("TRAIN SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.yscale('log')
        plt.subplot(224)
        h1 = plt.hist(data_train,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(corrBkg_train,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TRAIN SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.yscale('log')
        plt.savefig(self.outDir+"train_beforeAfter_{0}.pdf".format(var))
        
        plt.figure(figsize=(16,16))
        plt.subplot(221)
        h1 = plt.hist(data_test,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(bkg_test,bins=bins,histtype='step',label="Raw Bkg")
        plt.title("TEST SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.subplot(222)
        h1 = plt.hist(data_test,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(corrBkg_test,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TEST SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.subplot(223)
        h1 = plt.hist(data_test,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(bkg_test,bins=bins,histtype='step',label="Raw Bkg")
        plt.title("TEST SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.yscale('log')
        plt.subplot(224)
        h1 = plt.hist(data_test,bins=bins,histtype='step',label="Data")
        h2 = plt.hist(corrBkg_test,bins=bins,histtype='step',label="Corr Bkg")
        plt.title("TEST SET")
        plt.legend(loc='best')
        plt.xlabel(self.varDict[var])
        plt.yscale('log')
        plt.savefig(self.outDir+"test_beforeAfter_{0}.pdf".format(var))
        
    def plotCurrent(self,bins):
        currentVar = self.fitVars[self.current]
        self.plotVar(currentVar,bins)
        
    def plotAll(self,bins):
        for v in self.fitVars:
            self.plotVar(v,bins)
        
    def runCurrent(self,bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10,npt=100,bins=np.linspace(-8,8,num=100)):
        print("RUNNING TRAININGS")
        self.trainCurrent(bs=10000,n_epoch=100,patience=20,learning_rate=1e-3,print_interval=10)
        print("CORRECTING BKG")
        self.correctCurrent(npt=npt,n_per=1000)
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
            self.correctCurrent(npt=npt,n_per=1000)
            self.stepForward()
        
    def stepForward(self):
        self.current += 1
        
    def stepBack(self):
        self.current -= 1
        
    def stepTo(self,step):
        self.current = step