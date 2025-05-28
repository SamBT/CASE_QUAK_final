import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn.utils as utils

class NeuralNet(nn.Module):
    def __init__(self, *args,act=nn.ReLU(),out_act=nn.Sigmoid(),dropout=0):
        super().__init__()
        layers = []
        for i in range(len(args)-1):
            layers.append(nn.Linear(args[i],args[i+1]))
            if i < len(args)-2:
                layers.append(act)
                if dropout>0:
                    layers.append(nn.Dropout(p=dropout))
            else:
                layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class TimeDistributed(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module = module
        
    def forward(self,x):
        batch,nstep,n_input = x.shape
        x = x.view(batch*nstep,n_input)
        y = self.module(x)
        y = y.view(batch,nstep,y.shape[-1])
        return y
    
class LatentSum(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        y,weight = x[0],x[1]
        return torch.sum(y*weight,dim=1)

class NN(nn.Module):
    def __init__(self,in_dim,out_dim,widths,act=nn.ReLU(),out_act=nn.Sigmoid()):
        super().__init__()
        layers = [nn.Linear(in_dim,widths[0]),
                  act]
        for i in range(1,len(widths)-1):
            layers.append(nn.Linear(widths[i],widths[i+1]))
            layers.append(act)
        layers.append(nn.Linear(widths[-1],out_dim))
        layers.append(out_act)
        self.transform = nn.Sequential(*layers)

    def forward(self,x):
        return self.transform(x)
    
class SpectralNN(nn.Module):
    def __init__(self,in_dim,out_dim,widths,act=nn.ReLU(),out_act=nn.Sigmoid()):
        super().__init__()
        layers = [utils.spectral_norm(nn.Linear(in_dim,widths[0])),act]
        for i in range(1,len(widths)-1):
            layers.append(utils.spectral_norm(nn.Linear(widths[i],widths[i+1])))
            layers.append(act)
        layers.append(utils.spectral_norm(nn.Linear(widths[-1],out_dim)))
        layers.append(out_act)
        self.transform = nn.Sequential(*layers)

    def forward(self,x):
        return self.transform(x)
    
def distributed_block_constructor(in_dim,out_dim,widths,act=nn.ReLU(),out_act=nn.Sigmoid(),spectral=False):
    if spectral:
        return TimeDistributed(SpectralNN(in_dim,out_dim,widths,act,out_act))
    else:
        return TimeDistributed(NN(in_dim,out_dim,widths,act,out_act))
    
def block_constructor(in_dim,out_dim,widths,act=nn.ReLU(),out_act=nn.Sigmoid(),spectral=False):
    if spectral:
        return SpectralNN(in_dim,out_dim,widths,act,out_act)
    else:
        return NN(in_dim,out_dim,widths,act,out_act)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')

class PFN(nn.Module):
    def __init__(self,in_dim,out_dim,Phi_sizes,F_sizes,Phi_act=nn.ReLU(),F_act=nn.ReLU(),output_act=nn.Sigmoid(),spectral=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = Phi_sizes[-1]
        self.Phi = distributed_block_constructor(in_dim=in_dim,
                                     out_dim=self.latent_dim,
                                     widths=Phi_sizes[:-1],
                                     act=Phi_act,
                                     out_act=Phi_act,
                                     spectral=spectral)
        self.latent_sum = LatentSum()
        self.F =  block_constructor(in_dim=self.latent_dim,
                                     out_dim=self.out_dim,
                                     widths=F_sizes,
                                     act=F_act,
                                     out_act=output_act,
                                     spectral=spectral)

    
    def forward(self,x):
        weights = (x[:,:,:1]>0).to(torch.float32)
        latent = self.Phi(x)
        latent = latent*weights
        latent = torch.sum(latent,dim=1)
        return self.F(latent)
    
class EFN(PFN):
    def __init__(self,in_dim,out_dim,Phi_sizes,F_sizes,Phi_act=nn.ReLU(),F_act=nn.ReLU(),output_act=nn.Sigmoid(),spectral=False):
        super().__init__(in_dim,out_dim,Phi_sizes,F_sizes,Phi_act=Phi_act,F_act=F_act,output_act=output_act,spectral=spectral)
    
    def forward(self,x):
        x,z = x[:,:,1:], x[:,:,:1]
        weights = z*(z>0).to(torch.float32)
        latent = self.Phi(x)
        latent = self.latent_sum([latent,weights])
        return self.F(latent)

class JetEncoder(nn.Module):
    def __init__(self,in_dim,enc_dim,hidden_sizes,act=nn.ReLU(),out_act=nn.Sigmoid(),spectral=False):
        super().__init__()
        self.in_dim = in_dim
        self.enc_dim = enc_dim
        self.Phi = distributed_block_constructor(in_dim=in_dim,
                                                out_dim=self.enc_dim,
                                                widths=hidden_sizes,
                                                act=act,
                                                out_act=out_act,
                                                spectral=spectral)
        self.latent_sum = LatentSum()
    
    def forward(self,x):
        x,z = x[:,:,1:], x[:,:,:1]
        weights = z*(z>0).to(torch.float32)
        latent = self.Phi(x)
        latent = self.latent_sum([latent,weights])
        return latent
    
class PFNJetEncoder(nn.Module):
    def __init__(self,in_dim,enc_dim,hidden_sizes,act=nn.ReLU(),out_act=nn.Sigmoid(),spectral=False):
        super().__init__()
        self.in_dim = in_dim
        self.enc_dim = enc_dim
        self.Phi = distributed_block_constructor(in_dim=in_dim,
                                                out_dim=self.enc_dim,
                                                widths=hidden_sizes,
                                                act=act,
                                                out_act=out_act,
                                                spectral=spectral)
        self.out_act = nn.Identity()
        
    def forward(self,x):
        z = x[:,:,:1]
        latent = self.Phi(x)
        msk = (z>0).to(torch.float32)
        latent = torch.sum(latent*msk,dim=1)/torch.count_nonzero(msk,dim=1)
        return latent