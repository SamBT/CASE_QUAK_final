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

h5_map = {
    "mjj":"jet_kinematics/0",
    "dEta":"jet_kinematics/1",
    "jet1_pt":"jet_kinematics/2",
    "jet1_eta":"jet_kinematics/3",
    "jet1_phi":"jet_kinematics/4",
    "jet1_mass":"jet_kinematics/5",
    "jet2_pt":"jet_kinematics/6",
    "jet2_eta":"jet_kinematics/7",
    "jet2_phi":"jet_kinematics/8",
    "jet2_mass":"jet_kinematics/9",
    "jet3_pt":"jet_kinematics/10",
    "jet3_eta":"jet_kinematics/11",
    "jet3_phi":"jet_kinematics/12",
    "jet3_mass":"jet_kinematics/13",
    "jet1_tau1":"jet1_extraInfo/0",
    "jet1_tau2":"jet1_extraInfo/1",
    "jet1_tau3":"jet1_extraInfo/2",
    "jet1_tau4":"jet1_extraInfo/3",
    "jet1_lsf3":"jet1_extraInfo/4",
    "jet1_btagscore":"jet1_extraInfo/5",
    "jet2_tau1":"jet2_extraInfo/0",
    "jet2_tau2":"jet2_extraInfo/1",
    "jet2_tau3":"jet2_extraInfo/2",
    "jet2_tau4":"jet2_extraInfo/3",
    "jet2_lsf3":"jet2_extraInfo/4",
    "jet2_btagscore":"jet2_extraInfo/5",
    "truth_label":"truth_label/0",
    "year":"event_info/6",
    "MET":"event_info/1",
    "MET_phi":"event_info/2",
    "genWeight":"event_info/3",
    "eventNum":"event_info/0",
    "runNum":"event_info/5",
    "presel_eff":"preselection_eff",
    "jet1_rho":"jet1_rho",
    "jet1_tau21":"jet1_tau21",
    "jet1_tau32":"jet1_tau32",
    "jet1_tau43":"jet1_tau43",
    "jet1_tauS":"jet1_tauS",
    "jet1_logTauS":"jet1_logTauS",
    "jet1_pb":"jet1_pb",
    "jet1_npf":"jet1_extraInfo/6",
    "jet2_rho":"jet2_rho",
    "jet2_tau21":"jet2_tau21",
    "jet2_tau32":"jet2_tau32",
    "jet2_tau43":"jet2_tau43",
    "jet2_tauS":"jet2_tauS",
    "jet2_logTauS":"jet2_logTauS",
    "jet2_pb":"jet2_pb",
    "jet2_npf":"jet2_extraInfo/6",
    "signed_dEta":"signed_dEta",
    "nom_sys_wgt":"sys_weights/0"
}

varName_map = {
    r'$M_{j1}$':"jet1_mass",
    r'$\log(M_{j1})$':"jet1_logMass",
    r'Jet 1 $\tau_{21}$':"jet1_tau21", 
    r'Jet 1 $\tau_{32}$':"jet1_tau32", 
    r'Jet 1 $\tau_{43}$':"jet1_tau43", 
    r'Jet 1 $\tau_s$':"jet1_tauS", 
    r'Jet 1 $P_b$':"jet1_pb", 
    r'Jet 1 $n_{pf}$':"jet1_npf",
    r'Jet 1 $p_T$':"jet1_pt",
    r'Jet 1 $\eta$':"jet1_eta",
    r'Jet 1 $\phi$':"jet1_phi",
    r'Jet 1 $\rho$':"jet1_rho",
    r'Jet 1 $\log(1+\tau_s)$':"jet1_logTauS",
    r'Jet 1 LSF$_3$':"jet1_lsf3",
    r'$M_{j2}$':"jet2_mass", 
    r'$\log(M_{j2})$':"jet2_logMass",
    r'Jet 2 $\tau_{21}$':"jet2_tau21", 
    r'Jet 2 $\tau_{32}$':"jet2_tau32", 
    r'Jet 2 $\tau_{43}$':"jet2_tau43", 
    r'Jet 2 $\tau_s$':"jet2_tauS", 
    r'Jet 2 $P_b$':"jet2_pb", 
    r'Jet 2 $n_{pf}$':"jet2_npf",
    r'Jet 2 $p_T$':"jet2_pt",
    r'Jet 2 $\eta$':"jet2_eta",
    r'Jet 2 $\phi$':"jet2_phi",
    r'Jet 2 $\rho$':"jet2_rho",
    r'Jet 2 $\log(1+\tau_s)$':"jet2_logTauS",
    r'Jet 2 LSF$_3$':"jet2_lsf3",
    r'$\Delta \eta (j_1,j_2)$':"dEta",
    r'Missing $E_T$':"MET",
    "Gen Weight":"genWeight",
    r'MET $\phi$':"MET_phi",
    "Event Number":"eventNum",
    "Year":"year",
    "Truth Label":"truth_label",
    "Run Number":"runNum",
    "Preselection Efficiency":"presel_eff",
    "File Name":"fname",
    r'$\Delta \eta$':"signed_dEta"
}
varTitle_map = {varName_map[k]:k for k in varName_map}

def getVariable(f,variable,useName=False):
    if useName:
        return f[variable][()]
    else:
        if variable in h5_map.keys():
            loc = h5_map[variable]
            if "/" in loc:
                branch = loc.split("/")[0]
                ind = int(loc.split("/")[1])
                return f[branch][:,ind]
            else:
                return f[loc][()]
        else:
            return f[variable][()]
        
def getSidebandCut(f):
    dEta = np.abs(getVariable(f,"dEta"))
    jet1_pt = getVariable(f,"jet1_pt")
    jet2_pt = getVariable(f,"jet2_pt")
    jet3_pt = getVariable(f,"jet3_pt")
    mjj = getVariable(f,"mjj")
    cutVal = 2 * jet1_pt * jet2_pt * (np.cosh(dEta)+1) / (mjj*mjj)
    jet_asymm = np.abs((jet1_pt-jet2_pt)/(jet1_pt+jet2_pt))
    cut = (dEta > 2.0) & (dEta < 2.5) & (jet1_pt > 300) & (jet2_pt > 300) & (jet3_pt < 300) & ( (cutVal>1.0) | (cutVal<0.95) | (jet_asymm > 0.1) )
    return cut
    
def loadVariables(file,variables,dEta_cut=1.3,mode="all",useName=False,sideband=False):
    with h5py.File(file,"r") as f:
        output = []
        for v in variables:
            output.append(getVariable(f,v,useName=useName).reshape(-1,1))
        output = np.concatenate(output,axis=1)
        if dEta_cut > 0:
            dEta = getVariable(f,"dEta")
            cut = np.abs(dEta) < dEta_cut
            output = output[cut]
        elif sideband:
            cut = getSidebandCut(f)
            output = output[cut]
    if mode == "train":
        output = output[:int(0.9*len(output))]
    if mode == "test":
        output = output[int(0.9*len(output)):]
    return output
        

def loadData(sample,variables,nMax=-1,dEta_cut=1.3,mode="all",sideband=False):
    from samples import data_merged_dir, bkg_merged_dir, signal_merged_dir, signal_map
    useName = False
    if sample == "BKG":
        files = [bkg_merged_dir+f for f in os.listdir(bkg_merged_dir) if ".h5" in f]
    elif sample == "BKG_mjjFlat":
        files = ["input_h5s/BKG/BKG_mjjFlat_nodEtaCut.h5"]
        useName = True
        dEta_cut = -1
    elif sample == "ttbar":
        files = ["input_h5s/BKG/ttbar.h5"]
        useName = True
        dEta_cut = -1
    elif sample == "DATA":
        files = [data_merged_dir+f for f in os.listdir(data_merged_dir) if ".h5" in f]
    elif sample == "DATA_mjjFlat":
        files = ["input_h5s/DATA/DATA_sideband_mjjFlat.h5"]
        useName = True
        dEta_cut = -1
    else:
        files = [signal_merged_dir+signal_map[sample]]
    
    #print(f"loading files {files}")
    tot_loaded = 0
    data = []
    for file in tqdm(files):
        if nMax > 0 and tot_loaded >= nMax:
            break
        data.append(loadVariables(file,variables,dEta_cut=dEta_cut,mode=mode,useName=useName,sideband=sideband))
        tot_loaded += data[-1].shape[0]
    data = np.concatenate(data,axis=0)
    if nMax > 0:
        cutoff = min(nMax,data.shape[0])
        data = data[:cutoff]
    return data

def calculateMeansStds(sample,variables,dEta_cut=1.3,sideband=False):
    outName = "meanStd"
    if dEta_cut > 0:
        outName += f"_dEta{dEta_cut:.1f}"
    if sideband:
        outName += "_sideband"
    outName += ".h5"
    with h5py.File(f"input_h5s/{sample}/{outName}","w") as f:
        data = loadData(sample,variables,dEta_cut=dEta_cut,sideband=sideband)
        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        for i,v in enumerate(variables):
            f.create_dataset(f"{v}_mean",data=mean[i])
            f.create_dataset(f"{v}_std",data=std[i])
            
def create_vars(file,isSig=False):
    f = h5py.File(file, "r+")
    
    jet_kinematics = f['jet_kinematics']
    jet1_extraInfo = f['jet1_extraInfo']
    jet2_extraInfo = f['jet2_extraInfo']
    
    # signed dEta
    if "signed_dEta" in f.keys():
        pass
    else:
        jet1_eta = jet_kinematics[:,3]
        jet2_eta = jet_kinematics[:,7]
        signed_dEta = jet1_eta - jet2_eta
        #print("Making signed_dEta")
        f.create_dataset("signed_dEta",data=signed_dEta)
    
    if isSig:
        jet1_jme = f["jet1_JME_vars"]
        jet2_jme = f["jet2_JME_vars"]
    
    Mj1 = jet_kinematics[:,5]
    if isSig:
        Mj1_JES_up = jet1_jme[:,1]
        Mj1_JES_down = jet1_jme[:,3]
        Mj1_JER_up = jet1_jme[:,5]
        Mj1_JER_down = jet1_jme[:,7]
        Mj1_JMS_up = jet1_jme[:,8]
        Mj1_JMS_down = jet1_jme[:,9]
        Mj1_JMR_up = jet1_jme[:,10]
        Mj1_JMR_down = jet1_jme[:,11]
    
    Mj2 = jet_kinematics[:,9]
    if isSig:
        Mj2_JES_up = jet2_jme[:,1]
        Mj2_JES_down = jet2_jme[:,3]
        Mj2_JER_up = jet2_jme[:,5]
        Mj2_JER_down = jet2_jme[:,7]
        Mj2_JMS_up = jet2_jme[:,8]
        Mj2_JMS_down = jet2_jme[:,9]
        Mj2_JMR_up = jet2_jme[:,10]
        Mj2_JMR_down = jet2_jme[:,11]
    
    jet1_pt = jet_kinematics[:,2]
    if isSig:
        jet1_pt_JES_up = jet1_jme[:,0]
        jet1_pt_JES_down = jet1_jme[:,2]
        jet1_pt_JER_up = jet1_jme[:,4]
        jet1_pt_JER_down = jet1_jme[:,6]
    
    jet2_pt = jet_kinematics[:,6]
    if isSig:
        jet2_pt_JES_up = jet2_jme[:,0]
        jet2_pt_JES_down = jet2_jme[:,2]
        jet2_pt_JER_up = jet2_jme[:,4]
        jet2_pt_JER_down = jet2_jme[:,6]
    
    ##### JET 1 VARIABLES #####
    jet1_tau1 = jet1_extraInfo[:,0]
    jet1_tau2 = jet1_extraInfo[:,1]
    jet1_tau3 = jet1_extraInfo[:,2]
    jet1_tau4 = jet1_extraInfo[:,3]
    
    if "jet1_rho" in f.keys():
        #print("Skipping jet1_rho, exists")
        pass
    else:
        jet1_rho = Mj1/jet1_pt
        #print("Making jet1_rho")
        f.create_dataset("jet1_rho",data=jet1_rho)
    
    if isSig:
        if "jet1_rho_JES_up" in f.keys():
            #print("Skipping jet1_rho_JES_up, exists")
            pass
        else:
            jet1_rho_JES_up = Mj1_JES_up/jet1_pt_JES_up
            #print("Making jet1_rho_JES_up")
            f.create_dataset("jet1_rho_JES_up",data=jet1_rho_JES_up)

        if "jet1_rho_JES_down" in f.keys():
            #print("Skipping jet1_rho_JES_down, exists")
            pass
        else:
            jet1_rho_JES_down = Mj1_JES_down/jet1_pt_JES_down
            #print("Making jet1_rho_JES_down")
            f.create_dataset("jet1_rho_JES_down",data=jet1_rho_JES_down)

        if "jet1_rho_JER_up" in f.keys():
            #print("Skipping jet1_rho_JER_up, exists")
            pass
        else:
            jet1_rho_JER_up = Mj1_JER_up/jet1_pt_JER_up
            #print("Making jet1_rho_JER_up")
            f.create_dataset("jet1_rho_JER_up",data=jet1_rho_JER_up)

        if "jet1_rho_JER_down" in f.keys():
            #print("Skipping jet1_rho_JER_down, exists")
            pass
        else:
            jet1_rho_JER_down = Mj1_JER_down/jet1_pt_JER_down
            #print("Making jet1_rho_JER_down, exists")
            f.create_dataset("jet1_rho_JER_down",data=jet1_rho_JER_down)

        if "jet1_rho_JMS_up" in f.keys():
            #print("Skipping jet1_rho_JMS_up, exists")
            pass
        else:
            jet1_rho_JMS_up = Mj1_JMS_up/jet1_pt
            f.create_dataset("jet1_rho_JMS_up",data=jet1_rho_JMS_up)

        if "jet1_rho_JMS_down" in f.keys():
            #print("Skipping jet1_rho_JMS_down, exists")
            pass
        else:
            jet1_rho_JMS_down = Mj1_JMS_down/jet1_pt
            f.create_dataset("jet1_rho_JMS_down",data=jet1_rho_JMS_down)

        if "jet1_rho_JMR_up" in f.keys():
            #print("Skipping jet1_rho_JMR_up, exists")
            pass
        else:
            jet1_rho_JMR_up = Mj1_JMR_up/jet1_pt
            f.create_dataset("jet1_rho_JMR_up",data=jet1_rho_JMR_up)

        if "jet1_rho_JMR_down" in f.keys():
            #print("Skipping jet1_rho_JMR_down, exists")
            pass
        else:
            jet1_rho_JMR_down = Mj1_JMR_down/jet1_pt
            f.create_dataset("jet1_rho_JMR_down",data=jet1_rho_JMR_down)
    
    if "jet1_tau21" in f.keys():
        #print("Skipping jet1_tau21, exists")
        pass
    else:
        jet1_tau21 = jet1_tau2 / jet1_tau1
        jet1_tau21[~np.isfinite(jet1_tau21)] = 1.0
        f.create_dataset("jet1_tau21",data=jet1_tau21)
        
    if "jet1_tauS" in f.keys():
        #print("Skipping jet1_tauS, exists")
        pass
    else:
        jet1_tauS = np.sqrt(jet1_tau21) / jet1_tau1
        jet1_tauS[~np.isfinite(jet1_tauS)] = 1.0
        f.create_dataset("jet1_tauS",data=jet1_tauS)
        f.create_dataset("jet1_logTauS",data=np.log(1+jet1_tauS))
    
    if "jet1_tau32" in f.keys():
        #print("Skipping jet1_tau32, exists")
        pass
    else:
        jet1_tau32 = jet1_tau3 / jet1_tau2
        jet1_tau32[~np.isfinite(jet1_tau32)] = 1.0
        f.create_dataset("jet1_tau32",data=jet1_tau32)
        
    if "jet1_tau43" in f.keys():
        #print("Skipping jet1_tau43, exists")
        pass
    else:
        jet1_tau43 = jet1_tau4 / jet1_tau3
        jet1_tau43[~np.isfinite(jet1_tau43)] = 1.0
        f.create_dataset("jet1_tau43",data=jet1_tau43)
        
    if "jet1_pb" in f.keys():
        #print("Skipping jet1_pb, exists")
        pass
    else:
        jet1_btag = jet1_extraInfo[:,5]
        jet1_btag[jet1_btag<0] = 0
        f.create_dataset("jet1_pb",data=jet1_btag)
        
    if "jet1_pb2" in f.keys():
        #print("Skipping jet1_pb2, exists")
        pass
    else:
        jet1_btag = jet1_extraInfo[:,5]
        rng = np.random.default_rng(seed=4793)
        nSamp = len(jet1_btag[jet1_btag<0])
        jet1_btag[jet1_btag<0] += rng.normal(loc=0,scale=0.1,size=nSamp)
        f.create_dataset("jet1_pb2",data=jet1_btag)
        
    ##### JET 2 VARIABLES #####
    jet2_tau1 = jet2_extraInfo[:,0]
    jet2_tau2 = jet2_extraInfo[:,1]
    jet2_tau3 = jet2_extraInfo[:,2]
    jet2_tau4 = jet2_extraInfo[:,3]
    
    if "jet2_rho" in f.keys():
        #print("Skipping jet2_rho, exists")
        pass
    else:
        jet2_rho = Mj2/jet2_pt
        f.create_dataset("jet2_rho",data=jet2_rho)
    
    if isSig:
        if "jet2_rho_JES_up" in f.keys():
            #print("Skipping jet2_rho_JES_up, exists")
            pass
        else:
            jet2_rho_JES_up = Mj2_JES_up/jet2_pt_JES_up
            f.create_dataset("jet2_rho_JES_up",data=jet2_rho_JES_up)

        if "jet2_rho_JES_down" in f.keys():
            #print("Skipping jet2_rho_JES_down, exists")
            pass
        else:
            jet2_rho_JES_down = Mj2_JES_down/jet2_pt_JES_down
            f.create_dataset("jet2_rho_JES_down",data=jet2_rho_JES_down)

        if "jet2_rho_JER_up" in f.keys():
            #print("Skipping jet2_rho_JER_up, exists")
            pass
        else:
            jet2_rho_JER_up = Mj2_JER_up/jet2_pt_JER_up
            f.create_dataset("jet2_rho_JER_up",data=jet2_rho_JER_up)

        if "jet2_rho_JER_down" in f.keys():
            #print("Skipping jet2_rho_JER_down, exists")
            pass
        else:
            jet2_rho_JER_down = Mj2_JER_down/jet2_pt_JER_down
            f.create_dataset("jet2_rho_JER_down",data=jet2_rho_JER_down)

        if "jet2_rho_JMS_up" in f.keys():
            #print("Skipping jet2_rho_JMS_up, exists")
            pass
        else:
            jet2_rho_JMS_up = Mj2_JMS_up/jet2_pt
            f.create_dataset("jet2_rho_JMS_up",data=jet2_rho_JMS_up)

        if "jet2_rho_JMS_down" in f.keys():
            #print("Skipping jet2_rho_JMS_down, exists")
            pass
        else:
            jet2_rho_JMS_down = Mj2_JMS_down/jet2_pt
            f.create_dataset("jet2_rho_JMS_down",data=jet2_rho_JMS_down)

        if "jet2_rho_JMR_up" in f.keys():
            #print("Skipping jet2_rho_JMR_up, exists")
            pass
        else:
            jet2_rho_JMR_up = Mj2_JMR_up/jet2_pt
            f.create_dataset("jet2_rho_JMR_up",data=jet2_rho_JMR_up)

        if "jet2_rho_JMR_down" in f.keys():
            #print("Skipping jet2_rho_JMR_down, exists")
            pass
        else:
            jet2_rho_JMR_down = Mj2_JMR_down/jet2_pt
            f.create_dataset("jet2_rho_JMR_down",data=jet2_rho_JMR_down)
    
    if "jet2_tau21" in f.keys():
        #print("Skipping jet2_tau21, exists")
        pass
    else:
        jet2_tau21 = jet2_tau2 / jet2_tau1
        jet2_tau21[~np.isfinite(jet2_tau21)] = 1.0
        f.create_dataset("jet2_tau21",data=jet2_tau21)
        
    if "jet2_tauS" in f.keys():
        #print("Skipping jet2_tauS, exists")
        pass
    else:
        jet2_tauS = np.sqrt(jet2_tau21) / jet2_tau1
        jet2_tauS[~np.isfinite(jet2_tauS)] = 1.0
        f.create_dataset("jet2_tauS",data=jet2_tauS)
        f.create_dataset("jet2_logTauS",data=np.log(1+jet2_tauS))
    
    if "jet2_tau32" in f.keys():
        #print("Skipping jet2_tau32, exists")
        pass
    else:
        jet2_tau32 = jet2_tau3 / jet2_tau2
        jet2_tau32[~np.isfinite(jet2_tau32)] = 1.0
        f.create_dataset("jet2_tau32",data=jet2_tau32)
        
    if "jet2_tau43" in f.keys():
        #print("Skipping jet2_tau43, exists")
        pass
    else:
        jet2_tau43 = jet2_tau4 / jet2_tau3
        jet2_tau43[~np.isfinite(jet2_tau43)] = 1.0
        f.create_dataset("jet2_tau43",data=jet2_tau43)

    if "jet2_pb" in f.keys():
        #print("Skipping jet2_pb, exists")
        pass
    else:
        jet2_btag = jet2_extraInfo[:,5]
        jet2_btag[jet2_btag<0] = 0
        f.create_dataset("jet2_pb",data=jet2_btag)
        
    if "jet2_pb2" in f.keys():
        #print("Skipping jet2_pb2, exists")
        pass
    else:
        jet2_btag = jet2_extraInfo[:,5]
        rng = np.random.default_rng(seed=1204)
        nSamp = len(jet2_btag[jet2_btag<0])
        jet2_btag[jet2_btag<0] += rng.normal(loc=0,scale=0.1,size=nSamp)
        f.create_dataset("jet2_pb2",data=jet2_btag)
    

def extract_trainVars(sample,year,sideband=False,nev_max=-1,additional=None):
    f = h5py.File(sample, "r")
    fname = str(sample.split("/")[-1])

    jet_kinematics = f['jet_kinematics']
    jet1_extraInfo = f['jet1_extraInfo']
    jet2_extraInfo = f['jet2_extraInfo']
    truth_label = f['truth_label'][:,0].reshape(-1,1)
    yearVals = f['event_info'][:,6].reshape(-1,1)

    np.seterr(invalid = 'ignore')

    delta_eta = np.reshape(jet_kinematics[:,1],(-1,1))

    Mjj = np.reshape(jet_kinematics[:,0], (-1,1))
    Mj1 = np.reshape(jet_kinematics[:,5], (-1,1))
    Mj2 = np.reshape(jet_kinematics[:,9], (-1,1))

    jet1_pt = np.reshape(jet_kinematics[:,2], (-1,1))
    jet2_pt = np.reshape(jet_kinematics[:,6], (-1,1))
    jet3_pt = np.reshape(jet_kinematics[:,10], (-1,1))
    
    jet1_rho = Mj1/jet1_pt
    jet2_rho = Mj2/jet2_pt

    jet1_tau1 = np.reshape(jet1_extraInfo[:,0], (-1,1))
    jet1_tau2 = np.reshape(jet1_extraInfo[:,1], (-1,1))
    jet1_tau3 = np.reshape(jet1_extraInfo[:,2], (-1,1))
    jet1_tau4 = np.reshape(jet1_extraInfo[:,3], (-1,1))
    jet1_lsf3 = np.reshape(jet1_extraInfo[:,4], (-1,1))
    jet1_numpfconst = np.reshape(jet1_extraInfo[:,6],(-1,1))

    jet1_tau21 = jet1_tau2 / jet1_tau1
    jet1_tau32 = jet1_tau3 / jet1_tau2
    jet1_tau43 = jet1_tau4 / jet1_tau3
    jet1_sqrt_tau21 = np.sqrt(jet1_tau21) / jet1_tau1
    # fix A/0 or 0/0 to 1
    jet1_tau21[~np.isfinite(jet1_tau21)] = 1.0
    jet1_tau32[~np.isfinite(jet1_tau32)] = 1.0
    jet1_tau43[~np.isfinite(jet1_tau43)] = 1.0
    jet1_sqrt_tau21[~np.isfinite(jet1_sqrt_tau21)] = 1.0
    # make log tauS
    jet1_log_tauS = np.log(1+jet1_sqrt_tau21)

    jet2_tau1 = np.reshape(jet2_extraInfo[:,0], (-1,1))
    jet2_tau2 = np.reshape(jet2_extraInfo[:,1], (-1,1))
    jet2_tau3 = np.reshape(jet2_extraInfo[:,2], (-1,1))
    jet2_tau4 = np.reshape(jet2_extraInfo[:,3], (-1,1))
    jet2_lsf3 = np.reshape(jet2_extraInfo[:,4], (-1,1))
    jet2_numpfconst = np.reshape(jet2_extraInfo[:,6],(-1,1))

    jet2_tau21 = jet2_tau2 / jet2_tau1
    jet2_tau32 = jet2_tau3 / jet2_tau2
    jet2_tau43 = jet2_tau4 / jet2_tau3
    jet2_sqrt_tau21 = np.sqrt(jet2_tau21) / jet2_tau1
    # fix A/0 or 0/0 to 1
    jet2_tau21[~np.isfinite(jet2_tau21)] = 1.0
    jet2_tau32[~np.isfinite(jet2_tau32)] = 1.0
    jet2_tau43[~np.isfinite(jet2_tau43)] = 1.0
    jet2_sqrt_tau21[~np.isfinite(jet2_sqrt_tau21)] = 1.0
    # make log tauS
    jet2_log_tauS = np.log(1+jet2_sqrt_tau21)

    # Fixing the discrete -1 and -2 values of b tag score
    jet1_btag = jet1_extraInfo[:,5]
    jet2_btag = jet2_extraInfo[:,5]
    #n1,n2 = np.count_nonzero(jet1_btag < 0), np.count_nonzero(jet2_btag < 0)
    #rng1, rng2 = np.random.default_rng(7291), np.random.default_rng(8137)
    #jet1_btag[jet1_btag<0] += rng1.normal(scale=0.1,size=n1)
    #jet2_btag[jet2_btag<0] += rng2.normal(scale=0.1,size=n2)
    jet1_btag[jet1_btag<0] = 0
    jet2_btag[jet2_btag<0] = 0
    jet1_btag = jet1_btag.reshape(-1,1)
    jet2_btag = jet2_btag.reshape(-1,1)
    
    # file name for bookkeeping
    if 'batch' in fname:
        bnum = int(re.search('(\d+).h5',fname).group(1))
    else:
        bnum = -1
    fname = np.array([[bnum]]).repeat(jet1_btag.shape[0],axis=0)

    all_vars = [Mj1, jet1_logMass, jet1_rho, jet1_tau21, jet1_tau32, jet1_tau43, jet1_sqrt_tau21, jet1_log_tauS, jet1_btag, jet1_numpfconst,
                Mj2, jet2_logMass, jet2_rho, jet2_tau21, jet2_tau32, jet2_tau43, jet2_sqrt_tau21, jet2_log_tauS, jet2_btag, jet2_numpfconst,
                fname]
    varNames = [r'$M_{j1}$',r'$\log(M_{j1})$',r'Jet 1 $\rho$',r'Jet 1 $\tau_{21}$', r'Jet 1 $\tau_{32}$', r'Jet 1 $\tau_{43}$', r'Jet 1 $\tau_s$', r'Jet 1 $\log(1+\tau_s)$', r'Jet 1 $P_b$', r'Jet 1 $n_{pf}$',
                r'$M_{j2}$',r'$\log(M_{j2})$',r'Jet 2 $\rho$',r'Jet 2 $\tau_{21}$', r'Jet 2 $\tau_{32}$', r'Jet 2 $\tau_{43}$', r'Jet 2 $\tau_s$', r'Jet 2 $\log(1+\tau_s)$', r'Jet 2 $P_b$', r'Jet 2 $n_{pf}$',
                'File Name']
    
    if additional is not None:
        for vname in additional:
            if "/" in h5_map[vname]:
                bname,ind = h5_map[vname].split("/")[0], int(h5_map[vname].split("/")[1])
                arr = f[bname][:,ind].reshape(-1,1)
            else:
                bname = h5_map[vname]
                arr = f[bname][()].reshape(-1,1)
            if arr.shape[0] == 1:
                arr = np.tile(arr,(Mjj.shape[0],1))
            all_vars.append(arr)
            varNames.append(varTitle_map[vname])

    data = all_vars
    
    indices = ((yearVals == year))[:,0]
    
    if sideband:
        cutval = 2 * jet1_pt * jet2_pt * (np.cosh(delta_eta)+1) / (Mjj*Mjj)
        jet_asymm = np.abs((jet1_pt-jet2_pt)/(jet1_pt+jet2_pt))
        indices = indices & ( (delta_eta > 2.0)
                          & (delta_eta < 2.5)
                          & (jet1_pt > 300)
                          & (jet2_pt > 300)
                          & (jet3_pt < 300)
                          & ( (cutval > 1.0) | (cutval < 0.95) | (jet_asymm > 0.1 ) ) )[:,0]
    print("keeping {0} events of {1} -- {2} cut".format(np.count_nonzero(indices),np.count_nonzero(yearVals==year),np.count_nonzero(yearVals==year)-np.count_nonzero(indices)))

    data = [d[indices] for d in data]
    masses = Mjj[indices]

    if nev_max > 0 and norm_data.shape[0] > nev_max:
        norm_data = norm_data[:nev_max]
        masses = masses[:nev_max]

    return data, masses, varNames