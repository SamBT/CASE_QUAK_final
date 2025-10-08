import helper_functions as hf
from imports import *
import importlib

num_available_cpus = multiprocessing.cpu_count()

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else print ('cpu')

torch.set_num_threads(num_available_cpus)

import qRegNN as qrnn

Xtrain = 2*np.pi*np.random.rand(100000).reshape(-1,1)
Ytrain = np.random.normal(loc=np.sin(Xtrain),scale=0.1)
quantiles = np.array([0.05,0.5,0.95])
Ytrain = Ytrain.repeat(len(quantiles),axis=1)

Xtrain = torch.tensor(Xtrain).to(device)
Ytrain = torch.tensor(Ytrain).to(device)
quantiles = torch.tensor(quantiles).to(device)
model = qrnn.qreg(1,quantiles,widths=[30,30,30])
model = model.double()
model = model.cuda()
qrnn.train(model,Xtrain,Ytrain,lr=3e-4,n_epoch=3000)