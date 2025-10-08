from imports import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class qreg(nn.Module):
    def __init__(self,input_dim,quantiles,widths=[30,30,30],act=nn.ReLU(),final_act=nn.Identity(),p=0.0,final_scale=None):
        super().__init__()
        self.quantiles = quantiles
        #self.final_scale = nn.Parameter(final_scale*torch.ones(len(self.quantiles)),requires_grad=True)
        self.final_scale = final_scale
        #print('final scale = ',self.final_scale)
        out_dim = len(self.quantiles)
        layers = [nn.Linear(input_dim,widths[0]),act]
        for i in range(len(widths)-1):
            if p > 0:
                layers.append(nn.Dropout(p=p))
            layers.append(nn.Linear(widths[i],widths[i+1]))
            layers.append(act)
            #layers.append(nn.BatchNorm1d(widths[i+1]))
        layers.append(nn.Linear(widths[-1],out_dim))
        layers.append(final_act)
        #if final_scale is not None:
        #    layers.append(ScaleLayer(init_value=final_scale))
        #layers = nn.ModuleList(layers)
        self.layers = layers
        self.transform = nn.Sequential(*layers)

    def forward(self,x):
        return self.final_scale*self.transform(x)
    

class ScaleLayer(nn.Module):

    def __init__(self, init_value=10):
        super(ScaleLayer,self).__init__()
        self.scale = nn.Parameter(torch.rand(1))
        self.scale.requires_grad = True

    def forward(self, x):
        return self.scale*x

class qreg_linear(nn.Module):
    def __init__(self,input_dim,quantiles,widths=[30,30,30],act=nn.ReLU(),final_act=nn.Identity(),p=0.0):
        super().__init__()
        self.quantiles = quantiles
        out_dim = len(self.quantiles)
        layers = [nn.Linear(input_dim,out_dim)]
        layers.append(final_act)
        self.transform = nn.Sequential(*layers)

    def forward(self,x):
        return self.transform(x)

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(MyDataSet, self).__init__()
        # store the raw tensors
        self._x = x
        self._y = y

    def __len__(self):
        # a DataSet must know it size
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index,:]
        y = self._y[index,:]
        return x, y    
    
def qLoss(Ypred,Ytrue,quantiles,alpha=0.1):
    qcast = quantiles.reshape(1,-1).repeat(Ytrue.shape[0],1)
    err = Ytrue - Ypred
    loss1 = torch.mean(torch.maximum(qcast*err,(qcast-1)*err),dim=1)
    loss1 = torch.mean(loss1)
    zeros = torch.zeros((Ypred.shape[0],Ypred.shape[1]-1))
    loss2 = torch.mean(torch.mean(torch.maximum(Ypred[:,:-1]-Ypred[:,1:],zeros),dim=1))
    ymax,ymin = torch.max(Ypred), torch.min(Ypred)
    #z2 = torch.zeros(Ypred.shape)
    #loss3 = torch.mean(torch.mean(torch.maximum(Ypred-ymax,z2),dim=1)) + torch.mean(torch.mean(torch.maximum(ymin-Ypred,z2),dim=1))
    del err,zeros
    return loss1+alpha*loss2

def train(model,Xtrain,Ytrain,n_epoch=1000,lr=1e-4,check=100,name='nothing',alpha=0.1,wd=0.0,patience=50):
    Xtrain = torch.tensor(Xtrain).to(device)
    Ytrain = torch.tensor(Ytrain).to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    losses = []
    min_loss = 99999
    epochs_no_improve = 0
    for i in range(n_epoch):
        epochs_no_improve += 1
        Ypred = model(Xtrain)
        loss = qLoss(Ypred,Ytrain,model.quantiles,alpha=alpha)
        losses.append(loss.detach().cpu().numpy())
        if loss.detach().cpu().numpy() < min_loss:
            min_loss = loss.detach().cpu().numpy()
            torch.save(model,"QR_models/{0}.pt".format(name))
            epochs_no_improve = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del Ypred,loss
        if (i+1)%check == 0:
            print("Epoch {}, Loss: {}".format(i+1,losses[-1]))
        if epochs_no_improve==patience:
            print("stopping early at epoch {0}".format(i+1))
            break
    plt.plot(np.arange(len(losses)),losses)
    
def train_gen_v2(model,X,Y,bs=10000,n_epoch=1000,lr=1e-4,check=100,name='nothing',alpha=0.1,wd=0.0,patience=50):
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    losses = []
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    min_loss = 9999999
    epochs_no_improve = 0
    for i in range(n_epoch):
        perms = torch.randperm(len(X)).split(bs)
        for perm in perms:
            Xtrain = X[perm]
            Ytrain = Y[perm]
            optimizer.zero_grad()
            Ypred = model(Xtrain)
            loss = qLoss(Ypred,Ytrain,model.quantiles,alpha=alpha)
            if loss.detach().cpu().numpy() < min_loss:
                min_loss = loss.detach().cpu().numpy()
                torch.save(model,"QR_models/{0}.pt".format(name))
                epochs_no_improve = 0
            loss.backward()
            optimizer.step()
            del Ypred,loss
        epochs_no_improve += 1
        losses.append(min_loss)
        if (i+1)%check == 0:
            print("Epoch {}, Loss: {}".format(i+1,losses[-1]))
        if epochs_no_improve==patience:
            print("stopping early at epoch {0}".format(i+1))
            break
    plt.plot(np.arange(len(losses)),losses)

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
            
@guvectorize([(float32[:],float32[:,:],float32[:,:],float32[:])], '(n),(n,k),(n,k)->(n)')
def correctAll(Y,qbkg,qdata,Yout):
    N,M = qbkg.shape
    bad = 0
    bad_bkg = 0
    bad_data = 0
    underflow = 0
    overflow = 0
    good = 0
    ymin = np.quantile(Y,100/N)
    ymax = np.quantile(Y,(N-100)/N)
    for i in range(N):
        qb = qbkg[i]
        qd = qdata[i]
        y = Y[i]
        
        # veto events with bad (non-increasing) quantiles
        badOrder_bkg = np.any(qb[1:]-qb[:-1]<0)
        badOrder_data = np.any(qd[1:]-qd[:-1]<0)
        if badOrder_bkg or badOrder_data:
            Yout[i] = y
            bad += 1
            bad_bkg += int(badOrder_bkg)
            bad_data += int(badOrder_data)
            continue
        
        if y < qb[0]:
            ybl = ymin
            ybh = qb[0]
            ydl = ymin
            ydh = qd[0]
            underflow += 1
        elif y >= qb[-1]:
            ybl = qb[-1]
            ybh = ymax
            ydl = qd[-1]
            ydh = ymax
            overflow += 1
        else:
            for j in range(M-1):
                if y >= qb[j] and y < qb[j+1]:
                    ybl = qb[j]
                    ybh = qb[j+1]
                    ydl = qd[j]
                    ydh = qd[j+1]
                    good += 1
        ycorr = ydl + ((ydh-ydl)/(ybh-ybl))*(y-ybl)
        Yout[i] = ycorr
    
    print(bad," Events with bad quantiles (",100*bad/N," %)")
    print("\t",bad_bkg," from bkg (",100*bad_bkg/N," %)")
    print("\t",bad_data," from data (",100*bad_data/N," %)")
    print(underflow," Underflow (",100*underflow/N," %)")
    print(overflow," Overflow (",100*overflow/N," %)")    

class chainedTrainer:
    def __init__(self,data,bkg,varNames,varOrder,varDict,width_tuples,quantileList,act=nn.ReLU(),final_act=nn.Identity(),final_scale=None,alpha=0.1,p=0.0,wd=0.0,xlims=None):      
        self.names = varNames
        self.data_dict = {n:data[:,self.names.index(n)] for n in self.names}
        self.bkg_dict = {n:bkg[:,self.names.index(n)] for n in self.names}
        self.corrBkg_dict = {}
        self.alpha = alpha
        self.p = p
        self.wd = wd
        self.widths = []
        self.xlims = xlims
        self.final_scale = final_scale
        for wt in width_tuples:
            if type(wt) == list or type(wt) == str:
                self.widths.append(wt)
            else:
                self.widths.append([wt[0] for _ in range(wt[1])])
        self.act = act
        self.final_act = final_act
        self.varOrder = varOrder
        self.varDict = varDict
        self.quantileList = quantileList
        self.data_models = [None for _ in range(len(varOrder)-1)]
        self.bkg_models = [None for _ in range(len(varOrder)-1)]
        self.current = varOrder[1]
    
    def trainQuantiles(self,trainDict,i,nTrain,corrDict=None,lr=1e-3,n_epoch=1000,bs=10000,batched=False):
        control = self.varOrder[:i]
        regress = self.varOrder[i]
        quantiles = self.quantileList[i-1]
        
        if corrDict is None: # training on data, don't need to retrieve corrected val
            Xtrain = np.concatenate([trainDict[self.varDict[v]][:nTrain].reshape(-1,1) for v in control],axis=1)
        else:
            Xtrain = np.concatenate([corrDict[self.varDict[v]][:nTrain].reshape(-1,1) for v in control],axis=1)
        Ytrain = trainDict[self.varDict[regress]][:nTrain].reshape(-1,1).repeat(len(quantiles),axis=1)

        if self.widths[i-1] == 'linear':
            model = qreg_linear(Xtrain.shape[1],torch.tensor(quantiles).to(device),
                     widths=self.widths[i-1],act=self.act,final_act=self.final_act,p=self.p)
        else:
            model = qreg(Xtrain.shape[1],torch.tensor(quantiles).to(device),
                     widths=self.widths[i-1],act=self.act,final_act=self.final_act,
                         final_scale=self.final_scale,p=self.p)
        model = model.to(device)
        if not batched:
            train(model,Xtrain,Ytrain,n_epoch=n_epoch,lr=lr,check=100,name=regress,alpha=self.alpha,wd=self.wd)
        else:
            train_gen_v2(model,Xtrain,Ytrain,bs=bs,n_epoch=n_epoch,lr=lr,check=100,name=regress,alpha=self.alpha,wd=self.wd)
        
        del model
        model = torch.load("QR_models/{0}.pt".format(regress))
        model_cpu = model.to('cpu')
        return model_cpu
    
    def makeQuantilePlots(self,vert=False,log=False,data=True):
        if data:
            mods = self.data_models
        else:
            mods = self.bkg_models
        if data:
            source = self.data_dict
        else:
            source = self.bkg_dict
        plt.figure(figsize=(12,12))
        N = len(mods)
        nrow = N//3 if N%3==0 else N//3 + 1
        iMax = self.varOrder.index(self.current)
        for i,mod in enumerate(mods[:iMax]):
            mod = self.data_models[i].to(device)
            inp_data = torch.tensor(np.concatenate([source[self.varDict[n]].reshape(-1,1) for n in self.varOrder[:i+1]],axis=1)).to(device)
            out = mod(inp_data).detach().cpu().numpy()

            plt.subplot(nrow,3,i+1)
            h1 = plt.hist(source[self.varDict[self.varOrder[i+1]]],bins=np.arange(-20,20.05,step=0.05),histtype='step',density=True)
            mean = np.mean(out,axis=0)
            std = np.std(out,axis=0)
            if self.xlims is not None:
                plt.xlim(self.xlims[self.varOrder[i+1]])
            if log:
                plt.yscale('log')
            for j,m in enumerate(mean):
                sd = std[j]
                plt.axvline(m,linestyle='--',linewidth=0.6)
                plt.axvspan(m-sd,m+sd,alpha=0.2,color='C{0}'.format(i))
            plt.xlabel(self.varDict[self.varOrder[i+1]])
            """plt.subplot(122)
            plt.boxplot([out[:,i] for i in range(out.shape[1])],showfliers=True,flierprops={'marker':".",'markersize':1},vert=vert)"""

        plt.show()
        
    def plotQuantile(self,variable,dtype='data',xlim=None,vert=False,log=False):
        models = self.data_models if dtype=='data' else self.bkg_models
        data_dict = self.data_dict if dtype=='data' else self.bkg_dict
        idx = self.varOrder.index(variable)
        mod = models[idx-1].to(device)
        inp_data = torch.tensor(np.concatenate([data_dict[self.varDict[n]].reshape(-1,1) for n in self.varOrder[:idx]],axis=1)).to(device)
        out = mod(inp_data).detach().cpu().numpy()
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        h1 = plt.hist(data_dict[self.varDict[self.varOrder[idx]]],bins=np.arange(-20,20.05,step=0.05),histtype='step',density=True)
        plt.xlim([np.quantile(data_dict[self.varDict[self.varOrder[idx]]],0.0001)-0.2,
                  np.quantile(data_dict[self.varDict[self.varOrder[idx]]],0.9999)+0.2])
        mean = np.mean(out,axis=0)
        std = np.std(out,axis=0)
        if log:
            plt.yscale('log')
        for j,m in enumerate(mean):
            sd = std[j]
            plt.axvline(m,linestyle='--',linewidth=0.6)
            plt.axvspan(m-sd,m+sd,alpha=0.2,color='C0')
        if xlim is not None:
            plt.xlim(xlim)

        plt.subplot(122)
        plt.boxplot([out[:,i] for i in range(out.shape[1])],showfliers=True,flierprops={'marker':".",'markersize':1},vert=vert)
    
    def trainData(self,nTrain=200000,iMin=1,iMax=None,lr=1e-3,n_epoch=1000,bs=10000,batched=False,correctFirst=False):
        trainDict = self.data_dict
        endRange = iMax if iMax is not None else len(self.varOrder)
        for i in range(iMin,endRange):
            print("---------- Training Data, {0} ----------".format(self.varOrder[i]))
            mod = self.trainQuantiles(self.data_dict,i,nTrain,lr=lr,n_epoch=n_epoch,bs=bs,batched=batched)
            self.data_models[i-1] = mod
            
    def trainBkg(self,nTrain=200000,iMin=1,iMax=None,lr=1e-3,n_epoch=1000,bs=10000,batched=False,correctFirst=False):
        trainDict = self.bkg_dict
        firstVar = self.varDict[self.varOrder[0]]
        if iMin==1:
            if correctFirst:
                bkg = self.bkg_dict[firstVar].copy().astype('float32')
                data = self.data_dict[firstVar].copy().astype('float32')
                quantiles = [0.0005,0.001,0.005,0.01] + list(np.arange(0.05,1,0.05)) + [0.99,0.995,0.999,0.9995]
                qb = np.quantile(bkg,quantiles).reshape(1,-1).astype('float32')
                qd = np.quantile(data,quantiles).reshape(1,-1).astype('float32')
                bkg_corr = bkg.copy().astype('float32')
                correctAll(bkg,qb.repeat(len(bkg),axis=0),qd.repeat(len(bkg),axis=0),bkg_corr)
                self.corrBkg_dict[firstVar] = bkg_corr
            else:
                self.corrBkg_dict[firstVar] = trainDict[firstVar].copy()
        endRange = iMax if iMax is not None else len(self.varOrder)
        for i in range(iMin,endRange):
            print("---------- Training Bkg, {0} ----------".format(self.varOrder[i]))
            quantiles = self.quantileList[i-1]
            model = self.trainQuantiles(trainDict,i,nTrain,corrDict=self.corrBkg_dict,lr=lr,n_epoch=n_epoch,bs=bs,batched=batched).to(device)

            # Prepare tensors
            Xeval = torch.tensor(np.concatenate([self.corrBkg_dict[self.varDict[v]].reshape(-1,1) for v in self.varOrder[:i]],axis=1),dtype=torch.float32).to(device)
            Yorig = trainDict[self.varDict[self.varOrder[i]]]
            ymin,ymax = np.min(Yorig), np.max(Yorig)

            # load data model and evaluate both
            data_model = self.data_models[i-1].to(device)
            bkgQuantiles = model(Xeval).detach().cpu().numpy()
            dataQuantiles = data_model(Xeval).detach().cpu().numpy()
            
            # correct variable
            Yfinal = -999*np.ones_like(Yorig,dtype='float32')
            numOut = -1*np.ones_like(Yorig,dtype='int32')
            correctAll(Yorig,bkgQuantiles,dataQuantiles,Yfinal)

            # add corrected data to dictionary
            self.corrBkg_dict[self.varDict[self.varOrder[i]]] = Yfinal.copy().astype('float32')

            model_cpu = model.to('cpu')
            self.bkg_models[i-1] = model_cpu
            del model, data_model
            
    def changeWidth(self,variable,width_tuple):
        idx = self.varOrder.index(variable)-1
        self.widths[idx] = [width_tuple[0] for _ in range(width_tuple[1])]
    
    def changeQuantile(self,variable,quantList):
        idx = self.varOrder.index(variable)-1
        self.quantileList[idx] = quantList
        
    def retrainFrom(self,var,varStop=None,**kwargs):
        iMin = self.varOrder.index(var)
        iMax = self.varOrder.index(varStop)+1 if varStop is not None else None
        self.trainData(iMin=iMin,iMax=iMax,**kwargs)
        self.trainBkg(iMin=iMin,iMax=iMax,**kwargs)
        
    def trainCurrent(self,**kwargs):
        print("Current variable: {0}".format(self.current))
        iMin = self.varOrder.index(self.current)
        iMax = self.varOrder.index(self.current)+1
        self.trainData(iMin=iMin,iMax=iMax,**kwargs)
        self.trainBkg(iMin=iMin,iMax=iMax,**kwargs)
        
    def stepToNext(self):
        i = self.varOrder.index(self.current)
        self.current = self.varOrder[i+1]
        
    def stepBack(self):
        i = self.varOrder.index(self.current)
        self.current = self.varOrder[i-1]
        
    def stepTo(self,var):
        self.current = var
    
    def addVariable(self,var,width=[20,20],quantiles=np.arange(0.05,1,0.05)):
        if var not in self.varOrder:
            self.varOrder.append(var)
            self.widths.append(width)
            self.quantileList.append(quantiles)
            self.data_models.append(None)
            self.bkg_models.append(None)
        else:
            print("Error: variable already registered in sequence")
    
    def removeLastVar(self):
        self.varOrder = self.varOrder[:-1]
        self.widths = self.widths[:-1]
        self.quantileList = self.quantileList[:-1]
        self.data_models = self.data_models[:-1]
        self.bkg_models = self.bkg_models[:-1]
        
    def plotCorrected(self):
        for v in self.varOrder[:self.varOrder.index(self.current)+1]:
            vbkg = self.corrBkg_dict[self.varDict[v]]
            vbkg_old = self.bkg_dict[self.varDict[v]]
            vdata = self.data_dict[self.varDict[v]]
            xmin = np.min([np.quantile(vdata,0.0001),np.quantile(vbkg,0.0001),np.quantile(vbkg_old.min(),0.0001)])
            xmax = np.max([np.quantile(vdata,0.99999),np.quantile(vbkg,0.99999),np.quantile(vbkg_old.min(),0.99999)])
            bins = np.arange(-20,20.1,step=0.1)
            plt.figure(figsize=(16,6))

            plt.subplot(121)
            h1 = plt.hist(vdata,bins=bins,label='Data',histtype='step',density=True)
            h2 = plt.hist(vbkg,bins=bins,label='corrBkg',histtype='step',density=True)
            plt.xlabel(self.varDict[v],fontsize=14)
            if self.xlims is not None:
                plt.xlim(self.xlims[v])
            plt.gca().legend(loc='best')

            plt.subplot(122)
            h1 = plt.hist(vdata,bins=bins,label='Data',histtype='step',density=True)
            h2 = plt.hist(vbkg,bins=bins,label='corrBkg',histtype='step',density=True)
            plt.xlabel(self.varDict[v],fontsize=14)
            if self.xlims is not None:
                plt.xlim(self.xlims[v])
            plt.yscale('log')
            #plt.gca().legend(loc='best')
            