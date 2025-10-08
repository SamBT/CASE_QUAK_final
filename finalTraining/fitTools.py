import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

def massArraySlicer(contents, binval, mass, loss1, loss2, masspoint, contentmax, binstoadd):
    rbins = np.linspace(0, 25, 10)
    abins = np.linspace(0,(0.4 + 0.02*binstoadd)*np.pi, 20+binstoadd)
    nullXBounds = []
    nullYBounds = []
    contentsmin = contents.min()
    newmax = contents.max() + 1
    
    if(contentsmin==contentmax):
        return 0,[],True
    
    for a in range(20+binstoadd-1):
        for r in range(10-1):
            if(contents[a,r]==contentsmin and contents[a,r]<contentmax):
                nullXBounds.append(a)
                nullYBounds.append(r)
                contents[a,r]=newmax


    angles = np.arctan(loss1/loss2)
    radii = np.sqrt(loss1*loss1 + loss2*loss2)
    thebool = []
    for b in range(len(nullXBounds)):
        thebool.append( (angles > abins[nullXBounds[b]]) & (angles < abins[nullXBounds[b]+1]) & (radii > rbins[nullYBounds[b]]) & (radii < rbins[nullYBounds[b]+1]) )

    print(binval, contentsmin, len(thebool), contentmax, contentsmin)
    if(len(thebool) == 0):
        return 0,[],False
    
    thefinalbool = thebool[0]
    for b in range(1,len(nullXBounds)):
        thefinalbool = thefinalbool | thebool[b]

    masstoreturn = mass[thefinalbool]
    acceptMass = (masstoreturn > (masspoint-700)) & (masstoreturn < (masspoint+300))
    massinwindow = masstoreturn[acceptMass]
    return len(massinwindow), masstoreturn,False

def makething(fname,masspoint,windowsize,l1,l2,lab):
    # loss1 = signal loss, loss2 = bkg_loss
    f1 = h5py.File(fname)
    mass = f1['mjj'][()][:,0]
    loss1 = f1[l1][()]
    loss2 = f1[l2][()]
    label = lab*np.ones(len(loss1))


    window_low = mass > (masspoint-(windowsize+200))
    window_high = mass < (masspoint+(windowsize-200))
    massWindowIN = window_low & window_high & (loss2 > 8)
    print('num in window?', len(mass[massWindowIN]))

    window_low = mass > (masspoint-(windowsize+200))
    window_high = mass < (masspoint+(windowsize-200))
    massWindow = window_low & window_high & (loss2 > 8)
    for a in range(6):
        rbins = np.linspace(0, 25, 10)
        abins = np.linspace(0,(0.4 + 0.02*a)*np.pi, 20+a)
        loss2out = loss2[massWindow] - 8
        loss1out = loss1[massWindow]
        angles = np.arctan(loss1out/loss2out)
        radii = np.sqrt(loss1out*loss1out + loss2out*loss2out)
        hist2, _, _ = np.histogram2d(list(angles), list(radii), bins=(abins, rbins))
        A, R = np.meshgrid(abins, rbins)
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        pc = ax.pcolormesh(A, R, hist2.T, cmap="magma_r", norm=LogNorm())
        fig.colorbar(pc)
        plt.close()
        print(a, hist2.sum())
        if(hist2.sum()>5000):
            break

    binstoadd = a

    massWindow = (label == 1) & (loss2 > 8)
    if(len(loss2[massWindow])>0):
        rbins = np.linspace(0, 25, 10)
        abins = np.linspace(0,(0.4 + 0.02*binstoadd)*np.pi, 20+binstoadd)
        loss2out = loss2[massWindow] - 8
        loss1out = loss1[massWindow]
        angles = np.arctan(loss1out/loss2out)
        radii = np.sqrt(loss1out*loss1out + loss2out*loss2out)
        hist, _, _ = np.histogram2d(list(angles), list(radii), bins=(abins, rbins))
        A, R = np.meshgrid(abins, rbins)
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        pc = ax.pcolormesh(A, R, hist.T, cmap="magma_r", norm=LogNorm())
        fig.colorbar(pc)
        plt.show()
        plt.close()


    print(mass)
    massMin = mass > 1000
    testWindow1 = mass < (masspoint-500)
    testWindow2 = mass > (masspoint+500)
    massWindow = massMin & (testWindow1 | testWindow2)
    massWindow = massMin & (testWindow1 | testWindow2) & (loss2 > 8)

    testWindow1 = (mass > (masspoint-(windowsize+600))) & (mass < (masspoint-(windowsize+200)))
    testWindow2 = (mass > (masspoint+(windowsize-200))) & (mass < (masspoint+(windowsize+600)))
    massWindow = massMin & (testWindow1 | testWindow2)
    massWindow = massMin & (testWindow1 | testWindow2) & (loss2 > 8)

    print(len(loss1))
    print('total bkg = ', len(loss1[massWindow]))

    # define binning
    rbins = np.linspace(0, 25, 10)
    abins = np.linspace(0,(0.4 + 0.02*binstoadd)*np.pi, 20+binstoadd)

    #calculate histogram
    loss2out = loss2[massWindow] - 8
    loss1out = loss1[massWindow]

    angles = np.arctan(loss1out/loss2out)
    radii = np.sqrt(loss1out*loss1out + loss2out*loss2out)
    print(loss2out)
    print(loss1out)
    print(angles)
    print(radii)
    hist, _, _ = np.histogram2d(list(angles), list(radii), bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)

    # plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

    print(len(hist), len(hist[0]))
    print(len(hist.T), len(hist.T[0]))
    pc = ax.pcolormesh(A, R, hist.T, cmap="magma_r", norm=LogNorm())
    fig.colorbar(pc)

    plt.close()

    totalmassinwindow = 0
    themassarray = []
    themax = hist.max()
    for b in range(1000):
        sliced = massArraySlicer(hist, b, mass[loss2>8], loss1[loss2>8], loss2[loss2>8]-8, masspoint, themax, binstoadd)
        totalmassinwindow += sliced[0]
        themassarray = themassarray + list(sliced[1])
        print(totalmassinwindow)
        if(totalmassinwindow>max(100, 100.*np.power((masspoint/1000.)-3, 4)) or sliced[2]):
            break

    print(len(themassarray))
    plt.hist(themassarray, bins=100, range=[1600,6500])
    plt.show()
