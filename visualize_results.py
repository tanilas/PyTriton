import scipy.io as sio
import pylab as plt
import numpy as np


filepath="acc.mat"
mat = sio.loadmat(filepath)
arr=mat['vect']

clini=1
permi=1
sig1=np.mean(arr[:,:,1,0],axis=1)
ssig1=np.std(arr[:,:,1,0],axis=1)
sig2=np.mean(arr[:,:,1,1],axis=1)
ssig2=np.std(arr[:,:,1,1],axis=1)
plt.plot(sig1,"b");plt.plot(sig1+ssig1,"b:")
plt.plot(sig2,"r");plt.plot(sig2+ssig2,"r:")

plt.show()