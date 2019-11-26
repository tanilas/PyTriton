import scipy.io as sio
import pylab as plt
import numpy as np


filepath="acc.mat"
mat = sio.loadmat(filepath)
arr=mat['vect']

print("Mean 0: "+str(np.mean(arr[:,0])))
print("Mean 1: "+str(np.mean(arr[:,1])))
plt.plot(arr[:,0],"b")
plt.plot(arr[:,1],"r")

plt.show()