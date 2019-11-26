import h5py
import scipy.io as sio
import os
import glob
import numpy as np
filepath="./results/result_p0_clinical1_window23_327487.mat"
#f = h5py.File(filepath,'r')
import pylab as plt


# The data are transposed (meaning that the first variable is read as last), so we have to transpose back (using the .T)
#acc=np.array(f['vect']).T 

path = 'results/'


no_runs=50
acc_mat=np.zeros((no_runs,2))
for permi in range(2):

            search_str=path + "**/result_p"+str(permi)+"_clinical1_window0_*.mat"
            
            files = [f for f in glob.glob(search_str, recursive=True)]
            
            for fi in range(no_runs):

                mat = sio.loadmat(files[fi])
                acc=mat['vect']
                acc_mat[fi,permi]=acc
                

sio.savemat("acc.mat", {'vect':acc_mat})        

#plt.plot(acc_mat)
#plt.show()
