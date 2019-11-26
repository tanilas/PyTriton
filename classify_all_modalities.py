# srun --pty --gres gpu --constraint='kepler|pascal|volta' --mem-per-cpu 10G bash
# BL_5PCsConsN15MaxCorr_nonparamCompCount_covmat_36pt29hc_2mm_zscore_butterBandpass/
# /m/nbe/scratch/psykoosi/Jonatans_folder/Deep_NN/Triton_scripts/NN_files/fmri_rem12_clinical_48xFMRIpca_multipTRWsliding_windowavg.mat
# import pdb; pdb.set_trace()
import scipy.io as sio
import h5py
import pylab as plt
import numpy as np
import torch
import random
import math
from sklearn import svm
from sklearn import metrics
from statistics import mean 
from scipy import stats
import argparse


def classify(permutation=0,clinical_data=1,timewindow=1):
    #select file to analyse
    print("Is permutation: " + str(permutation))
    print("Including clinical data: "+str(clinical_data))
    print("Selected time window: "+str(timewindow))
    
    #filename='fmri_rem12_clinical_48xFMRIpca_multipTRWsliding_windowavg.mat' # filename
    filename='test_31_rem12_clinical_2xISC.mat'
    path='/m/nbe/scratch/psykoosi/Jonatans_folder/Deep_NN/Triton_scripts/NN_files/'

    filepath=path+filename
    ### LOADING DATA PART
    # Ths is the command to load the matlab files (the ones that are compressed with the -v7.3 argument)
    f = h5py.File(filepath,'r')
    # The data are transposed (meaning that the first variable is read as last), so we have to transpose back (using the .T)
    X=np.array(f['X']).T 
    Y=np.array(f['Y']).T
    X[np.isnan(X)]=0
    X=stats.zscore(X)
    X[np.isnan(X)]=0
    import pdb; pdb.set_trace()
    #X=0.0*np.random.randn(X.shape[0],X.shape[1])+Y;
    #import pdb; pdb.set_trace()
    modality_index=np.array(f['modality_index']).T
    clin_var_n=np.array(f['clinical_var_n'])
    data_sets=np.array(f['data_sets']) 
    set_sizes=np.array(f['set_sizes']) 
    data_set_n=np.array(f['data_set_n']) 
    data_set_n=int(data_set_n[0,0])
    clin_var_n=int(clin_var_n[0,0])

    class0_inds,val=np.where(Y==0)
    class1_inds,val=np.where(Y==1)

    ## Randomly permuting the classes so that they can be split
    class0_inds=np.random.permutation(class0_inds)
    class1_inds=np.random.permutation(class1_inds)

    numOfEqualSubjects=np.min((len(class0_inds),len(class1_inds)))

    all_inds=np.arange(numOfEqualSubjects)
    splitting_sub=int(numOfEqualSubjects*0.8)


    X_train1=X[class0_inds[np.arange(splitting_sub)],:]
    X_train2=X[class1_inds[np.arange(splitting_sub)],:]
    X_train=np.concatenate((X_train1,X_train2));

    X_test1=X[class0_inds[np.arange(splitting_sub,numOfEqualSubjects)],:]
    X_test2=X[class1_inds[np.arange(splitting_sub,numOfEqualSubjects)],:]
    X_test=np.concatenate((X_test1,X_test2));

    Y_train=np.concatenate((Y[class0_inds[np.arange(splitting_sub)],:],Y[class1_inds[np.arange(splitting_sub)],:]))
    Y_test=np.concatenate((Y[class0_inds[np.arange(splitting_sub,numOfEqualSubjects)],:],Y[class1_inds[np.arange(splitting_sub,numOfEqualSubjects)],:]))

    if (permutation==1):
        # Permute the labels of the training set
        Y_train=np.concatenate((Y[class0_inds[np.arange(splitting_sub)],:],Y[class1_inds[np.arange(splitting_sub)],:]))

    ## Here starts my stuff, splitting data into modalities and keeping only a few variables per modality
    num_of_modality=int(np.max(modality_index)) # Get the number of modalities as the max number in the vector
    data=np.array([]) # Make an empty matrix to add the data from different modalities
    how_many_inds=5 # How many variables to keep per modality
    variable_modality=np.array([]) # Keeping the indices for each of the updated variable_modality 
    index_variables=np.array([])

    # Get the indices for each modality and keep only a bunch of variables for each modality
    #modalities_to_include=np.arange(num_of_modality)+1
    #modalities_to_include=np.array([1]);
    #modalities_to_include=np.array([1,2]);

    include_clinical=clinical_data
    time_window_to_use=timewindow
    
    if include_clinical==0:
        modalities_to_include=np.array([time_window_to_use]);
    else:
        modalities_to_include=np.array([0,time_window_to_use]);

    if time_window_to_use==0:
        modalities_to_include=np.array([0]);

    print("Including modalities:")
    print(modalities_to_include)
    if permutation==1:
        Y_train=np.random.permutation(Y_train)                                                                                                       

    for i in range(np.alen(modalities_to_include)):
        inds,nonsense=np.where(modality_index==(modalities_to_include[i]+1)) # The second output (nonsense) just returns the values of these indices, pointless in our case.
        how_many_inds=np.alen(inds);
        index_variables=np.append(index_variables,inds[np.arange(how_many_inds)])
        variable_modality=np.append(variable_modality,np.ones(how_many_inds)*(modalities_to_include[i]))


    index_variables=index_variables.astype(np.int32)
    X_train=X_train[:,index_variables]
    X_test=X_test[:,index_variables]

    D_in=np.alen(index_variables) # How many input variables we have in the end
    D_out=2 # We do a classification task of two classes, the second one is a 1 - Y
    N=data.shape[0] # How many data samples we have

    W_mask=np.zeros((index_variables.shape[0],np.alen(modalities_to_include)))
    
    for i in np.arange(np.alen(modalities_to_include)):
        inds=np.where(variable_modality==modalities_to_include[i])
        W_mask[inds,i]=1

    ''' (This is a multiline comment)
    The for loop above creates a Weight mask, telling to the classifier which variable is mapped to which modality
    This constrols that each variable is connected only to the node corresponding for each modality

    variable 1 of modality 1 [ 1 0 0]
    variable 2 of modality 1 [ 1 0 0 ]
    ...
    variable 1 of modality 2 [ 0 1 0]
    variable 2 of modality 1 [ 0 1 0]
    ...
    variable 1 of modality 3 [ 0 0 1]

    etc.


    '''
    ## NOW WE ARE READY TO SET UP THE NEURAL NETWORK

    train_inds=range(N) 
    test_inds=range(N)

    # If the machine has capability to run on GPU ('cuda') then do that, otherwise run it on a CPU (setting the device as GPU or CPU for later)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    labels_train=np.array((Y_train,1-Y_train)).transpose((1,0,2))
    labels_test=np.array((Y_test,1-Y_test)).transpose((1,0,2))
    WW=torch.from_numpy(W_mask.T).float().to(device) # We turn the Weight mask into a PyTorch Tensor to be able to multiply with the weight matrix later on
    x=torch.from_numpy(X_train).float() # The input data
    y=torch.from_numpy(labels_train).float() # The labels

    x_test=torch.from_numpy(X_test).float() # The input data
    y_test=torch.from_numpy(labels_test).float() # The labels

    # Binary Cross Entropy Loss
    loss_fn=torch.nn.BCELoss()

    # Mean Square Error Loss function
    #loss_fn=torch.nn.MSELoss()
    # Here we define the model structure:
    # A Linear layer, inputsxoutputs followed by a softmax activation function
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, np.alen(modalities_to_include)),
        torch.nn.Tanh(),
        torch.nn.Linear(np.alen(modalities_to_include), D_out),
        torch.nn.Softmax(dim=1),
    ).to(device)

    # What optimization algorithm to use, and which ones to optimize and with what parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    # How many samples to use for each training iteration
    # This is the one that introduces speed-up due to paralellization
    batchsize=10
    sample_range=range(N);
    print("Started training")
    noepochs=1000 # How many epochs to run

    model[0].weight=torch.nn.Parameter(model[0].weight*WW) # Masks the weight matrix

    samples=range(X_train.shape[0])

    for epoch in range(noepochs):
        # Get random indices to form the next batch of training data

        ind=random.sample(samples,batchsize);
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x[ind,:].to(device))
        
        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the loss.
        loss = loss_fn(y_pred, y[ind].to(device))
        
        # Give an update regarding the error every 1000 epochs
        if epoch%10==0 and epoch>0:
            print(epoch, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()
        
        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.

        # Push the gradients learned from the loss to the rest of the network
        loss.backward()
        # Update the weights handled by the optiimzer (the weights of the netoork)
        optimizer.step()
        #print(WW.shape)
        
        with torch.no_grad():
            #print(model[0].weight.shape)
            #print(WW.shape)
            
            model[0].weight=torch.nn.Parameter(model[0].weight*WW)
            #model[0].weight = (model[0].weight*WW)
            '''
            print(model[0].weight)
        '''

    # See that all the weights where the mask is zero, are zero
    #print(model[0].weight)

    print("Finished training")

    # Get the output for the test data
    y_pred = model(x_test.to(device))

    # Check the ground truth for the test data
    target=labels_test[:,0]

    # Turn the outputs to classes (continuous values to integers of 0 or 1)
    predicted=np.floor(y_pred.detach().cpu().numpy()+0.5)[:,0]

    # Print out the accuracy on the test data
    accuracy=np.sum(target.T==predicted)/len(target)*100
    print("Accuracy",accuracy,"%")
    output_name="result_p"+str(permutation)+"_clinical"+str(clinical_data)+"_window"+str(timewindow)
    result_filename="./results/"+output_name+"_"+str(np.random.randint(1000000))+'.mat'
    
    sio.savemat(result_filename, {'vect':accuracy})



if __name__ == '__main__':
    classify(0,1,5);
