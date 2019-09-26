import pickle
import gzip
#import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import time 
from math import floor

from NN_no_hidden import NN
#from NN_pr import pruning_module as pruning
#from NN_pr import WS_module as ws
from sklearn.metrics import r2_score

for i in [3]:
    with h5py.File('Resource/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
        data = f.get('Sb') 
        bin_data = np.array(data, dtype=np.bool)
        bin_data = np.transpose(bin_data)
        bin_data = np.flip(bin_data,axis=1)
        dim_set = len(bin_data)

    labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
    labels = labels/len(bin_data)
    labels = np.reshape(labels, (-1, 1))

    p = np.random.RandomState(seed=42).permutation(dim_set)

    bin_data = bin_data[p]
    labels = labels[p]
    
    now = time.time()


    nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
    nn.addLayers(['leakyrelu'])
    loss=nn.train(stop_function=3, num_epochs=2)
    print(nn.get_memory_usage())

    later = time.time()
    difference = int(later - now)

    #print("R2: {}".format(r2_score(labels, nn.predict(bin_data))))
    max_err=0
    mean_err=0
    for j in range(dim_set):
        pr = floor(nn.predict(bin_data[j])[0]*dim_set)
        val=abs(pr-labels[j]*dim_set)
        if val > max_err:
            max_err = val
        mean_err += val
    #with open("res02.txt", "a+") as mf:
        #mf.write(
    print("0 hidden --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss))
