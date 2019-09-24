import pickle
import gzip
#import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import time 
from math import floor

from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws
from sklearn.metrics import r2_score

for lr in [0.003,0.0025,0.002,0.0015,0.001]: # [0.001,0.0009,0.00075,0.0005]:
    for i in [3,7,10]:
        with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
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

        nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=lr, mu=0.9, lambd=0, minibatch=64, disableLog=False)
        nn.addLayers([256], ['leakyrelu','leakyrelu'])
        loss=nn.train(stop_function=3, num_epochs=20000)
    
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
        with open("res02.txt", "a+") as mf:
            mf.write("1 hidden --> file {2}, lr={5}, dim={4}: maxerr={0} -- %err={1} -- meanErr={6} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, lr,floor(loss*dim_set)))
        ww=nn.getWeight()
        with open ('NN_2_file{}_lr{}'.format(i, lr), 'wb') as f:
            pickle.dump(ww, f)
    
    
        now = time.time()
        nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=lr, mu=0.9, lambd=0, minibatch=64, disableLog=False)

        nn.addLayers([256,256], ['leakyrelu','leakyrelu','leakyrelu'])
        loss = nn.train(stop_function=3, num_epochs=20000)
    
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
        with open("res02.txt", "a+") as mf:
            mf.write("2 hidden --> file {2}, lr={5} dim={4}: maxerr={0} -- %err={1} -- meanErr={6} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, lr, floor(loss*dim_set)))
        ww=nn.getWeight()
        with open('NN_3_file{}_lr{}'.format(i, lr), 'wb') as f:
            pickle.dump(ww, f)
    
    '''
    w = (nn.getWeight())
    nn_pr=pruning.NN_pruned(training=[bin_data_perm, labels_perm], testing=[[0],[0]], lr=lr, mu=0.9, lambd=1.e-5 ,minibatch=64, disableLog=False)
    
    for p in range(10,96,5):
        w1=np.copy(w)
        nn_pr.addLayers([256], ['leakyrelu','tanh'], w1)
        nn_pr.set_pruned_layers(p, w1)
        
        nn_pr.train(stop_function=2, num_epochs=20000)
        max_err=0
        mean_err=0
        for j in range(dim_set):
            pr = floor(nn_pr.predict(bin_data[j])[0]*dim_set)
            val=abs(pr-labels[j]*dim_set)
            if val>max_err:
                max_err = val
            mean_err += val

        print("1 hidden, {3}% pruning --> file {2}: max error={0} -- mean error={1}".format(max_err[0], mean_err[0]/(len(bin_data)), i, p))

    nn_ws=ws.NN_WS(training=[bin_data_perm, labels_perm], testing=[[0],[0]], lr=lr, mu=0.9, lambd=1.e-5 ,minibatch=64, disableLog=False)
    c=[4096,64]
    w1=np.copy(w)
    nn_ws.addLayers([256], ['leakyrelu','tanh'], w1)
    nn_ws.set_ws(c, w1)
    
    nn_ws.train(stop_function=2, num_epochs=20000)
    max_err=0
    mean_err=0
    for j in range(dim_set):
        pr = floor(nn_ws.predict(bin_data[j])[0]*dim_set)
        val=abs(pr-labels[j]*dim_set)
        if val>max_err:
            max_err = val
        mean_err += val

    print("1 hidden, {3} clusters --> file {2}: max error={0} -- mean error={1}".format(max_err[0], mean_err[0]/(len(bin_data)), i, c))
    
    
    '''
