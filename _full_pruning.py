import pickle
import gzip
import numpy as np
import sys
import h5py
import time 
from math import floor


from NN_no_hidden import NN as NN1
from NN_no_hidden import pruning_module as pr1
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws

with open('weights_palermo', 'rb') as f:
    d = pickle.load(f) 

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
    
    w=np.copy(d['f{}_1'.format(i)])
    now = time.time()
     
    nn = NN1.NN(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
    nn.addLayers(['leakyrelu'], w)
    #loss=nn.train(stop_function=3, num_epochs=20000)
    
    later = time.time()
    difference = int(later - now)

    max_err=0
    mean_err=0
    for j in range(dim_set):
        pr = floor(nn.predict(bin_data[j])[0]*dim_set)
        val=abs(pr-labels[j]*dim_set)
        if val > max_err:
            max_err = val
        mean_err += val
    mean_err/=dim_set
    with open("res_nn1.txt", "a+") as mf:
        mf.write("0 hidden --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, mean_err))
    
       
    
    
    nn_pr=pr1.NN_pruned(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0 ,minibatch=64, disableLog=True)   
    for p in range(10,96,10):
        w=np.copy(d['f{}_1'.format(i)])
        nn_pr.addLayers(['leakyrelu'], w)
        nn_pr.set_pruned_layers(p, w)
        now = time.time() 
        loss = nn_pr.train(stop_function=3, num_epochs=20000)
        
        later = time.time()
        difference = int(later - now)
        
        max_err=0
        mean_err=0
        for j in range(dim_set):
            pr = floor(nn_pr.predict(bin_data[j])[0]*dim_set)
            val=abs(pr-labels[j]*dim_set)
            if val>max_err:
                max_err = val
            mean_err += val
        
        with open("res_nn1.txt", "a+") as mf:
            mf.write("0 hidden, {6}% pr --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, p))
            
            
            


    w=np.copy(d['f{}_2'.format(i)])
    now = time.time()
     
    nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
    nn.addLayers([256],['leakyrelu','leakyrelu'], w)
    #loss=nn.train(stop_function=3, num_epochs=20000)
    
    later = time.time()
    difference = int(later - now)

    max_err=0
    mean_err=0
    for j in range(dim_set):
        pr = floor(nn.predict(bin_data[j])[0]*dim_set)
        val=abs(pr-labels[j]*dim_set)
        if val > max_err:
            max_err = val
        mean_err += val
    mean_err/=dim_set
    with open("res_nn2.txt", "a+") as mf:
        mf.write("1 hidden --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, mean_err))
    
       
    
    
    nn_pr=pruning.NN_pruned(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0 ,minibatch=64, disableLog=True)   
    for p in range(10,96,10):
        w=np.copy(d['f{}_2'.format(i)])
        nn_pr.addLayers([256],['leakyrelu','leakyrelu'], w)
        nn_pr.set_pruned_layers(p, w)
        now = time.time() 
        loss = nn_pr.train(stop_function=3, num_epochs=20000)
        
        later = time.time()
        difference = int(later - now)
        
        max_err=0
        mean_err=0
        for j in range(dim_set):
            pr = floor(nn_pr.predict(bin_data[j])[0]*dim_set)
            val=abs(pr-labels[j]*dim_set)
            if val>max_err:
                max_err = val
            mean_err += val
        
        with open("res_nn2.txt", "a+") as mf:
            mf.write("1 hidden, {6}% pr --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, p))
            
            
            
    w=np.copy(d['f{}_3'.format(i)])
    now = time.time()
     
    nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
    nn.addLayers([256, 256],['leakyrelu','leakyrelu','leakyrelu'], w)
    #loss=nn.train(stop_function=3, num_epochs=20000)
    
    later = time.time()
    difference = int(later - now)

    max_err=0
    mean_err=0
    for j in range(dim_set):
        pr = floor(nn.predict(bin_data[j])[0]*dim_set)
        val=abs(pr-labels[j]*dim_set)
        if val > max_err:
            max_err = val
        mean_err += val
    mean_err/=dim_set
    with open("res_nn3.txt", "a+") as mf:
        mf.write("2 hidden --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, mean_err))
    
       
    
    
    nn_pr=pruning.NN_pruned(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0 ,minibatch=64, disableLog=True)   
    for p in range(10,96,10):
        w=np.copy(d['f{}_3'.format(i)])
        nn_pr.addLayers([256,256],['leakyrelu', 'leakyrelu', 'leakyrelu'], w)
        nn_pr.set_pruned_layers(p, w)
        now = time.time() 
        loss = nn_pr.train(stop_function=3, num_epochs=20000)
        
        later = time.time()
        difference = int(later - now)
        
        max_err=0
        mean_err=0
        for j in range(dim_set):
            pr = floor(nn_pr.predict(bin_data[j])[0]*dim_set)
            val=abs(pr-labels[j]*dim_set)
            if val>max_err:
                max_err = val
            mean_err += val
        
        with open("res_nn3.txt", "a+") as mf:
            mf.write("2 hidden, {6}% pr --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, p))
            
    
    
    
        
    
        
        
