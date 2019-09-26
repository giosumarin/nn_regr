import pickle
import gzip
import numpy as np
import sys
import h5py
import time 
from math import floor


from NN_no_hidden import NN as NN1
from NN_no_hidden import pruning_module as pr1
from NN_no_hidden import WS_module as ws1
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws

with open('weights_palermo_full', 'rb') as f:
    d = pickle.load(f) 

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
    
    w=np.copy(d['f{}_1'.format(i)])
    now = time.time()
     
    nn = NN1.NN(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
    nn.addLayers(['leakyrelu'], w)
    #loss=nn.train(stop_function=3, num_epochs=20000)
    loss = nn.loss(bin_data, labels)
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
    with open("res_nn1_ws.txt", "a+") as mf:
        mf.write("0 hidden --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s -- spaceOVH={6}KB\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, nn.get_memory_usage()))
       
    
    
    nn_ws=ws1.NN_WS(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0 ,minibatch=64, disableLog=True)   
    for p in range(10,91,10):
        w=np.copy(d['f{}_1'.format(i)])
        nn_ws.addLayers(['leakyrelu'], w)
        c=ws.num_cluster_from_perc(w, p/100)
        
        if not np.any(np.array(c) <= 0):
            nn_ws.set_ws(c, w)
            now = time.time() 
            loss = nn_ws.train(stop_function=3, num_epochs=20000)
            
            later = time.time()
            difference = int(later - now)
            
            max_err=0
            mean_err=0
            for j in range(dim_set):
                pr = floor(nn_ws.predict(bin_data[j])[0]*dim_set)
                val=abs(pr-labels[j]*dim_set)
                if val>max_err:
                    max_err = val
                mean_err += val
            
            with open("res_nn1_ws.txt", "a+") as mf:
                mf.write("0 hidden, {6}% space ({7}) --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s -- spaceOVH={8}KB\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, p, c, nn_ws.get_memory_usage()))


    w=np.copy(d['f{}_2'.format(i)])
    now = time.time()
     
    nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
    nn.addLayers([256],['leakyrelu','leakyrelu'], w)
    #loss=nn.train(stop_function=3, num_epochs=20000)
    loss = nn.loss(bin_data, labels)
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
    with open("res_nn2_ws.txt", "a+") as mf:
        mf.write("1 hidden --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s -- spaceOVH={6}KB\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, nn.get_memory_usage()))
       
    
    
    nn_ws=ws.NN_WS(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0 ,minibatch=64, disableLog=True)   
    for p in range(10,91,10):
        w=np.copy(d['f{}_2'.format(i)])
        nn_ws.addLayers([256],['leakyrelu','leakyrelu'], w)
        c=ws.num_cluster_from_perc(w, p/100)
        if not np.any(np.array(c) <= 0):
            nn_ws.set_ws(c, w)
            now = time.time() 
            loss = nn_ws.train(stop_function=3, num_epochs=20000)
            
            later = time.time()
            difference = int(later - now)
            
            max_err=0
            mean_err=0
            for j in range(dim_set):
                pr = floor(nn_ws.predict(bin_data[j])[0]*dim_set)
                val=abs(pr-labels[j]*dim_set)
                if val>max_err:
                    max_err = val
                mean_err += val
            
            with open("res_nn2_ws.txt", "a+") as mf:
                mf.write("1 hidden, {6}% space ({7}) --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s -- spaceOVH={8}KB\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, p, c, nn_ws.get_memory_usage()))
            
            

    w=np.copy(d['f{}_3'.format(i)])
    now = time.time()
 
 
     
    nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
    nn.addLayers([256,256],['leakyrelu','leakyrelu','leakyrelu'], w)
    #loss=nn.train(stop_function=3, num_epochs=20000)
    loss = nn.loss(bin_data, labels)
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
    with open("res_nn3_ws.txt", "a+") as mf:
        mf.write("2 hidden --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s -- spaceOVH={6}KB\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, nn.get_memory_usage()))

    nn_ws=ws.NN_WS(training=[bin_data, labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0 ,minibatch=64, disableLog=True)   
    for p in range(10,91,10):
        w=np.copy(d['f{}_3'.format(i)])
        nn_ws.addLayers([256,256],['leakyrelu','leakyrelu','leakyrelu'], w)
        c=ws.num_cluster_from_perc(w, p/100)
        if not np.any(np.array(c) <= 0):
            nn_ws.set_ws(c, w)
            now = time.time() 
            loss = nn_ws.train(stop_function=3, num_epochs=20000)
            
            later = time.time()
            difference = int(later - now)
            
            max_err=0
            mean_err=0
            for j in range(dim_set):
                pr = floor(nn_ws.predict(bin_data[j])[0]*dim_set)
                val=abs(pr-labels[j]*dim_set)
                if val>max_err:
                    max_err = val
                mean_err += val
            with open("res_nn3_ws.txt", "a+") as mf:
                mf.write("2 hidden, {6}% space ({7}) --> file {2}, dim={4}: maxerr={0} -- %err={1} -- meanErr={5} -- time={3}s -- spaceOVH={8}KB\n".format(max_err[0], round(max_err[0]/dim_set*100,3), i, difference, dim_set, loss, p, c, nn_ws.get_memory_usage()))
