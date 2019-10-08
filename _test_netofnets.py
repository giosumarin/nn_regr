import pickle
import gzip
import numpy as np
import sys
import h5py
import time 
from math import floor, sqrt


from NN_no_hidden import NN as NN1
from NN_no_hidden import pruning_module as pr1
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws

def make_structured_input_for_root_NN(bin_data, labels, split, dim_set):
    position_labels = np.copy(labels)
    size_batch = dim_set // split
    #if dim_set % split == 0: 
    for i in range(0, split, 1):
        position_labels[(i*size_batch):((i+1)*size_batch)]=i
    #position_labels[result>split-1] = split-1
    position_labels = np.reshape(position_labels, (-1,1))
    splitted_labels = [np.reshape(labels[(i*size_batch):((i+1)*size_batch)]/(dim_set/split), (-1,1)) for i in range(split)]
    splitted_bin_data = [bin_data[(i*size_batch):((i+1)*size_batch),:] for i in range(split)] 

    p = np.random.RandomState(seed=42).permutation(dim_set)
    p_split = np.random.RandomState(seed=42).permutation(dim_set//split)

    perm_splitted_bin_data = [splitted_bin_data[i][p_split] for i in range(split)]
    perm_position_labels = position_labels[p]
    perm_splitted_labels = [splitted_labels[i][p_split] for i in range(split)]


    return bin_data[p], perm_splitted_bin_data, perm_position_labels, perm_splitted_labels
            #sistema ultimo

N_FEATURES = 64
N_CLASSES = 1
np.random.RandomState(0)

weights = np.random.randn(N_FEATURES, N_CLASSES).astype(np.float32) * sqrt(2/N_FEATURES)
#weights = (np.zeros((N_FEATURES, N_CLASSES))).astype(np.float32)
bias = np.ones((1, N_CLASSES)).astype(np.float32)*0.001
#bias = np.random.randn(1, N_CLASSES).astype(np.float32) * sqrt(2/N_CLASSES)
w= [[weights, bias]]



for i in [3,7]:
    with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
        data = f.get('Sb') 
        bin_data = np.array(data, dtype=np.bool)
        bin_data = np.transpose(bin_data)
        bin_data = np.flip(bin_data,axis=1)
        dim_set = len(bin_data)
    for spl in [4]:
        split = spl
        labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
        bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)
        
        #labels = labels/len(bin_data)
        #labels = np.reshape(labels, (-1, 1))
        #bin_data = bin_data[p]
        #labels = labels[p]
        
        nn = NN1.NN(training=[bin_data, position_labels], testing=[[0],[0]], lr=0.001, mu=0.9, lambd=0, minibatch=64, disableLog=True)
        nn.addLayers(['leakyrelu'])
        loss = nn.train(stop_function=3, num_epochs=20000)
        
        for s in range(split):
            nn = NN1.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.1, mu=0.9, lambd=0, minibatch=64, disableLog=True)
            nn.addLayers(['leakyrelu'], w)
            loss = nn.train(stop_function=3, num_epochs=20000)

            max_err=0
            mean_err=0
            for j in range(dim_set // split):
                pr = floor(nn.predict(splitted_bin_data[s][j])[0]*(dim_set//split))
                val=abs(pr-splitted_labels[s][j]*(dim_set//split))
                if val > max_err:
                    max_err = val
                mean_err += val
            mean_err/=dim_set
            difference=0
            print("0 hidden --> file {}, split={}, dim={}: maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
            .format(i, spl, (dim_set/split), max_err, round(max_err[0]/(dim_set/split)*100,3), loss, difference, nn.get_memory_usage()))



        for s in range(split):
            nn = NN.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.05, mu=0.9, lambd=0, minibatch=64, disableLog=True)
            nn.addLayers([100],['leakyrelu','leakyrelu'])
            loss = nn.train(stop_function=3, num_epochs=20000)

            max_err=0
            mean_err=0
            for j in range(dim_set // split):
                pr = floor(nn.predict(splitted_bin_data[s][j])[0]*(dim_set//split))
                val=abs(pr-splitted_labels[s][j]*(dim_set//split))
                if val > max_err:
                    max_err = val
                mean_err += val
            mean_err/=dim_set
            difference=0
            print("1 hidden --> file {}, split={}, dim={}: maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
            .format(i, spl, (dim_set/split), max_err, round(max_err[0]/(dim_set/split)*100,3), loss, difference, nn.get_memory_usage()))
