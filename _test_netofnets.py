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

# def make_bin_tree(bin_data, labels, dim_set, layers):
#     position_labels = np.copy(labels)
#     split = 2
#     size_batch = dim_set // split
#     size_batch = [dim_set // (split * l) for l in range(1, layers, 1)]
#     mul_for_layers = 
#     pos_labels_list = []
#     #if dim_set % split == 0:
#     for l in range (1,layers,1):
#         for i in range (split):
#             copy = np.copy(labels)
#         pos_labels_list[l] = [position_labels[i*size_batch[l-1]:(i+1)*size_batch[l-1]]=i for i in range(split*l)]
#         pos_labels_list.append([copy[i*size_batch[l-1]:(i+1)*size_batch[l-1]] = i for i in range(split*l)])
        




#     pos_labels_list[1] = [position_labels[(i*size_batch[1]:(i+1)*size_batch[1])]=i for i in range(split*l]




    # #for l in range(layers-1):
    # for l in range(1, layers, 1):
    #     pos_labels_list[l] = [position_labels[(i*size_batch[l]):((i+1))*size_batch[l])]=i for i in range(0, split, 1)]
    # #position_labels[result>split-1] = split-1
    # position_labels = np.reshape(position_labels, (-1,1))
    # splitted_labels = [np.reshape(labels[(i*size_batch):((i+1)*size_batch)]/(dim_set/split), (-1,1)) for i in range(split)]
    # splitted_bin_data = [bin_data[(i*size_batch):((i+1)*size_batch),:] for i in range(split)] 

    # p = np.random.RandomState(seed=42).permutation(dim_set)
    # p_split = np.random.RandomState(seed=42).permutation(dim_set//split)

    # perm_splitted_bin_data = [splitted_bin_data[i][p_split] for i in range(split)]
    # perm_position_labels = position_labels[p]
    # perm_splitted_labels = [splitted_labels[i][p_split] for i in range(split)]


    # return bin_data[p], perm_splitted_bin_data, perm_position_labels, perm_splitted_labels


N_FEATURES = 64
N_CLASSES = 1
np.random.RandomState(42)

weights = np.random.randn(N_FEATURES, N_CLASSES).astype(np.float32) * sqrt(2/N_FEATURES)
#weights = (np.zeros((N_FEATURES, N_CLASSES))).astype(np.float32)
bias = np.ones((1, N_CLASSES)).astype(np.float32)*0.001
#bias = np.random.randn(1, N_CLASSES).astype(np.float32) * sqrt(2/N_CLASSES)
w= [[weights, bias]]


for spl in [2]:
    for i in [10]:
        with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
            data = f.get('Sb') 
            bin_data = np.array(data, dtype=np.bool)
            bin_data = np.transpose(bin_data)
            bin_data = np.flip(bin_data,axis=1)
            dim_set = len(bin_data)
        
            split = spl
            labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
            bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)
        
        #labels = labels/len(bin_data)
        #labels = np.reshape(labels, (-1, 1))
        #bin_data = bin_data[p]
        #labels = labels[p]
        
        nn = NN1.NN(training=[bin_data, position_labels], testing=[[0],[0]], lr=0.1, mu=0.9, lambd=0, minibatch=32, disableLog=True)
        nn.addLayers(['sigmoid'])
        now = time.time()
        loss = nn.train(stop_function=4, num_epochs=20000)
        diff = time.time() - now
        res = nn.predict(bin_data)
        res[res <= 0.5]=0
        res[res > 0.5]=1
        sum = np.sum(np.abs(res-position_labels))
        print("epoch = {} - loss on sel_brancher = {} - sum error = {} - time = {}s".format(nn.epoch, loss, sum, round(diff,5)))
        
        '''
        f3: epoch = 1013 - loss on sel_brancher = 0.0237729 - sum error = 0.0 - time = 0.97598s
        f7: epoch = 12679 - loss on sel_brancher = 0.0024501 - sum error = 0.0 - time = 176.93983s
        '''

        for s in range(split):
            nn = NN1.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.01, mu=0.9, lambd=0, minibatch=32, disableLog=True)
            nn.addLayers(['leakyrelu'])
            now=time.time()
            loss = nn.train(stop_function=3, num_epochs=20000)
            difference = round(time.time() - now, 5)
            max_err=0
            mean_err=0
            for j in range(dim_set // split):
                pr = floor(nn.predict(splitted_bin_data[s][j])[0]*(dim_set//split))
                val=abs(pr-splitted_labels[s][j]*(dim_set//split))
                if val > max_err:
                    max_err = val
                mean_err += val
            mean_err/=dim_set
            
            print("0 hidden --> file {}, split={}, dim={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
            .format(i, spl, (dim_set/split), nn.epoch, max_err[0], round(max_err[0]/(dim_set/split)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(),5)))



        # for s in range(split):
        #     nn = NN.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.01, mu=0.9, lambd=0, minibatch=32, disableLog=True)
        #     nn.addLayers([32],['leakyrelu','leakyrelu'])
        #     now=time.time()
        #     loss = nn.train(stop_function=3, num_epochs=20000)
        #     difference = round(time.time() - now, 3)
        #     max_err=0
        #     mean_err=0
        #     for j in range(dim_set // split):
        #         pr = floor(nn.predict(splitted_bin_data[s][j])[0]*(dim_set//split))
        #         val=abs(pr-splitted_labels[s][j]*(dim_set//split))
        #         if val > max_err:
        #             max_err = val
        #         mean_err += val
        #     mean_err/=dim_set
            
        #     print("1 hidden --> file {}, split={}, dim={}: maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
        #     .format(i, spl, (dim_set/split), max_err, round(max_err[0]/(dim_set/split)*100,3), loss, difference, nn.get_memory_usage()))
        print("-*-*"*35)




        '''
        0 hidden --> file 3, split=2, dim=256.0: epoch: 155 -- maxerr=6.0 -- %err=2.344 -- meanErr=0.00529 -- time=0.17509s -- spaceOVH=0.09918
0 hidden --> file 3, split=2, dim=256.0: epoch: 1286 -- maxerr=7.0 -- %err=2.734 -- meanErr=0.00972 -- time=1.30557s -- spaceOVH=0.09918
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
0 hidden --> file 7, split=2, dim=4096.0: epoch: 27 -- maxerr=37.0 -- %err=0.903 -- meanErr=0.00308 -- time=0.40611s -- spaceOVH=0.0062
0 hidden --> file 7, split=2, dim=4096.0: epoch: 231 -- maxerr=24.0 -- %err=0.586 -- meanErr=0.00148 -- time=3.22929s -- spaceOVH=0.0062
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
0 hidden --> file 10, split=2, dim=524288.0: epoch: 28 -- maxerr=592.0 -- %err=0.113 -- meanErr=0.00034 -- time=47.1023s -- spaceOVH=5e-05
0 hidden --> file 10, split=2, dim=524288.0: epoch: 51 -- maxerr=470.0 -- %err=0.09 -- meanErr=0.00022 -- time=85.51323s -- spaceOVH=5e-05
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        '''
