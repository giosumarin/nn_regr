import pickle
import gzip
import numpy as np
import sys
import h5py
import time 
from math import floor, sqrt, ceil


from NN_no_hidden import NN as NN1
from NN_no_hidden import pruning_module as pr1
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws

def make_structured_input_for_root_NN(bin_data, labels, split, dim_set):
    #p = np.random.RandomState(seed=42).permutation(dim_set)
    position_labels = np.copy(labels)
    # position_labels = position_labels[p]
    # bin_data = bin_data[p]
    # labels[p] = labels[p]
    size_batch = dim_set // split
    remain = ceil((dim_set / split - size_batch) * split)
    for i in range(0, split, 1):
        position_labels[(i*size_batch):((i+1)*(size_batch if i < (split-1) else size_batch+remain))]=i
    position_labels = np.reshape(position_labels, (-1,1))
    splitted_labels = [np.reshape(labels[(i*size_batch):((i+1)*(size_batch if i < (split-1) else size_batch+remain))]/dim_set, (-1,1)) for i in range(split)]
    splitted_bin_data = [bin_data[(i*size_batch):((i+1)*(size_batch if i < (split-1) else size_batch+remain)),:] for i in range(split)] 

    p = np.random.RandomState(seed=42).permutation(dim_set)
    p_split = np.random.RandomState(seed=42).permutation(dim_set//split)

    perm_splitted_bin_data = [splitted_bin_data[i][p_split] for i in range(split)]
    perm_position_labels = position_labels[p]
    perm_splitted_labels = [splitted_labels[i][p_split] for i in range(split)]


    return bin_data[p], perm_splitted_bin_data, perm_position_labels, perm_splitted_labels
    
    
def make_labels_for_class(position_labels):
    n_rows = position_labels.shape[0]
    result = np.zeros((n_rows,2))
    for i in range(n_rows):
        result[i,position_labels[i]] = 1
    return result


N_FEATURES = 64
N_CLASSES = 1
np.random.RandomState(42)

weights = np.random.randn(N_FEATURES, N_CLASSES).astype(np.float32) * sqrt(2/N_FEATURES)
#weights = (np.zeros((N_FEATURES, N_CLASSES))).astype(np.float32)
bias = np.ones((1, N_CLASSES)).astype(np.float32)*0.001
#bias = np.random.randn(1, N_CLASSES).astype(np.float32) * sqrt(2/N_CLASSES)
w= [[weights, bias]]



for i in [3,7,10]:
    with open("to_tex.txt", "a+") as tex:
            tex.write("\nfile {}\n".format(i))
    for spl in [2,3,4,5,6,7,8]:
        with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
            data = f.get('Sb') 
            bin_data = np.array(data, dtype=np.bool)
            bin_data = np.transpose(bin_data)
            bin_data = np.flip(bin_data,axis=1)
            dim_set = len(bin_data)
        
        split = spl
        labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
        bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)

        # for i in range(split): 
        #     print(np.max(splitted_labels[i]))
        #     print(np.min(splitted_labels[i]))

        # labels = labels/len(bin_data)
        # labels = np.reshape(labels, (-1, 1))
        # bin_data = bin_data[p]
        # labels = labels[p]

        # position_labels_class = make_labels_for_class(position_labels.astype('int8'))
        # nn = NN1.NN(training=[bin_data, position_labels_class], testing=[[0],[0]], lr=0.03, mu=0.9, output_classes=2, minibatch=32, disableLog=True)
        # nn.addLayers(['softmax'])
        # now = time.time()
        # loss = nn.train(stop_function=4, num_epochs=20000)
        # diff = time.time() - now
        # res = np.argmax(nn.predict(bin_data),axis=1).reshape(-1,1)
        # sum = np.sum(np.abs(res-position_labels))
        # print("epoch = {} - loss on sel_brancher = {} - sum error = {} - time = {}s".format(nn.epoch, loss, sum, round(diff,5)))
        
        
        
        
        # position_labels_class = make_labels_for_class(position_labels.astype('int8'))
        # nn = NN.NN(training=[bin_data, position_labels_class], testing=[[0],[0]], lr=0.03, mu=0.9, output_classes=2, minibatch=32, disableLog=True)
        # nn.addLayers([32],['relu', 'softmax'])
        # now = time.time()
        # loss = nn.train(stop_function=4, num_epochs=100)
        # diff = time.time() - now
        # res = np.argmax(nn.predict(bin_data),axis=1).reshape(-1,1)
        # sum = np.sum(np.abs(res-position_labels))
        # print("H: epoch = {} - loss on sel_brancher = {} - sum error = {} - time = {}s".format(nn.epoch, loss, sum, round(diff,5)))


        #H: epoch = 45 - loss on sel_brancher = 9.1e-06 - sum error = 0.0 - time = 462.80945s
        
        
        #nn = NN1.NN(training=[bin_data, position_labels], testing=[[0],[0]], lr=0.1, mu=0.9, lambd=0, minibatch=32, disableLog=True)
        #nn.addLayers(['sigmoid'])
        #now = time.time()
        #loss = nn.train(stop_function=4, num_epochs=20000)
        #diff = time.time() - now
        #res = nn.predict(bin_data)
        #res[res <= 0.5]=0
        #res[res > 0.5]=1
        #sum = np.sum(np.abs(res-position_labels))
        #print("epoch = {} - loss on sel_brancher = {} - sum error = {} - time = {}s".format(nn.epoch, loss, sum, round(diff,5)))
        
        '''
        f3: epoch = 1013 - loss on sel_brancher = 0.0237729 - sum error = 0.0 - time = 0.97598s
        f7: epoch = 12679 - loss on sel_brancher = 0.0024501 - sum error = 0.0 - time = 176.93983s
        '''
        max_errs = []
        for s in range(split):
            ww = np.copy(w)
            nn = NN1.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.05, mu=0.9, output_classes=1, lambd=0, minibatch=32, disableLog=True)
            nn.addLayers(['leakyrelu'], ww)
            now=time.time()
            loss = nn.train(stop_function=3, num_epochs=20000)
            difference = round(time.time() - now, 5)
 
           
            
            # max_err=0
            # mean_err=0
            # for j in range(dim_set // split) :
            #     pr = floor(nn.predict(splitted_bin_data[s][j])[0]*(dim_set))
            #     val=abs(pr-splitted_labels[s][j]*(dim_set))
            #     if val > max_err:
            #         max_err = val
            #     mean_err += val
            # mean_err/=dim_set
            
            pr = np.floor(nn.predict(splitted_bin_data[s]) * dim_set)
            lab = splitted_labels[s] *dim_set
            max_err = np.max(np.abs(pr-lab)).astype("int32")
            max_errs.append(max_err)
            print("0 hidden --> file {}, split={}, dim={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
            .format(i, spl, (dim_set/split), nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(),5)))
        
        with open("to_tex.txt", "a+") as tex:
            tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(spl, list(max_errs), max(max_errs), round(nn.get_memory_usage()*spl,5)))
        



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
