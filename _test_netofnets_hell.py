import pickle
import gzip
import numpy as np
from numpy.random import seed
import sys
import h5py
import time 
from math import floor, sqrt, ceil
from sklearn.preprocessing import MinMaxScaler
from itertools import chain


from NN_no_hidden import NN as NN1
from NN_no_hidden import NN_hellinger as NN2
from NN_no_hidden import pruning_module as pr1
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws



def make_structured_input_for_root_NN(bin_data, labels, split, dim_set):
    position_labels = np.copy(labels)
    size_batch = dim_set // split
    remain = dim_set % split #ceil((dim_set / split - size_batch) * split)
    for i in range(0, split, 1):
        position_labels[(i*size_batch):(((i+1)*size_batch if i < (split-1) else (i+1)*size_batch+remain))]=i
    
    position_labels = np.reshape(position_labels, (-1,1))
    splitted_labels = [np.reshape(labels[(i*size_batch):(((i+1)*size_batch if i < (split-1) else (i+1)*size_batch+remain))]/dim_set, (-1,1)) for i in range(split)]
    splitted_bin_data = [bin_data[(i*size_batch):(((i+1)*size_batch if i < (split-1) else (i+1)*size_batch+remain))] for i in range(split)] 
    
    perm_splitted_bin_data = []
    perm_splitted_labels = []
    for i in range(split):
        p_split = np.random.RandomState(seed=0).permutation(len(splitted_bin_data[i]))
        
        perm_splitted_labels.append(splitted_labels[i][p_split])
        perm_splitted_bin_data.append(splitted_bin_data[i][p_split])
    
    
    p = np.random.RandomState(seed=0).permutation(dim_set)
    perm_position_labels = position_labels[p]


    return bin_data[p], perm_splitted_bin_data, perm_position_labels, perm_splitted_labels
    
    
def make_labels_for_class(position_labels):
    n_rows = position_labels.shape[0]
    result = np.zeros((n_rows,2))
    for i in range(n_rows):
        result[i,position_labels[i]] = 1
    return result

def scaler(original_label):
    return (original_label - np.min(original_label))/(np.max(original_label) - np.min(original_label))

def descaler(original_label, predicted_scaled):
    return predicted_scaled * (np.max(original_label) - np.min(original_label)) + np.min(original_label)

def number_to_tex(num):
    exp=0
    while(num<1 and num != 0):
        num *= 10
        exp-=1
    if exp == 0:
        return "{}".format(round(num,2))
    else:
        return "{}\\times 10^{{{}}}".format(round(num,2), exp)


N_FEATURES = 64
N_CLASSES = 1


#weights = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(N_FEATURES, N_CLASSES)).astype(np.float32)
#bias = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(1, N_CLASSES)).astype(np.float32)
weights = np.zeros((N_FEATURES, N_CLASSES)).astype(np.float32)
bias = np.zeros((1, N_CLASSES)).astype(np.float32)

w= [[weights, bias]]
print(w[0][0][0][0])
for l in [1,0]:
    for i in [3,7,10]:
        with open("to_tex_all_manythings.txt", "a+") as tex:
            tex.write("\nfile {} \n".format(i))
        for mb in [64]:
            for lr in [1e-7,1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
                # concatenated_splits = chain(range(1,22,1), range(22,65,4), range(64,143, 8))
                concatenated_splits = [1]

                for spl in concatenated_splits: 
                    with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
                        data = f.get('Sb') 
                        bin_data = np.array(data, dtype=np.bool)
                        bin_data = np.transpose(bin_data)
                        bin_data = np.flip(bin_data,axis=1)
                        dim_set = len(bin_data)
                    
                    split = spl
                    labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
                    bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)
                    
                    max_errs = []
                    minibatchsize = mb
                    
                    
                        
                    for s in range(split):
                        ww = np.copy(w)
                        if l==0:
                            nn = NN1.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=lr, mu=0.9, output_classes=1, lambd=0, minibatch=minibatchsize, disableLog=True)
                        else:
                            nn = NN2.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=lr, mu=0.9, output_classes=1, lambd=0, minibatch=minibatchsize, disableLog=True)
                            


                        nn.addLayers(['relu'], ww)
                        nn.set_patience(10)
                        now=time.time()
                        loss = nn.train(stop_function=0, num_epochs=1)
                        difference = round(time.time() - now, 5)
                        predict = nn.predict(splitted_bin_data[s])
                        pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
                        lab = splitted_labels[s] * dim_set
                        max_err = np.max(np.abs(pr-lab)).astype("int64")
                        max_errs.append(max_err)
                        print("0 hidden --> file {}, split={}, dim={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                        .format(i, spl, ceil(dim_set/split), nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                    
                    with open("to_tex_all_manythings.txt", "a+") as tex:
                        tex.write("${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & {} & ${}$ \\\ \n".format("MSE" if l==0 else "HELLINGER", number_to_tex(lr), mb, spl, max(max_errs), round(np.mean(max_errs),2), number_to_tex(loss), number_to_tex(nn.get_memory_usage(dim_set)*spl)))
                        #tex.write("${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format("MSE" if l==0 else "ABS", number_to_tex(lr), mb, max(max_errs), number_to_tex(loss), number_to_tex(nn.get_memory_usage(dim_set)*spl)))

                    print("-*-*"*35)

