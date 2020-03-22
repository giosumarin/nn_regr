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

from utility_funs import make_structured_input_for_root_NN, make_labels_for_class, number_to_tex, remove_column_zero, reshape_nozero

from NN_pr import NN as NN1
from NN_pr import NN_max as NN2
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws

N_FEATURES = 64
N_CLASSES = 1

weights = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(N_FEATURES, N_CLASSES)).astype(np.float32)
bias = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(1, N_CLASSES)).astype(np.float32)
w= [[weights, bias]]
for i in [10]:
     
    #concatenated_splits = chain(range(1,22,1), range(22,65,4), range(64,143, 8))
    concatenated_splits = [1]
    for spl in concatenated_splits: 
        with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
            data = f.get('Sb') 
            bin_data = np.array(data, dtype=np.bool)
            bin_data = np.transpose(bin_data)
            bin_data = np.flip(bin_data,axis=1)
            dim_set = len(bin_data)
        
        bin_data, first_nonzero = remove_column_zero(bin_data)
        split = spl
        labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
        bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)
        for lr in [2e-2,3e-2,4e-2,5e-2]: 
            for mb in [256, 512, 1024,dim_set]:
                for neurons in [4,8,16]:
                    
                        max_errs = []
                            
                        for s in range(split):
                            #mb = dim_set if i==3 else 4096
                            #mb=512
                            #lr = 0.003 #if i==3 else 0.5
                            ww = np.copy(w)
                            nn = NN2.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=lr, mu=.9, output_classes=1, lambd=0, minibatch=mb, disableLog=False)
                            nn.addLayers([neurons], ['leakyrelu','leakyrelu'])
                            nn.set_patience(5)
                            now=time.time()
                            loss = nn.train(stop_function=3, num_epochs=300)
                            difference = round(time.time() - now, 5)
                            predict = nn.predict(splitted_bin_data[s])
                            pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
                            lab = splitted_labels[s] * dim_set
                            max_err = np.max(np.abs(pr-lab)).astype("int64")
                            max_errs.append(max_err)
                            print("{} neurons --> file {}, lr={}, mb={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                            .format(neurons, i, lr, mb, nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                        
                        with open("to_tex_max_more1.txt", "a+") as tex:
                             tex.write("${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(i, mb, neurons, lr, max_err))
                        
                        with open("to_tex_max_more1_det.txt", "a+") as tex:
                             tex.write("{} neurons --> file {}, lr={}, mb={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                            .format(neurons, i, lr, mb, nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                        # with open("to_tex_all_nn_remove0.txt", "a+") as tex:
                        #     tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(spl, max(max_errs), round(np.mean(max_errs),2), number_to_tex(nn.get_memory_usage(dim_set)*spl)))

                        print("-*-*"*35)

