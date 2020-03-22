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
from NN_no_hidden import NN_abs as NN2
from NN_no_hidden import pruning_module as pr1
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws

from utility_funs import make_structured_input_for_root_NN, make_labels_for_class, number_to_tex, remove_column_zero, reshape_nozero

N_FEATURES = 64
N_CLASSES = 1
for neurons in [32]: #16
    weights1 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(N_FEATURES, neurons)).astype(np.float32)
    bias1 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(1, neurons)).astype(np.float32)

    weights2 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(neurons, N_CLASSES)).astype(np.float32)
    bias2 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(1, N_CLASSES)).astype(np.float32)
    w= [[weights1, bias1],[weights2, bias2]]

    for i in [3,7,10]:
        with open("to_tex_all_moreneurons_nozero.txt", "a+") as tex:
            tex.write("\nfile {} \n".format(i))
            # concatenated_splits = chain(range(1,22,1), range(22,65,4), range(64,143, 8))
            concatenated_splits = [1,2,4]

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
                
                for mb in [64]:
                    max_errs = []
                    minibatchsize = mb
                        
                    for s in range(split):
                        ww = np.copy(w)
                        
                        nn = NN.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=1e-2, mu=0.9, output_classes=1, lambd=0, minibatch=minibatchsize, disableLog=True)
        
                        nn.addLayers([neurons], ["relu", "relu"], reshape_nozero(ww, first_nonzero))
                        nn.set_patience(10)
                        now=time.time()
                        loss = nn.train(stop_function=3, num_epochs=20000)
                        difference = round(time.time() - now, 5)
                        predict = nn.predict(splitted_bin_data[s])
                        pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
                        lab = splitted_labels[s] * dim_set
                        max_err = np.max(np.abs(pr-lab)).astype("int64")
                        max_errs.append(max_err)
                        print("0 hidden --> file {}, split={}, dim={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                        .format(i, spl, ceil(dim_set/split), nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                    
                    with open("to_tex_all_moreneurons_nozero.txt", "a+") as tex:
                        #tex.write("${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & {} & ${}$ \\\ \n".format(af[0], number_to_tex(lr), mb, spl, max(max_errs), round(np.mean(max_errs),2), number_to_tex(loss), number_to_tex(nn.get_memory_usage(dim_set)*spl)))
                        tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(neurons, number_to_tex(max(max_errs)), number_to_tex(loss), number_to_tex(difference)))

                    print("-*-*"*35)

