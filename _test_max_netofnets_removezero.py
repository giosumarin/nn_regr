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

from NN_no_hidden import NN as NN1
from NN_no_hidden import NN_max as NN2
from NN_no_hidden import pruning_module as pruning
#from NN_pr import NN
#from NN_pr import pruning_module as pruning
from NN_no_hidden import WS_module as ws
from NN_no_hidden import combined_module as comb

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
        max_errs = []
            
        #riga doppia su file 3 512 == dim_set
        for lr in [0.1]: #[1e-5,1e-4,1e-3,1e-2,1e-1]:
            for mb in [512]: #[64,128,256,512,1024,dim_set]:
                for s in range(split):
                    for l in ["LSE"]:
                        ww = np.copy(w)
                        nn = NN1.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=lr, mu=.9, output_classes=1, lambd=0, minibatch=mb, disableLog=True)
                        nn.addLayers(activation_fun=['leakyrelu'], loss=l, weights=reshape_nozero(ww, first_nonzero))
                        #nn.addLayers(activation_fun=['leakyrelu'], weights=reshape_nozero(ww, first_nonzero))
                        nn.set_patience(10)
                        now=time.time()
                        loss = nn.train(stop_function=3, num_epochs=20000, total_example=dim_set)
                        difference = round(time.time() - now, 5)
                        predict = nn.predict(splitted_bin_data[s])
                        pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
                        lab = splitted_labels[s] * dim_set
                        max_err = np.max(np.abs(pr-lab)).astype("int64")
                        max_errs.append(max_err)
                        print("0 hidden --> file {}, lr={}, mb={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                        .format(i, lr, mb, nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                        
                        for wss in [10,12,14,16]:
                            nn_compr = ws.NN_WS(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.01, mu=.9, output_classes=1, lambd=0, minibatch=mb, disableLog=True)
                            nn_compr.addLayers(activation_fun=['leakyrelu'], loss=l, weights=np.copy(nn.layers))
                            nn_compr.set_ws([wss],np.copy(nn.layers))
                            nn_compr.set_patience(10)
                            loss = nn_compr.train(stop_function=3, num_epochs=1000, total_example=dim_set)
                            difference = round(time.time() - now, 5)
                            predict = nn_compr.predict(splitted_bin_data[s])
                            pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
                            lab = splitted_labels[s] * dim_set
                            max_err = np.max(np.abs(pr-lab)).astype("int64")
                            max_errs.append(max_err)
                            print("{}ws --> file {}, lr={}, mb={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                            .format(wss,i, lr, mb, nn_compr.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn_compr.get_memory_usage(),5)))
                            
                            with open("to_tex_max_nohidden_ws.txt", "a+") as tex:
                                    tex.write("${}$ & ${}$ \\\ \n".format(wss, max_err))
                        
                        for p in [10,20,30,40,50,60,70,80,90]:
                            nn_compr = pruning.NN_pruned(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.01, mu=.9, output_classes=1, lambd=0, minibatch=mb, disableLog=True)
                            nn_compr.addLayers(activation_fun=['leakyrelu'], loss=l, weights=np.copy(nn.layers))
                            nn_compr.set_pruned_layers(p,np.copy(nn.layers))
                            nn_compr.set_patience(10)
                            loss = nn_compr.train(stop_function=3, num_epochs=1000, total_example=dim_set)
                            difference = round(time.time() - now, 5)
                            predict = nn_compr.predict(splitted_bin_data[s])
                            pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
                            lab = splitted_labels[s] * dim_set
                            max_err = np.max(np.abs(pr-lab)).astype("int64")
                            max_errs.append(max_err)
                            print("{}%  --> file {}, lr={}, mb={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                            .format(p, i, lr, mb, nn_compr.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn_compr.get_memory_usage(),5)))
                            
                            with open("to_tex_max_nohidden_prun.txt", "a+") as tex:
                                    tex.write("${}%$ & ${}$ \\\ \n".format(p, max_err))
                            
                        for wss in [10,12,14,16]:
                            for p in [10,20,30,40,50]:
                        #p = 20
                        #wss = 12  
                                nn_compr = comb.NN_combined(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.01, mu=.9, output_classes=1, lambd=0, minibatch=mb, disableLog=True)
                                nn_compr.addLayers(activation_fun=['leakyrelu'], loss=l, weights=np.copy(nn.layers))
                                nn_compr.set_combined_compression(p,[wss],np.copy(nn.layers))
                                nn_compr.set_patience(10)
                                loss = nn_compr.train(stop_function=3, num_epochs=1000, total_example=dim_set)
                                difference = round(time.time() - now, 5)
                                predict = nn_compr.predict(splitted_bin_data[s])
                                pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
                                lab = splitted_labels[s] * dim_set
                                max_err = np.max(np.abs(pr-lab)).astype("int64")
                                max_errs.append(max_err)
                                print("{}% {}cluster  --> file {}, lr={}, mb={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                                .format(p, wss, i, lr, mb, nn_compr.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn_compr.get_memory_usage(),5)))
                                
                                with open("to_tex_max_nohidden_comb.txt", "a+") as tex:
                                    tex.write("${}%$ & ${}$ & ${}$ \\\ \n".format(p, wss, max_err))
                            
                        
                        
                        
                        #with open("to_tex_max_nohidden.txt", "a+") as tex:
                        #     tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(i, lr, mb, max_err))
                             
                        with open("to_tex_max_nohidden_det.txt", "a+") as tex:
                             tex.write("0 hidden --> file {}, lr={}, mb={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={} \n"
                        .format(i, lr, mb, nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(),5)))
        
        # with open("to_tex_all_nn_remove0.txt", "a+") as tex:
        #     tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(spl, max(max_errs), round(np.mean(max_errs),2), number_to_tex(nn.get_memory_usage(dim_set)*spl)))

        print("-*-*"*35)

