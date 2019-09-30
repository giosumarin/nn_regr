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
from NN_pr import activation_function as af

def predict_with_csc1(csc_layers, data):
    now = time.time()
    y=af.LReLU((csc_layers[0][0].T.dot(data.T)).T + csc_layers[0][1])
    end_time=time.time()-now
    return y, end_time
    
def predict_with_csc2(csc_layers, data):
    now = time.time()
    h1 = af.LReLU((csc_layers[0][0].T.dot(data.T)).T + csc_layers[0][1])
    y = af.LReLU((csc_layers[1][0].T.dot(h1.T)).T + csc_layers[1][1]) 
    end_time=time.time()-now
    return y, end_time
    
def predict_with_csc3(csc_layers, data):
    now = time.time()
    h1 = af.LReLU((csc_layers[0][0].T.dot(data.T)).T + csc_layers[0][1])
    h2 = af.LReLU((csc_layers[1][0].T.dot(h1.T)).T + csc_layers[1][1])
    y = af.LReLU((csc_layers[2][0].T.dot(h1.T)).T + csc_layers[2][1])
    end_time=time.time()-now
    return y, end_time

percList = [10,50,80]
distr = "uniform"

for i in [3,7,10]:
    for p in range(10,96,10):
        with open('NN1/nn1_file{}_pr{}'.format(i, p), 'rb') as f:
            w_csc1 = pickle.load(f)
        with open('NN2/nn2_file{}_pr{}'.format(i, p), 'rb') as f:
            w_csc2 = pickle.load(f)
        with open('NN3/nn3_file{}_pr{}'.format(i, p), 'rb') as f:
            w_csc3 = pickle.load(f)
            
        for perc in percList:
            print("File"+str(i)+ " query "+str(perc))
            bin_data = []
            labels = []
            with h5py.File('./Query/file'+str(i)+distr+'Query'+str(perc)+'_bin.mat','r') as f:
                data = f.get('Sb') 
                bin_data = np.array(data, dtype=np.bool) # For converting to numpy array
            bin_data = np.transpose(bin_data)
            bin_data = np.flip(bin_data,axis=1)
            dim_set = len(bin_data)
            
            _, t1 = predict_with_csc1(w_csc1, bin_data)
            _, t2 = predict_with_csc2(w_csc2, bin_data)
            _, t3 = predict_with_csc3(w_csc3, bin_data)
            
            with open("res_nn1_pred_pr.txt", "a+") as mf:
                mf.write("NN_1 file{} perc{} pr{}% {}ms\n".format(i, perc, p, t1*1000))
            with open("res_nn2_pred_pr.txt", "a+") as mf:
                mf.write("NN_2 file{} perc{} pr{}% {}ms\n".format(i, perc, p, t2*1000))
            with open("res_nn3_pred_pr.txt", "a+") as mf:
                mf.write("NN_3 file{} perc{} pr{}% {}ms\n".format(i, perc, p, t3*1000))
