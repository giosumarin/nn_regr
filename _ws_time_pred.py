import pickle
#import cdot
import gzip
import numpy as np
import sys
import h5py
import time 
from math import floor
from joblib import Parallel, delayed


from NN_no_hidden import NN as NN1
from NN_no_hidden import pruning_module as pr1
from NN_no_hidden import WS_module as ws_m
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws
from NN_pr import activation_function as af

percList = [10,50,80]
distr = "uniform"

def predict(index, center, bias, a):
    c = np.zeros((a.shape[0], index.shape[1]), dtype="float32")
    '''for i in range(a.shape[0]):
        for k in range(index.shape[1]):
            c[i][k] = np.dot(a[i], center[index[:,k]])
    '''
    #for k in range(index.shape[1]):
    c = np.dot(a, center[index].reshape(*index.shape))
    c += bias
    return af.LReLU(c)
    

for i in [3,7,10]:
     for perc in percList:
        #print("File"+str(i)+ " query "+str(perc))
        with h5py.File('./Query/file'+str(i)+distr+'Query'+str(perc)+'_bin.mat','r') as f:
            data = f.get('Sb') 
            bin_data = np.array(data, dtype=np.bool) # For converting to numpy array
        bin_data = np.transpose(bin_data)
        bin_data = np.flip(bin_data,axis=1)
        dim_set = len(bin_data)
        for p in range(30,91,10):
            with open('NN1/nn1_file{}_ws{}'.format(i, p), 'rb') as f:
                ws1 = pickle.load(f)
             
            num_layers = len(ws1[0])
            center=[ws1[1][j] for j in range(num_layers)]
            index = [ws1[0][j][0] for j in range(num_layers)]
            bias = [ws1[0][j][1] for j in range(num_layers)]
            inputs = bin_data
            
            now = time.time()
            for j in range(num_layers):
                outputs=predict(index[j], center[j], bias[j], inputs)
                inputs = outputs
            t1 = time.time()-now             
            
            with open("res_nn1_pred_ws.txt", "a+") as mf:
                    mf.write("NN_1 file{} perc{} compr{} {}ms\n".format(i,perc,p,t1*1000))
           
        for p in range(60,91,10):
            with open('NN2/nn2_file{}_ws{}'.format(i, p), 'rb') as f:
                ws1 = pickle.load(f)
                 
            num_layers = len(ws1[0])
            center=[ws1[1][j] for j in range(num_layers)]
            index = [ws1[0][j][0] for j in range(num_layers)]
            bias = [ws1[0][j][1] for j in range(num_layers)]
            inputs = bin_data
            now = time.time()
            for j in range(num_layers):
                 outputs = predict(index[j], center[j], bias[j], inputs)
                 inputs = outputs
            t1 = time.time()-now

            with open("res_nn2_pred_ws.txt", "a+") as mf:
                    mf.write("NN_2 file{} perc{} compr{} {}ms\n".format(i,perc,p,t1*1000))
                    
            with open('NN3/nn3_file{}_ws{}'.format(i, p), 'rb') as f:
                ws1 = pickle.load(f)
                 
            num_layers = len(ws1[0])
            center=[ws1[1][j] for j in range(num_layers)]
            index = [ws1[0][j][0] for j in range(num_layers)]
            bias = [ws1[0][j][1] for j in range(num_layers)]
            inputs = bin_data
            now = time.time()
            for j in range(num_layers):
                 outputs = predict(index[j], center[j], bias[j], inputs)
                 inputs = outputs
            t1 = time.time()-now
            with open("res_nn3_pred_ws.txt", "a+") as mf:
                    mf.write("NN_3 file{} perc{} compr{} {}ms\n".format(i,perc,p,t1*1000))
            
          
    


