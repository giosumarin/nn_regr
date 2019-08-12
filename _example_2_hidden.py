import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py 
from math import floor

from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws
from sklearn.metrics import r2_score

start_file = 1
end_file = 9

for i in range(start_file, end_file+1):
    with h5py.File('Resource/mat_file/file'+str(i)+'_bin64.mat','r') as f:
        data = f.get('Sb') 
        bin_data = np.array(data, dtype=np.uint8)
        bin_data = np.flip(bin_data,axis=1)
        dim_set = len(bin_data)

    labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
    labels = labels/len(bin_data)
    labels = np.reshape(labels, (-1, 1))

    p = np.random.RandomState(seed=42).permutation(dim_set)

    bin_data_perm = bin_data[p]
    labels_perm = labels[p]
        

    lr=0.003 if i>2 else 0.0005
    nn = NN.NN(training=[bin_data_perm, labels_perm], testing=[[0],[0]], lr=lr, mu=0.9, lambd=1.e-5, minibatch=64, disableLog=True)

    nn.addLayers([256,256], ['leakyrelu','leakyrelu','tanh'])
    nn.train(stop_function=2, num_epochs=20000)
    print("R2: {}".format(r2_score(labels, nn.predict(bin_data))))
    max_err=0
    mean_err=0
    for j in range(dim_set):
        pr = floor(nn.predict(bin_data[j])[0]*dim_set)
        val=abs(pr-labels[j]*dim_set)
        if val > max_err:
            max_err = val
        mean_err += val

    print("2 hidden --> file {2}: max error={0} -- mean error={1}".format(max_err[0], mean_err[0]/(len(bin_data)), i))
    
    w = (nn.getWeight())
    nn_pr=pruning.NN_pruned(training=[bin_data_perm, labels_perm], testing=[[0],[0]], lr=lr, mu=0.9, lambd=1.e-5 ,minibatch=64, disableLog=True)
    
    for p in range(10,96,10):
        w1=np.copy(w)
        nn_pr.addLayers([256,256], ['leakyrelu','leakyrelu','tanh'], w1)
        nn_pr.set_pruned_layers(p, w1)
        
        nn_pr.train(stop_function=2, num_epochs=20000)
        max_err=0
        mean_err=0
        for j in range(dim_set):
            pr = floor(nn_pr.predict(bin_data[j])[0]*dim_set)
            val=abs(pr-labels[j]*dim_set)
            if val>max_err:
                max_err = val
            mean_err += val

        print("2 hidden, {3}% pruning --> file {2}: max error={0} -- mean error={1}".format(max_err[0], mean_err[0]/(len(bin_data)), i, p))

    nn_ws=ws.NN_WS(training=[bin_data_perm, labels_perm], testing=[[0],[0]], lr=lr, mu=0.9, lambd=1.e-5 ,minibatch=64, disableLog=True)
    c=[8192,2048,32] # 1/8 of original dimension of each weights' matrix
    w1=np.copy(w)
    nn_ws.addLayers([256,256], ['leakyrelu','leakyrelu','tanh'], w1)
    nn_ws.set_ws(c, w1)
    
    nn_ws.train(stop_function=2, num_epochs=20000)
    max_err=0
    mean_err=0
    for j in range(dim_set):
        pr = floor(nn_ws.predict(bin_data[j])[0]*dim_set)
        val=abs(pr-labels[j]*dim_set)
        if val>max_err:
            max_err = val
        mean_err += val

    print("2 hidden, {3} clusters --> file {2}: max error={0} -- mean error={1}".format(max_err[0], mean_err[0]/(len(bin_data)), i, c))
