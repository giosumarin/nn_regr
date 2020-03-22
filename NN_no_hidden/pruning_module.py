import numpy as np
import math
from scipy.sparse import csc_matrix, issparse
from NN_no_hidden import NN 
from NN_no_hidden import logger as log
from NN_pr import activation_function as af

# def pruning_from_perc(tax_compression, layers):
#     weights_dimension = [[w.shape[0], w.shape[1]] for [w, b] in layers]
#     nonzero_perc = [sum(tax_compression - 1/m - 1/(m*n))/2 for [m, n] in weights_dimension]
#     return 1-max(nonzero_perc)
#     #controllare e sistemare
    

class NN_pruned(NN.NN):
    def set_pruned_layers(self, pruning, weights):
        layers = weights
        mask = []
        W=layers[0][0]
        m = np.abs(W) > np.percentile(np.abs(W), pruning)
        mask.append(m)	  
        W_pruned = W * m
        layers[0][0] = W_pruned
        v=[[0, 0]]
        
        self.layers = layers
        self.mask = mask
        self.v = v
        self.epoch = 0

    def update_layers(self, deltasUpd):
        self.v[0][0] = self.mu * self.v[0][0] - self.lr * deltasUpd[0][0] * self.mask[0]
        self.v[0][1] = self.mu * self.v[0][1] - self.lr * deltasUpd[0][1]

        self.layers[0][0] += self.v[0][0] 
        self.layers[0][1] += self.v[0][1]
        
    def make_compression(self):
        self.csc_layers = [[csc_matrix(w, dtype='float32'), b] for [w,b] in self.layers]
        
    def get_memory_usage(self):
        try:
            num = sum([
                csc.data.nbytes + 
                csc.indptr.nbytes + 
                csc.indices.nbytes + 
                (bias.shape[0] * bias.shape[1]) * 4 for [csc, bias] in self.csc_layers
            ])

            kbytes = np.round(num / 1024, 4)
            return kbytes * 100 / self.numEx
        except AttributeError:
            print('Error in get_memory_usage method, pruning_module.py source')
            return -1
