import numpy as np
import math
from scipy.sparse import csc_matrix, issparse
from NN_no_hidden import NN 
from NN_no_hidden import logger as log
from NN_pr import activation_function as af

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
        self.csc_layers = [[csc_matrix(w, dtype='float32'), b] for [w,b] in self.layers]]
