import numpy as np
import math
from scipy.sparse import csc_matrix, issparse
from NN_pr import NN 
from NN_pr import logger as log
from NN_pr import activation_function as af

class NN_pruned(NN.NN):
   
    def set_pruned_layers(self, pruning, weights):
        layers = weights
        num_layers = len(layers)
        mask = []
        v = []
	        
        for i in range(num_layers):
            W=layers[i][0]
            m = np.abs(W) > np.percentile(np.abs(W), pruning)
            mask.append(m)	  
            W_pruned = W * m
            layers[i][0] = W_pruned
            v.append([0, 0])
        self.layers = layers
        self.mask = mask
        self.v = v
        self.epoch = 0

    def update_layers(self, deltasUpd):

        for i in range(self.nHidden + 1):
            self.v[i][0] = self.mu * self.v[i][0] - self.lr * deltasUpd[i][0]
            self.v[i][1] = self.mu * self.v[i][1] - self.lr * deltasUpd[i][1]

        for i in range(self.nHidden + 1):
            self.layers[i][0] += self.v[i][0] * self.mask[i]
            self.layers[i][1] += self.v[i][1]
