import numpy as np
import math
from scipy.sparse import csc_matrix, issparse
from NN_no_hidden import NN
from NN_no_hidden import logger as log
from NN_pr import activation_function as af

def set_pruned_layers(nn, pruning, weights):
    layers = weights
    mask = []
    v = []

    NN.NN.update_layers = mask_update_layers

    W=layers[0][0]
    m = np.abs(W) > np.percentile(np.abs(W), pruning)
    mask.append(m)
    W_pruned = W * m
    layers[0][0] = W_pruned
    v.append([0, 0])
    nn.layers = layers
    nn.mask = mask
    nn.v = v
    nn.epoch = 0

def mask_update_layers(self, deltasUpd, momentumUpdate):
    self.v[0][0] = momentumUpdate*self.v[0][0] - deltasUpd[0][0]
    self.v[0][1] = momentumUpdate*self.v[0][1] - deltasUpd[0][1]

    self.layers[0][0] += self.v[0][0] * self.mask[0]
    self.layers[0][1] += self.v[0][1]
