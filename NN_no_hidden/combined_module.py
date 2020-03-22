import numpy as np
import math
from math import floor
from sklearn.cluster import KMeans,MiniBatchKMeans
import scipy.ndimage
from NN_no_hidden import NN
from NN_no_hidden import WS_module
from NN_no_hidden import logger as log
from NN_pr import activation_function as af
from NN_no_hidden import loss_function

def nearest_centroid_index(centers,value):
    centers = np.asarray(centers)
    idx = (np.abs(centers - value)).argmin()
    return idx

def build_clusters(cluster,weights):
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(np.hstack(weights).reshape(-1,1))
    #_, freq = np.unique(kmeans.cluster_centers_, return_counts=True)
    #print("unici centri = {}".format(len(freq)))
    return kmeans.cluster_centers_.astype('float32')

def redefine_weights(weights,centers):
    if weights.shape[0] * weights.shape[1] > 256:
        arr_ret = np.empty_like(weights).astype(np.uint16)
    else:
        arr_ret = np.empty_like(weights).astype(np.uint8)
    for i, row in enumerate(weights):
        for j, _ in enumerate(row):
            arr_ret[i,j] = nearest_centroid_index(centers,weights[i,j])
    return arr_ret

def idx_matrix_to_matrix(idx_matrix,centers,shape):
    return centers[idx_matrix.reshape(-1,1)].reshape(shape)

def centroid_gradient_matrix(idx_matrix,gradient,cluster, mask):
    gradient += 0.000000001
    gradient[np.logical_not(mask)] = 0
    return scipy.ndimage.sum(gradient,idx_matrix,index=range(cluster))
    #provare mean
    
def num_cluster_from_perc(layers, tax_compression):
    weights_dimension = [w.shape[0] * w.shape[1] for [w, b] in layers]
    dim = [16 if wd>256 else 8 for wd in weights_dimension]
    n_cluster = [floor((tax_compression*n*32-n*b)/32) for n, b in zip(weights_dimension, dim)]
    return n_cluster        

class NN_combined(WS_module.NN_WS):
    def set_combined_compression(self, pruning, cluster, weights):
        
        layers = weights
        mask = []
        self.v = [[0,0]]
        W = layers[0][0]
        m = np.abs(W) > np.percentile(np.abs(W), pruning)
        mask.append(m)	  
        W_pruned = W * m
        layers[0][0] = W_pruned
        self.layers = layers
        self.mask = mask
        
        self.layers_shape = layers[0][0].shape
        self.centers = [build_clusters(cluster[0], self.layers[0][0])]
        self.idx_layers=[[redefine_weights(self.layers[0][0],self.centers[0]), self.layers[0][1]]]
        #print(self.idx_layers[0][0].shape)
        self.epoch = 0
        self.cluster = cluster

        
    def update_layers(self, deltasUpd):
        cg = centroid_gradient_matrix(self.idx_layers[0][0],deltasUpd[0][0],self.cluster[0], self.mask[0])
        self.v[0][0] = self.mu*self.v[0][0] + self.lr * np.array(cg).reshape(self.cluster[0],1) 
        self.v[0][1] = self.mu*self.v[0][1] + self.lr * deltasUpd[0][1]
        
        self.centers[0] -= self.v[0][0] 
        bias_temp = self.idx_layers[0][1] - self.v[0][1]
        self.idx_layers[0][1] = bias_temp
        
        
        
    def updateMomentum(self, X, t):
        numBatch = self.numEx // self.minibatch

        for nb in range(numBatch):
            indexLow = nb * self.minibatch
            indexHigh = (nb + 1) * self.minibatch

            self.layers = [[idx_matrix_to_matrix(self.idx_layers[0][0], self.centers[0], self.layers_shape[0]).reshape(-1,self.N_CLASSES), self.idx_layers[0][1]]]   
            #print(self.layers[0][0].shape)
                      
            outputs = self.feedforward(X[indexLow:indexHigh])
            if self.p != None:
                mask = (np.random.rand(*outputs[0].shape) < self.p) / self.p
                outputs[0] *= mask

            y = outputs[-1]
            #print(self.act_fun[-1](y, True).shape)
            #print(self.loss_fun(y, t[indexLow:indexHigh], True).shape)
            #print((self.act_fun[-1](y, True) * self.loss_fun(y, t[indexLow:indexHigh], True)).shape)
            deltas = [self.act_fun[-1](y, True) * self.loss_fun(y, t[indexLow:indexHigh], True)]

            #print(deltas[0].shape)
            deltasUpd = []
            deltasUpd.append([ (np.dot(X[indexLow:indexHigh].T, deltas[0]) + (self.layers[0][0] * self.lambd)), np.sum(deltas[0], axis=0, keepdims=True)])

            self.update_layers(deltasUpd)
        

