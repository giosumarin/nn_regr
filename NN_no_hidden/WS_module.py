import numpy as np
import math
from math import floor
from sklearn.cluster import KMeans,MiniBatchKMeans
import scipy.ndimage
from NN_no_hidden import NN
from NN_no_hidden import logger as log
from NN_pr import activation_function as af

def nearest_centroid_index(centers,value):
    centers = np.asarray(centers)
    idx = (np.abs(centers - value)).argmin()
    return idx

def build_clusters(cluster,weights):
    kmeans = MiniBatchKMeans(n_clusters=cluster,init_size=3*cluster)
    kmeans.fit(np.hstack(weights).reshape(-1,1))
    return kmeans.cluster_centers_#.astype('float64')

def redefine_weights(weights,centers):
    if weights.shape[0] * weights.shape[1] > 256:
        arr_ret = np.empty_like(weights).astype(np.int16)
    else:
        arr_ret = np.empty_like(weights).astype(np.int8)
    for i, row in enumerate(weights):
        for j, _ in enumerate(row):
            arr_ret[i,j] = nearest_centroid_index(centers,weights[i,j])
    return arr_ret

def idx_matrix_to_matrix(idx_matrix,centers,shape):
    return centers[idx_matrix.reshape(-1,1)].reshape(shape)

def centroid_gradient_matrix(idx_matrix,gradient,cluster):
    return scipy.ndimage.sum(gradient,idx_matrix,index=range(cluster))
    #provare mean
    
def num_cluster_from_perc(layers, tax_compression):
    weights_dimension = [w.shape[0] * w.shape[1] for [w, b] in layers]
    dim = [16 if wd>256 else 8 for wd in weights_dimension]
    n_cluster = [floor((tax_compression*n*32-n*b)/32) for n, b in zip(weights_dimension, dim)]
    return n_cluster        

class NN_WS(NN.NN):
    def set_ws(self, cluster, weights):
        self.v=[[0, 0]]
        self.layers_shape = [weights[0][0].shape]
        self.centers = [build_clusters(cluster[0], weights[0][0])]
        self.idx_layers=[[redefine_weights(weights[0][0],self.centers[0]), weights[0][1]]]
        self.epoch = 0
        self.cluster = cluster 

        
    def update_layers(self, deltasUpd):
        cg = centroid_gradient_matrix(self.idx_layers[0][0],deltasUpd[0][0],self.cluster[0])
        self.v[0][0] = self.mu*self.v[0][0] + self.lr * np.array(cg).reshape(self.cluster[0],1) 
        self.v[0][1] = self.mu*self.v[0][1] + self.lr * deltasUpd[0][1]
        
        self.centers[0] -= self.v[0][0] 
        bias_temp= self.idx_layers[0][1] - self.v[0][1]
        self.idx_layers[0][1] = bias_temp

        
    def updateMomentum(self, X, t):
        numBatch = self.numEx // self.minibatch

        for nb in range(numBatch):
            indexLow = nb * self.minibatch
            indexHigh = (nb + 1) * self.minibatch

            self.layers = [[idx_matrix_to_matrix(self.idx_layers[0][0], self.centers[0], self.layers_shape[0]), self.idx_layers[0][1]]]   
                      
            outputs = self.feedforward(X[indexLow:indexHigh])
            if self.p != None:
                for i in range(len(outputs) - 1):
                    mask = (np.random.rand(*outputs[i].shape) < self.p) / self.p
                    outputs[i] *= mask

            y = outputs[-1]
            deltas = [self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh])]

            
            deltasUpd = []
            deltasUpd.append([ (np.dot(X[indexLow:indexHigh].T, deltas[0]) + (self.layers[0][0] * self.lambd)), np.sum(deltas[0], axis=0, keepdims=True)])

            self.update_layers(deltasUpd)
        

        

