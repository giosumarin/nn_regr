import numpy as np
import math
from math import floor
from sklearn.cluster import KMeans,MiniBatchKMeans
import scipy.ndimage
from scipy.sparse import csc_matrix, issparse
from NN_pr import NN 
from NN_pr import logger as log
from NN_pr import activation_function as af
from scipy import sparse

def nearest_centroid_index(centers,value):
    centers = np.asarray(centers)
    idx = (np.abs(centers - value)).argmin()
    return idx

def build_clusters(cluster,weights):
    #kmeans = MiniBatchKMeans(n_clusters=cluster,init_size=3*cluster, max_no_improvement=None, tol=0.)
    pesi, freq = np.unique(weights, return_counts=True)
    print("unici pesi = {}".format(len(freq)))
    #print(pesi)
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(np.hstack(weights).reshape(-1,1))
    pesi, freq = np.unique(kmeans.cluster_centers_, return_counts=True)
    print("unici centri = {}".format(len(freq)))
    #print(pesi)
    return kmeans.cluster_centers_.astype('float32')

def redefine_weights(weights,centers):
    #if weights.shape[0] * weights.shape[1] > 256:
    #    arr_ret = np.empty_like(weights).astype(np.uint16)
    #else:
    arr_ret = np.empty_like(weights).astype(np.uint64)
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


class NN_combined(NN.NN):

    def set_combined_compression(self, pruning, cluster, weights):
        
        layers = weights
        num_layers = len(layers)
        mask = []
        self.v = [[0,0] for _ in range(self.nHidden+1)]
        for i in range(num_layers):
            W = layers[i][0]
            m = np.abs(W) > np.percentile(np.abs(W), pruning)
            mask.append(m)	  
            W_pruned = W * m
            layers[i][0] = W_pruned
        self.layers = layers
        self.mask = mask
        self.epoch = 0
        
        if isinstance(cluster, int):
            cluster = [cluster]*len(weights)
        
        self.layers_shape = [layers[i][0].shape for i in range(self.nHidden+1)]
        self.centers = [build_clusters(cluster[i], self.layers[i][0]) for i in range(self.nHidden+1)]
        self.idx_layers=[[redefine_weights(self.layers[i][0],self.centers[i]), self.layers[i][1]] for i in range(self.nHidden+1)]
        self.epoch = 0
        self.cluster = cluster
        
        
        
    def update_layers(self, deltasUpd):
        for i in range(self.nHidden + 1):
            cg = centroid_gradient_matrix(self.idx_layers[i][0],deltasUpd[i][0],self.cluster[i], self.mask[i])
            self.v[i][0] = self.mu*self.v[i][0] + self.lr*np.array(cg).reshape(self.cluster[i],1)
            self.v[i][1] = self.mu*self.v[i][1] + self.lr*deltasUpd[i][1]
            
        

        for i in range(self.nHidden + 1):
            self.centers[i] -= self.v[i][0] 
            bias_temp = self.idx_layers[i][1] - self.v[i][1]
            self.idx_layers[i][1] = bias_temp
        
        
    def updateMomentum(self, X, t):
        numBatch = self.numEx // self.minibatch
        remain_elements = self.numEx % self.minibatch
        if numBatch <= 1:
            distribuited_elements = [0 for i in range(numBatch)]
            indexes_minibatchs = [[0, self.numEx]]
        else:
            distribuited_elements = [1 if remain_elements - i > 0 else 0 for i in range(1, numBatch+1)]
            while (remain_elements > sum(distribuited_elements)):
                distribuited_elements = [distribuited_elements[i]+1 if sum(distribuited_elements)+i < remain_elements else distribuited_elements[i] for i in range(0, numBatch)]
            adjusted_batch_sizes = [self.minibatch+distribuited_elements[i] for i in range(numBatch)]    
            indexes_minibatchs = [[sum(adjusted_batch_sizes[:i]), sum(adjusted_batch_sizes[:i+1])] for i in range(numBatch)]

        for [indexLow,indexHigh] in indexes_minibatchs:
            size_minibatch = indexHigh-indexLow
            
            self.layers = []
            for i in range(self.nHidden + 1):
                self.layers.append([idx_matrix_to_matrix(self.idx_layers[i][0], self.centers[i], self.layers_shape[i]), self.idx_layers[i][1]])    #self.layers_shape[i]) * self.mask[i]
                    
            outputs = self.feedforward(X[indexLow:indexHigh])
            
            if self.p != None:
                for i in range(len(outputs) - 1):
                    mask = (np.random.rand(*outputs[i].shape) < self.p) / self.p
                    outputs[i] *= mask

            y = outputs[-1]

            deltas = [self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh]) * 2/size_minibatch]
            for i in range(self.nHidden):
                deltas.append(np.dot(deltas[i], self.layers[self.nHidden - i][0].T) * self.act_fun[self.nHidden - i - 1](outputs[self.nHidden - i - 1], True))
            deltas.reverse()

            outputs_for_deltas = [X[indexLow:indexHigh]]+outputs[:-1] 

            deltas_weights = [np.dot(outputs_for_deltas[i].T, deltas[i]) + (self.layers[i][0] * self.lambd * 2/size_minibatch) for i in range(self.nHidden + 1)]
            deltas_bias = [np.sum(deltas[i], axis=0, keepdims=True) for i in range(self.nHidden + 1)]
            deltasUpd = [[w,b] for w, b in list(zip(deltas_weights, deltas_bias))]

            self.update_layers(deltasUpd)
