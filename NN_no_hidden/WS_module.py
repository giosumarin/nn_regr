import numpy as np
import math
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
    return kmeans.cluster_centers_

def redefine_weights(weights,centers):
    arr_ret = np.empty_like(weights).astype(np.int16)
    for i, row in enumerate(weights):
        for j, _ in enumerate(row):
            arr_ret[i,j] = nearest_centroid_index(centers,weights[i,j])
    return arr_ret

def idx_matrix_to_matrix(idx_matrix,centers,shape):
    return centers[idx_matrix.reshape(-1,1)].reshape(shape)

def centroid_gradient_matrix(idx_matrix,gradient,cluster):
    return scipy.ndimage.sum(gradient,idx_matrix,index=range(cluster))
    #provare mean
    
def set_ws(nn, cluster, weights):
    layers_shape = []
    centers = []
    idx_layers = []
    v = []
    
    cluster = [cluster]

    layers_shape.append(weights[0][0].shape)
    centers.append(build_clusters(cluster[0], weights[0][0]))
    idx_layers.append([redefine_weights(weights[0][0],centers[0]), weights[0][1]])  
    v.append([0,0])
    NN.NN.update_layers = ws_update_layers
    NN.NN.updateMomentum = ws_updateMomentum    
    layers_shape.append(weights[0][0].shape)
    centers.append(build_clusters(cluster[0], weights[0][0]))
    idx_layers.append([redefine_weights(weights[0][0],centers[0]), weights[0][1]])  
    v.append([0,0])
    
    nn.layers_shape = layers_shape
    nn.centers = centers
    nn.idx_layers = idx_layers
    nn.v = v
    nn.epoch = 0
    nn.cluster = cluster 

    
def ws_update_layers(self, deltasUpd, momentumUpdate):
    cg = centroid_gradient_matrix(self.idx_layers[0][0],deltasUpd[0][0],self.cluster[0])
    self.v[0][0] = momentumUpdate*self.v[0][0] - np.array(cg).reshape(self.cluster[0],1) 
    self.v[0][1] = momentumUpdate*self.v[0][1] - deltasUpd[0][1]
    
    self.centers[0] += self.v[0][0] 
    bias_temp= self.idx_layers[0][1] + self.v[0][1]
    self.idx_layers[0][1] = bias_temp

    
def ws_updateMomentum(self, X, t, nEpochs, learningRate, momentumUpdate):
    numBatch = (int)(self.numEx / self.minibatch)
    lr=learningRate

    for nb in range(numBatch):
        indexLow = nb * self.minibatch
        indexHigh = (nb + 1) * self.minibatch
        self.layers = []

        self.layers.append([idx_matrix_to_matrix(self.idx_layers[0][0], self.centers[0], self.layers_shape[0]), self.idx_layers[0][1]])    
                  
        outputs = self.feedforward(X[indexLow:indexHigh])
        if self.p != None:
            for i in range(len(outputs) - 1):
                mask = (np.random.rand(*outputs[i].shape) < self.p) / self.p
                outputs[i] *= mask

        y = outputs[-1]
        deltas = []
        deltas.append(self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh]))

        
        deltasUpd = []
        deltasUpd.append([lr * (np.dot(X[indexLow:indexHigh].T, deltas[0]) + (self.layers[0][0] * self.lambd)), lr * np.sum(deltas[0], axis=0, keepdims=True)])

        self.update_layers(deltasUpd, momentumUpdate)
    

    

