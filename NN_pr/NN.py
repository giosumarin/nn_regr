# 1 hidden --> file 3, dim dataset=512: max error=3.0 -- % error=0.5859375 -- time of train=39s
# 2 hidden --> file 3, dim dataset=512: max error=1.0 -- % error=0.1953125 -- time of train=32s
# 1 hidden --> file 7, dim dataset=8192: max error=45.0 -- % error=0.54931640625 -- time of train=29s
# 2 hidden --> file 7, dim dataset=8192: max error=36.0 -- % error=0.439453125 -- time of train=413s


import numpy as np
import math
from sklearn.metrics import r2_score
from NN_pr import logger as log
from NN_pr import activation_function as af

N_FEATURES = 64
N_CLASSES = 1
np.random.RandomState(42)

class NN:
    def __init__(self, training, testing, lr, mu, minibatch, lambd=0, dropout=None, disableLog=None, weights=None):
        self.train_set = training[0]
        self.test = testing[0]
        self.numEx = len(self.train_set)
        self.numTest = len(self.test)
        self.lr = lr
        self.mu = mu
        self.minibatch = minibatch
        self.p = dropout
        if disableLog:
            log.logNN.disabled=True
        self.layers = weights

        self.target_train = training[1]
        self.target_test = testing[1]
        self.epoch = 0
  
        
        self.lambd = lambd
        self.patience = 5     
            
    def addLayers(self, neurons, activation_fun, weights=None):
        self.epoch = 0
        log.logNN.info("neurons= "+str(neurons))
        self.nHidden = len(neurons)
        self.layers = []
        self.v = [[0,0] for _ in range(self.nHidden+1)]

        act_fun_factory = {"relu": lambda x, der: af.ReLU(x, der),
                           "sigmoid": lambda x, der: af.sigmoid(x, der),
                           "linear": lambda x, der: af.linear(x, der),
                           "tanh": lambda x, der: af.tanh(x, der),
                           "leakyrelu": lambda x, der: af.LReLU(x, der)}      
        self.act_fun = [act_fun_factory[f] for f in activation_fun]
        
        if weights == None:
            weights_hidden_shapes = list(zip([N_FEATURES]+neurons[:-1], neurons))   
            weights_hidden = [np.random.randn(row, col) * math.sqrt(2.0 / self.numEx) for row, col in weights_hidden_shapes] 
            #bias_hidden = [np.random.randn(1, n) * math.sqrt(2.0 / self.numEx) for n in neurons]
            #weights_hidden = [np.random.normal(scale=0.05, size=(row, col)) for row, col in weights_hidden_shapes] 
            #bias_hidden = [np.random.normal(scale=0.05, size=(1, n)) for n in neurons]
            bias_hidden = [np.ones((1, n))*0.001 for n in neurons]
            self.layers = [[w,b] for w, b in list(zip(weights_hidden, bias_hidden))]
            Wo = np.random.randn(neurons[-1], N_CLASSES) * math.sqrt(2.0 / self.numEx)
            #Wo = np.random.normal(scale=0.05,size=(neurons[-1], N_CLASSES))
            # bWo = np.random.randn(1, N_CLASSES) * math.sqrt(2.0 / self.numEx)
            bWo = np.ones((1, N_CLASSES))*0.001
            self.layers += [[Wo,bWo]]
        else:
            self.layers=weights
        
    def set_lambda_reg_l2(self, lambd):
        self.lambd=lambd

    def feedforward(self, X):
        outputs = []
        inputLayer = X
        for i in range(self.nHidden + 1):             
            H = self.act_fun[i](np.dot(inputLayer, self.layers[i][0]) + self.layers[i][1], False)
            outputs.append(H)
            inputLayer = H
        return outputs


    def predict(self, X):
        return self.feedforward(X)[-1]


    def loss(self, X, t):
        #predictions = self.predict(X)
        #loss = np.mean((predictions-t)**2, axis=0)
        #return loss
        predictions = self.predict(X)
        loss = np.mean(np.abs(predictions-t))
        #print(loss)
        return loss


    def updateMomentum(self, X, t):
        numBatch = self.numEx // self.minibatch

        for nb in range(numBatch):
            indexLow = nb * self.minibatch
            indexHigh = (nb + 1) * self.minibatch

            outputs = self.feedforward(X[indexLow:indexHigh])
            
            if self.p != None:
                for i in range(self.nHidden):
                    mask = (np.random.rand(*outputs[i].shape) < self.p) / self.p
                    outputs[i] *= mask

            y = outputs[-1]
            
            deltas = [self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh])]
            for i in range(self.nHidden):
                deltas.append(np.dot(deltas[i], self.layers[self.nHidden - i][0].T) * self.act_fun[self.nHidden - i - 1](outputs[self.nHidden - i - 1], True))
            deltas.reverse()
            
            outputs_for_deltas = [X[indexLow:indexHigh]]+outputs[:-1] 

            deltas_weights = [np.dot(outputs_for_deltas[i].T, deltas[i]) + (self.layers[i][0] * self.lambd) for i in range(self.nHidden + 1)]
            deltas_bias = [np.sum(deltas[i], axis=0, keepdims=True) for i in range(self.nHidden + 1)]
            deltasUpd = [[w,b] for w, b in list(zip(deltas_weights, deltas_bias))]

            self.update_layers(deltasUpd)

            
    def update_layers(self, deltasUpd):
        for i in range(self.nHidden + 1):
            self.v[i][0] = self.mu * self.v[i][0] - self.lr * deltasUpd[i][0]
            self.v[i][1] = self.mu * self.v[i][1] - self.lr * deltasUpd[i][1]
        for i in range(self.nHidden + 1):
            self.layers[i][0] += self.v[i][0] 
            self.layers[i][1] += self.v[i][1] 

    def stop_fun(self, t=0, num_epochs=None, loss_epoch=None):
        if t==0:
            if num_epochs > self.epoch:
                return True
            else:
                return False
        elif t==1:
            if abs(loss_epoch-self.best_loss) <= 1e-7:
                self.real_patience -= 1
                if self.real_patience == 0:
                    return False
                else:
                    return True
            else:
                if self.best_loss > loss_epoch:
                    self.best_loss=loss_epoch
                    self.best_ep=self.epoch
                if self.epoch < num_epochs:
                    self.real_patience = self.patience
                    return True
                else:
                    return False 
        elif t == 2:
            r2=r2_score(self.target_train, self.predict(self.train_set))
            if abs(r2-self.last_r2) < 1e-8: 
                self.real_patience -= 1
                if self.real_patience==0:
                    return False
            else:
                if self.epoch < num_epochs:
                    self.last_r2 = r2
                    self.real_patience = self.patience
                    return True
                else:
                    return False
        elif t==3:
            if self.best_loss - loss_epoch <= 0 :
                self.real_patience += 1
                if self.real_patience == self.patience:
                    return False
                else:
                    return True
            else:
                if self.best_loss > loss_epoch:
                    self.best_loss=loss_epoch
                if self.epoch < num_epochs:
                    self.real_patience = 0
                    return True
                else:
                    return False 


    def train(self, stop_function, num_epochs):
        
        self.best_loss = 100.
        last_loss = 99.
        self.last_r2= -100
            
        self.best_ep=0    
        self.real_patience = 0

        log.logNN.info("learning rate=" + str(self.lr) + " momentum update=" + str(self.mu) + " minibatch=" + str(self.minibatch))
        while self.stop_fun(stop_function, num_epochs, last_loss):
            self.updateMomentum(self.train_set, self.target_train)
            last_loss = self.loss(self.train_set, self.target_train)

            if self.epoch % 1 == 0:
                log.logNN.debug("Train - epoch {0} - MSE {1}".format(self.epoch, last_loss)) 
                
            
            self.epoch += 1

        log.logNN.info("Train - epoch {0} - MSE {1}".format(self.epoch, last_loss))
        log.logNN.debug("-------------------------------------------------------------------------------")
        return last_loss


    def getWeight(self):
        return self.layers




