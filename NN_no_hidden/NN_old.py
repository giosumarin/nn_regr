import numpy as np
import math
from sklearn.metrics import r2_score
from NN_pr import logger as log
from NN_pr import activation_function as af

N_FEATURES = 64
N_CLASSES = 1

class NN:
    def __init__(self, training, testing, lr, mu, minibatch, lambd=0, dropout=None, disableLog=None, weights=None):
        self.training = training
        self.testing = testing
        self.numEx = len(self.training[0])
        self.numTest = len(self.testing[0])
        self.lr = lr
        self.mu = mu
        self.minibatch = minibatch
        self.p = dropout
        if disableLog:
            log.logNN.disabled=True
        self.layers = weights

        self.target_train = self.training[1]
        self.target_test = self.testing[1]
        self.mask = 0
        self.epoch = 0
        
        self.layers_shape = []
        self.centers = []
        self.idx_layers = []  
        self.cluster=0      
        
        self.lambd = 0.00001
        self.patience = 5

        
        self.targetForUpd=self.target_train
        #self.targetForUpd = np.zeros((self.numEx, N_CLASSES), dtype=int)
        #for i in range(self.numEx):
        #    self.targetForUpd[i, training[1][i]] = 1
         
            
    def addLayers(self, activation_fun):
        self.epoch = 0
        self.layers = []
        self.v = []
        self.act_fun = []
        for i in range(1):
            if activation_fun[i] == 'relu':
                self.act_fun.append(lambda x, der: af.ReLU(x, der))
            elif activation_fun[i] == 'sigmoid':
                self.act_fun.append(lambda x, der: af.sigmoid(x, der))
            elif activation_fun[i] == 'linear':
                self.act_fun.append(lambda x, der: af.linear(x, der))
            elif activation_fun[i] == 'tanh':
                self.act_fun.append(lambda x, der: af.tanh(x, der))
            elif activation_fun[i] == 'leakyrelu':
                self.act_fun.append(lambda x, der: af.LReLU(x, der))
            Wo = np.random.randn(N_FEATURES, N_CLASSES) * math.sqrt(2.0 / self.numEx)
            bWo = np.random.randn(1, N_CLASSES) * math.sqrt(2.0 / self.numEx)
            self.layers.append([Wo, bWo])
            self.v.append([0, 0])
        

    def feedforward(self, X):
        outputs = []
        outputs.append(self.act_fun[-1]((np.dot(X, self.layers[-1][0]) + self.layers[-1][1]), False))
        return outputs

    def predict(self, X):
        return self.feedforward(X)[-1]

    def loss(self, X, t):
        lengthX = X.shape[0]
        predictions = self.predict(X)
        loss= np.sum((predictions-t)**2, axis=0)/lengthX
        #loss= np.sum(np.abs(predictions-t), axis=0)/lengthX
        return loss
        
        # l2_reg = 0
        # for i in range(self.nHidden + 1):
        #     l2_reg += np.sum(np.square(self.layers[i][0]))
        # return loss + self.lambd * l2_reg/(2*lengthX) 


    def updateMomentum(self, X, t, nEpochs, learningRate, momentumUpdate):
        numBatch = (int)(self.numEx / self.minibatch)
        lr = learningRate
        #lr=learningRate + 0.05/math.sqrt(self.epoch+1)

        # max_learning_rate = learningRate
        # min_learning_rate = 0.01
        # decay_speed = 1200.0
        # lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-self.epoch/decay_speed)

        for nb in range(numBatch):
            indexLow = nb * self.minibatch
            indexHigh = (nb + 1) * self.minibatch

            outputs = self.feedforward(X[indexLow:indexHigh])
            if self.p != None:
                for i in range(len(outputs) - 1):
                    mask = (np.random.rand(*outputs[i].shape) < self.p) / self.p
                    outputs[i] *= mask

            y = outputs[-1]
            deltas = []
            deltas.append(self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh]))
            #deltas.append(self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh])*1/self.minibatch)

            deltasUpd = []
            #deltasUpd.append([- lr * (np.dot(X[indexLow:indexHigh].T, deltas[0]) + (self.layers[0][0] * self.lambd * 1/self.minibatch)), - lr * np.sum(deltas[0], axis=0, keepdims=True)])
            deltasUpd.append([lr * (np.dot(X[indexLow:indexHigh].T, deltas[0]) + (self.layers[0][0] * self.lambd)), lr * np.sum(deltas[0], axis=0, keepdims=True)])

            self.update_layers(deltasUpd, momentumUpdate)

            
    def update_layers(self, deltasUpd, momentumUpdate):
        self.v[0][0] = momentumUpdate*self.v[0][0] - deltasUpd[0][0]
        self.v[0][1] = momentumUpdate*self.v[0][1] - deltasUpd[0][1]

        self.layers[0][0] += self.v[0][0]
        self.layers[0][1] += self.v[0][1]

    def stop_fun(self, t=0, num_epochs=None, loss_epoch=None):
        if t==0:
            if num_epochs > self.epoch:
                return True
            else:
                return False
        elif t==1:
            if abs(loss_epoch-self.best_loss) <= 0.1e-6:
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
            r2=r2_score(self.training[1], self.predict(self.training[0]))
            if abs(r2-self.last_r2) < .5e-6:
                return False
            else:
                if self.epoch < num_epochs:

                    self.last_r2 = r2
                    return True
                else: 
                    return False

    def train(self, stop_function, num_epochs):
        train = self.training[0]
        test = self.testing[0]
        
        self.best_loss = 100.
        last_loss = 100.
        self.last_r2= -100
            
        self.best_ep=0    
        self.real_patience = self.patience

        log.logNN.info("learning rate=" + str(self.lr) + " momentum update=" + str(self.mu) + " minibatch=" + str(self.minibatch))
        while self.stop_fun(stop_function, num_epochs, last_loss):
            self.updateMomentum(train, self.targetForUpd, num_epochs, self.lr, self.mu)
            last_loss = self.loss(train, self.target_train)

            if self.epoch % 5 == 0:
                log.logNN.debug("mse - epoch " + str(self.epoch) + ":  Train=" + str(
                    last_loss)) #+ "- Test=" + str(self.loss(test, self.target_test)))
                
            
            self.epoch += 1

        log.logNN.info("Train - epoch: " + str(self.epoch) + " loss:  " + str(self.loss(train, self.target_train)))
        #log.logNN.info("Test acc - epoch:" + str(self.loss(test, self.target_test)))
        log.logNN.debug("-------------------------------------------------------------------------------")
        return self.loss(train, self.target_train)#, self.loss(test, self.target_test)

    def getWeight(self):
        return self.layers



