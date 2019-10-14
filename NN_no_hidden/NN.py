import numpy as np
import math
from sklearn.metrics import r2_score
from NN_pr import logger as log
from NN_pr import activation_function as af

N_FEATURES = 64
N_CLASSES = 1
np.random.RandomState(0)

class NN:
    def __init__(self, training, testing, lr, mu, output_classes, lambd=0, minibatch=None, dropout=None, disableLog=None, weights=None):
        self.train_set = training[0]
        self.test = testing[0]
        self.numEx = len(self.train_set)
        self.numTest = len(self.test)
        self.lr = lr
        self.mu = mu
        if not minibatch:
            self.minibatch = self.numEx
        else:
            self.minibatch = minibatch
        self.p = dropout
        self.disableLog=disableLog
        if disableLog:
            log.logNN.disabled=True
        self.layers = weights

        self.target_train = training[1]
        self.target_test = testing[1]
        self.epoch = 0
        self.N_CLASSES = output_classes
  
        
        self.lambd = lambd
        self.patience = 25
        self.patience_5 = 100   
            
    def addLayers(self, activation_fun,weights=None):
        self.epoch = 0
        self.v = [[0,0]]
        act_fun_factory = {"relu": lambda x, der: af.ReLU(x, der),
                           "sigmoid": lambda x, der: af.sigmoid(x, der),
                           "linear": lambda x, der: af.linear(x, der),
                           "tanh": lambda x, der: af.tanh(x, der),
                           "leakyrelu": lambda x, der: af.LReLU(x, der),
                           "softmax": lambda x, der: af.softmax_function(x, der),
                           "softplus": lambda x, der: af.softplus_function(x,der)}      
        self.act_fun = [act_fun_factory[f] for f in activation_fun]
        if weights == None:
            Wo = np.random.randn(N_FEATURES, self.N_CLASSES).astype(np.float32) * math.sqrt(2/N_FEATURES)
            #Wo = (np.zeros((N_FEATURES, self.N_CLASSES))).astype(np.float32) 
            #Wo = np.random.normal(scale=0.01, size=(N_FEATURES, self.N_CLASSES)).astype(np.float32) 
            #Wo = np.random.normal(scale=0.001, size=(N_FEATURES, self.N_CLASSES)).astype(np.float32)
            bWo = np.ones((1, self.N_CLASSES)).astype(np.float32)*0.0001
            self.layers = [[Wo,bWo]]
        else:
            self.layers=weights
        
    def set_lambda_reg_l2(self, lambd):
        self.lambd=lambd

    def feedforward(self, X):
        inputLayer = X            
        return [self.act_fun[0](np.dot(inputLayer, self.layers[0][0]) + self.layers[0][1], False)]


    def predict(self, X):
        return self.feedforward(X)[-1]


    def loss(self, X, t):
        #predictions = self.predict(X)
        #loss = np.mean((predictions-t)**2, axis=0)
        #return loss

        #scegliere soglia e fare media pesat
        tr = 2 ** 5 
        if X.shape[0] > 2**1000:
            loss = 0
            batch=self.numEx // tr
            for n in range(batch):
                indexLow = n * tr
                indexHigh =(n+1) * tr 
                predictions = self.predict(X[indexLow:indexHigh])
                loss += np.mean(np.abs(predictions - t[indexLow:indexHigh]))
                
            loss/=batch
        else:
            predictions = self.predict(X)
            loss = np.mean(np.square(predictions-t)) #np.abs
        return loss #np.round(loss, 7)


    def updateMomentum(self, X, t):
        numBatch = self.numEx // self.minibatch

        #p = np.random.RandomState(seed=0).permutation(self.numEx)
        #self.train_set = self.train_set[p]
        #self.target_train =  self.target_train[p]


        for nb in range(numBatch):
            indexLow = nb * self.minibatch
            indexHigh = (nb + 1) * self.minibatch

            outputs = self.feedforward(X[indexLow:indexHigh])
            
            if self.p != None:
                
                mask = (np.random.rand(*outputs[0].shape) < self.p) / self.p
                outputs[0] *= mask

            y = outputs[-1]
            
            deltas = [self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh]) * 1/self.minibatch]

            deltasUpd= [([(np.dot(X[indexLow:indexHigh].T, deltas[0]) + (1/self.minibatch * self.layers[0][0] * self.lambd)), np.sum(deltas[0], axis=0, keepdims=True)])]

            self.update_layers(deltasUpd)

            
    # def update_layers(self, deltasUpd):
    #     lr = self.lr
    #     self.v[0][0] = self.mu * self.v[0][0] + lr * deltasUpd[0][0]
    #     self.v[0][1] = self.mu * self.v[0][1] + lr * deltasUpd[0][1]
        
    #     self.layers[0][0] -= self.v[0][0] 
    #     self.layers[0][1] -= self.v[0][1] 

    def update_layers(self, deltaUpd):
        v_prev = self.v.copy()
        self.v[0][0] = self.mu * self.v[0][0] - self.lr * deltaUpd[0][0]
        self.v[0][1] = self.mu * self.v[0][1] - self.lr * deltaUpd[0][1]
        
        self.layers[0][0] += -self.mu * v_prev[0][0] + (1+self.mu) * self.v[0][0] 
        self.layers[0][1] += -self.mu * v_prev[0][1] + (1+self.mu) * self.v[0][1]

    def exp_decay(self):
        k = .001
        return self.lr*math.exp(-k * self.epoch)


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
            if abs(r2-self.last_r2) < 1e-10: 
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
            # if self.epoch <= 50:
            #     return True
            if (self.best_loss - loss_epoch <= 0):
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
        elif t==4:
            if self.epoch > num_epochs:
                return False
            res=self.predict(self.train_set)
            res[res <= 0.5]=0
            res[res > 0.5]=1
            sum = np.sum(np.abs(res-self.target_train))
            
            print("epoch {} --> somma er {}, loss {}".format(self.epoch, sum, loss_epoch))
            if sum > 0:
                return True
            else:
                return False

        elif t==5:
            pr = np.floor(self.predict(self.train_set) * self.numEx)
            lab = self.target_train *self.numEx
            maxerrepoch = np.max(np.abs(pr-lab))
             
            if (self.maxerr <= maxerrepoch): #and (self.best_loss - loss_epoch <= 0):
                self.real_patience += 1
                if self.real_patience == self.patience_5:
                    return False
                else:
                    return True
            else:
                if self.epoch < num_epochs:
                    if self.maxerr > maxerrepoch:
                        self.maxerr = maxerrepoch
                        self.real_patience = 0
                    return True
                else:
                    return False
                
                # if self.best_loss > loss_epoch:
                #     self.best_loss = loss_epoch
                return True
                
                


    def train(self, stop_function, num_epochs):
        
        self.best_loss = 100.
        last_loss = 99.
        self.last_r2= -100
        self.maxerr = 2^28
        self.best_ep=0    
        self.real_patience = 0

        log.logNN.info("learning rate=" + str(self.lr) + " momentum update=" + str(self.mu) + " minibatch=" + str(self.minibatch))
        while self.stop_fun(stop_function, num_epochs, last_loss):
            self.updateMomentum(self.train_set, self.target_train)
            last_loss = self.loss(self.train_set, self.target_train)

            if self.epoch % 1 == 0 and self.disableLog==False:
                log.logNN.debug("Train - epoch {0} - MAE {1} - MeanErr {2}".format(self.epoch, last_loss, last_loss*self.numEx)) 
            
            #if self.epoch % 10 == 0:
            #    pred = np.floor(self.predict(self.train_set)*self.numEx)           
            #    dif=np.abs(pred-self.target_train*self.numEx)              
            #    m = max(dif)               
            #    print("epoch: {1} -- DELTA MAX: {0} -- MAE: {2}".format(m, self.epoch, last_loss))            
		
            #if self.epoch % 10 == 0:
            #    log.logNN.debug("Train - epoch {} - MAE {} - Delta max {}".format(self.epoch, last_loss, m))
            self.epoch += 1

        log.logNN.info("Train - epoch {0} - MAE {1} - MeanErr {2}".format(self.epoch, last_loss, last_loss*self.numEx))
        #print("epoch: {1} -- DELTA MAX: {0} -- MAE: {2}".format(m, self.epoch, last_loss))
        log.logNN.debug("-------------------------------------------------------------------------------")
        return last_loss


    def getWeight(self):
        return self.layers

    def get_memory_usage(self):
        matrices = sum([len(w) * len(w[0]) + len(b) * len(b[0]) for [w, b] in self.layers])
        # momentum = sum([len(w) * len(w[0]) + len(b) * len(b[0]) for [w, b] in self.v]) if self.epoch > 0 else sum([len(v) for v in self.v])

        floats_bytes = 4
        if type(self.layers[0][0][0][0]) == 'float16':
            floats_bytes = 2.0
        if type(self.layers[0][0][0][0]) == 'float64':
            floats_bytes = 8.0
        
        # tot_weights = matrices + momentum
        tot_weights = matrices
        
        kbytes = np.round(tot_weights * floats_bytes / 1024, 4)
        return kbytes * 100 / self.numEx

