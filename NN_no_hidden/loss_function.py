import numpy as np
from scipy.special import logsumexp

def delta_abs(X):
    output = np.copy(X)
    output[X>0] = 1
    output[X<0] = -1
    return output

def MSE(y, target, derivate=False):
    if not derivate:
        return np.mean(np.square(y-target))
    else:
        size_minibatch = len(y)
        return (y - target) * 2/size_minibatch
        
def MAE(y, target, derivate=False):
    if not derivate:
        return np.mean(np.abs(y-target))
    else:
        size_minibatch = len(y)
        return delta_abs(y-t[indexLow:indexHigh]) * 1/size_minibatch
        
def LSE(y, target, derivate=False):
    if not derivate:
        return logsumexp(np.square(y-target))
    else:
        dif = np.square(y-target)
        return (2*(y-target)*np.exp(dif))/sum(np.exp(dif)) 
        

