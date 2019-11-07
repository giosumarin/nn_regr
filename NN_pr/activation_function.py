import numpy as np


def ReLU(x, derivate=False):
    if not derivate:
        return np.maximum(0, x)
    else:
        return 1. * (x > 0)

def sigmoid(x, derivate=False):
    if not derivate:
        return 1 / (1 + np.exp(-x))
    else:
        return x * (1-x)
        
def linear(x, derivate=False):
    if not derivate:
        return x
    else:
        return np.ones(x.shape)
        
def tanh(x, derivate=False):
    x = np.tanh(x)
    if not derivate:
        return x
    else:
        return 1 - np.power(x,2)
        
def LReLU(x, derivate=False):
    alpha=0.05
    if not derivate:
        output=np.copy(x)
        output[output<0] *= alpha
        return output
    else:
        output = np.clip(x>0,alpha,1.)
        return output


def softmax_function( signal, derivative=False ):
    e_x = np.exp( signal - np.max(signal, axis=1, keepdims = True) )
    signal = e_x / np.sum( e_x, axis = 1, keepdims = True )
    
    if derivative:
        return np.ones( signal.shape )
    else:
        return signal
        
def softplus_function( signal, derivative=False ):
    if derivative:
        return np.exp(signal) / (1 + np.exp(signal))
    else:
        return np.log(1 + np.exp(signal))
