import numpy as np


def ReLU(x, derivate=False):
    if not derivate:
        return x * (x > 0)
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
        
def tanh(z, derivate=False):
    if not derivate:
        ez = np.exp(z)
        enz = np.exp(-z)
        return (ez - enz)/ (ez + enz)
    else:
        return 1 - z**2
        
def LReLU(x, derivate=False):
    alpha=0.05
    if not derivate:
        #pos = x * (x > 0)
        #neg = x * alpha * (x < 0)
        #return pos + neg
        output=np.copy(x)
        output[output<0] *= alpha
        return output
    else:
        #pos = 1. * (x > 0)
        #neg = alpha * (x < 0)
        #return pos + neg
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
        # Return the partial derivation of the activation function
        return np.exp(signal) / (1 + np.exp(signal))
    else:
        # Return the activation signal
        return np.log(1 + np.exp(signal))
