import numpy as np
from Layer import *

"""Softmax function"""
def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T


class Loss(Layer):
    """Base class for a loss layer"""
    def __call__(self,prev,input_layer):
        self.input_layer = input_layer
        return super(Loss,self).__call__(prev)

class MSELoss(Loss):
    """ Mean squared error loss class"""
    def __init__(self):
        super(MSELoss,self).__init__("MSELoss")
    def fw(self):
        X = self.prev[0].fw_result
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        sample_losses = np.square(self.input_layer.target - X)
        self.fw_result = np.repeat(np.reshape(np.mean(sample_losses,axis=0),((1,)+sample_losses.shape[1:])),self.batch_size,axis=0)
    def bw(self):
        sample_ds = self.prev[0].fw_result - self.input_layer.target
        self.bw_result = np.repeat(np.expand_dims(np.mean(sample_ds,axis=0),axis=0),self.batch_size,axis=0)
        
class CELoss(Loss):
    """Cross Entropy loss layer"""
    def __init__(self):
        super(CELoss,self).__init__("CELoss")
    def fw(self):
        y_pred = self.prev[0].fw_result
        n_pred = y_pred.shape[0]
        #print(n_pred)
        y_train = self.input_layer.target.astype(int)
        self.prob = softmax(y_pred)
        self.fw_result = -np.log(self.prob[range(n_pred), y_train])

        
    def bw(self):
        y_pred = self.prev[0].fw_result
        n_pred = y_pred.shape[0]
        y_train = self.input_layer.target.astype(int)
        self.bw_result = self.prob.copy()
        self.bw_result[range(n_pred), y_train] -= 1.
        self.bw_result /= n_pred
        #average across batch and repeat avg this should probably be elsewhere, not inside layer itself, maybe on solver...
        #self.bw_result = np.repeat(np.expand_dims(self.bw_result.mean(axis=0),axis=0),n_pred,axis=0)
