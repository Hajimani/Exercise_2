from Layer import *
import numpy as np
from Activations import *

class FC(Layer):
    
    w = None
    b = None
    
    def __init__(self,activation,n_out,name="fc"):
        self.__n_out = n_out
        self.b = None
        self.w = None
        #self.output_shape = None
        self.activation = Linear_activation("linear_activation") if activation != "relu" else RELU_activation("RELU_activation")
        super(FC,self).__init__(name)
    
    def __call__(self,prev_layer):
        super(FC,self).__call__(prev_layer)
        return self.activation(self)


    def fw(self):
        
        if self.b is None:
            self.b = np.zeros(self.__n_out)
            self.w = np.random.rand(np.prod(self.prev[0].fw_result.shape[1:]),self.__n_out)
            #self.output_shape = (self.prev[0].fw_result.shape[0],self.__n_out)
            self.batch_size = self.prev[0].fw_result.shape[0]

        reshaped_x = self.prev[0].fw_result.reshape(self.batch_size,-1) # dont care about nr of dimensions of input
        self.fw_result = np.dot(reshaped_x,self.w) + self.b.T 
        self.activation.fw() # apply activation
        self.fw_result = self.activation.fw_result # replace
    
    def bw(self):
        
        d_next = self.nxt[0].bw_result # we get bw computation directly from Activation sub layer

        reshaped_x = self.prev[0].fw_result.reshape(self.batch_size,-1)

        self.d_w = reshaped_x.T.dot(d_next)
        self.d_b = np.sum(d_next,axis=0)
        self.bw_result = np.dot(d_next,self.w.T).reshape(self.prev[0].fw_result.shape)
        # self.d_in = np.dot(d_next,self.w.T)
    def update(self,lr):
        self.w -= lr*self.d_w
        self.b -= lr*self.d_b