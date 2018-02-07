from itertools import cycle
import numpy as np
from Layer import *
class Input(Layer):
    
    __input_shape = None
    def __init__(self,name,input_shape,batch_size):
        super(Input,self).__init__(name)
        self.__input_shape = input_shape
        self.batch_size = batch_size

    def __call__(self,X,Y):
        self.__X = X
        self.__Y = Y
        self.__X_pool = cycle(X) #ciclical generator
        self.__Y_pool = cycle(Y)
        
        return self
    
    def fw(self):
        if self.fw_result is None:
            self.fw_result = np.empty((self.batch_size,)+self.__X.shape[1:])
            self.target = np.empty((self.batch_size,)+self.__Y.shape[1:])

        for i in range(self.batch_size):
            self.fw_result[i] = next(self.__X_pool)
            self.target[i] = next(self.__Y_pool)
