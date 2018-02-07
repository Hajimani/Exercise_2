from Layer import *
import numpy as np
class MaxP(Layer):
    def __init__(self,stride,filter_shape,padding=(0,0)):
        self.s = stride
        self.f_shape = filter_shape
        self.p = padding
        self.s_h = None
        self.s_w = None
        super(MaxP,self).__init__("MaxP")
        
    def fw(self):
        X = self.prev[0].fw_result
        batch_size = X.shape[0]
        channels = X.shape[1]
        self.X_reshaped = X.reshape((batch_size*channels,1,X.shape[-2],X.shape[-1]))
        if self.s_h is None: #lazily instantiate necessary things
            self.s_h = int(conv_size(X.shape[-2],self.f_shape[0],self.s[0],self.p[0]))
            self.s_w = int(conv_size(X.shape[-1],self.f_shape[1],self.s[1],self.p[1]))
            self.batch_size = batch_size
        self.X_col = im2col(self.X_reshaped,self.f_shape,self.s,self.s_h*self.s_w)  
        self.fw_result = self.X_col[np.argmax(self.X_col,axis=0),range(np.argmax(self.X_col,axis=0).size)]
        self.fw_result = self.fw_result.reshape(self.s_h,self.s_w,self.batch_size,channels).transpose(2,3,0,1)
        
    def bw(self):
        X = self.prev[0].fw_result
        
        self.bw_result = np.zeros(self.X_col.shape)
        dout = self.nxt[0].bw_result.transpose(2,3,0,1).reshape(-1)
        self.bw_result[np.argmax(self.X_col,axis=0),range(np.argmax(self.X_col,axis=0).size)] = dout
        self.bw_result = col2im(self.bw_result,self.f_shape,self.s,self.X_reshaped.shape,self.s_h,self.s_w).reshape(X.shape)

        
        