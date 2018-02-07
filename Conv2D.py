from Layer import *
from Loss import softmax
from Activations import *
import numpy as np
class Conv2D(Layer):
    w = None
    b = None
    
    def __init__(self,stride,filter_shape,n_filters,padding=(0,0),activation="linear_activation"):
        self.s = stride
        self.f_shape = filter_shape
        self.p = padding
        self.n = n_filters
        self.w = None
        self.activation = Linear_activation("linear_activation") if activation != "relu" else RELU_activation("RELU_activation")
        super(Conv2D,self).__init__("Conv2D")

    def __call__(self,prev_layer):
        super(Conv2D,self).__call__(prev_layer)
        return self.activation(self)

    def fw(self):
        X = self.prev[0].fw_result
        channels = X.shape[1]
        if self.w is None: #lazily instantiate some vars
            self.w = np.random.randn(self.n,channels,self.f_shape[0],self.f_shape[1])*np.sqrt(2.0/self.n)
            # since random function is selecteing from a uniform distribution, we are going to pass them to softmax
            #self.w =softmax(self.w)

            self.b = np.zeros((self.n)).reshape(-1,1)
            self.s_h = int(conv_size(X.shape[-2],self.f_shape[0],self.s[0],self.p[0]))
            self.s_w = int(conv_size(X.shape[-1],self.f_shape[1],self.s[1],self.p[1]))
            self.batch_size = X.shape[0]
    
        # change shape to make Conv2D a matrix mult operation
        self.X_col = im2col(X,self.f_shape,self.s,self.s_h*self.s_w)
        w_col = self.w.reshape(self.n, -1)
        
        self.fw_result = (np.dot(w_col,self.X_col) + self.b)

        # reshape to output format
        self.fw_result = self.fw_result.reshape(self.n, self.s_h, self.s_w, self.batch_size).transpose(3, 0, 1, 2)
        self.activation.fw() # apply activation
        self.fw_result = self.activation.fw_result # replace
        
    def bw(self):
        
        dout = self.nxt[0].bw_result

        self.d_b = np.sum(dout, axis=(0, 2, 3)).reshape(self.n)
        
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.n, -1)
        
        self.d_w = np.dot(dout_reshaped,self.X_col.T).reshape(self.w.shape) 

        dx_col = np.dot(self.w.reshape(self.n, -1).T,dout_reshaped)
        
        self.bw_result = col2im(dx_col,self.f_shape,self.s,self.prev[0].fw_result.shape,self.s_h,self.s_w)
        
    def update(self,lr):
        self.w -= lr*self.d_w
        self.b -= lr*self.d_b.reshape(self.b.shape)
