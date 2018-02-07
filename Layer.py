from skimage.util import view_as_windows
import numpy as np

# in order to get number of filter positions for convolution
conv_size = lambda width,filter_size,stride,padding: (width - filter_size + 2 * padding)/stride + 1 
valid_config = lambda widht,filter_size,stride,padding: (width - filter + 2 * padding)%stride == 0

class Layer:
    """A layer base class"""
   
    fw_result,bw_result = None,None  

    def __init__(self,name):
        self.name = name
        self.prev,self.nxt = [],[]
        self.batch_size = None
    """ Forward pass method """
    def fw(self):
        return None
    """ Backwards pass method """
    def bw(self):
        return None

    """ Method used to hook up consecutive layers """
    def __call__(self,prev_layer):
        prev_layer.nxt.append(self)
        self.prev.append(prev_layer)
        return self
    def update(self,lr):
        return
"""
Transforms input in a format suitable to perform the convolution as a single matrix multiplication operation
"""
def im2col(X,f_shape,stride,filter_positions):
    X_col = view_as_windows(X,X.shape[:2]+f_shape,X.shape[:2]+stride)
    #put in right format
    return X_col.transpose((0, 1, 5, 6, 7, 2, 3, 4)).reshape(-1,X.shape[0]*filter_positions)

"""
Reverts operation done by im2col but sums the values for the various positions of the filter if stride < filter_size
"""
## maybe this could have been done with some fancy indexing, instead of looping through the array...
def col2im(X_col,f_shape,stride,X_shape,s_h,s_w):
    new_X = np.zeros(X_shape)
    batches,channels = X_shape[0:2]
    for i in range(0,channels * np.prod(f_shape),np.prod(f_shape)):
        channel_pos = i // np.prod(f_shape)
        for j in range(s_h*s_w*batches):
            batch_pos = j % batches
            reshaped_in = X_col[i:i+np.prod(f_shape),j].reshape(f_shape)
            x_pos = (j //batches) % s_w
            y_pos = (j // (batches * s_w))
            x_pos *= stride[0]
            y_pos *= stride[1]
            new_X[batch_pos,channel_pos,y_pos:y_pos+f_shape[1],x_pos:x_pos+f_shape[0]] += reshaped_in
    return new_X