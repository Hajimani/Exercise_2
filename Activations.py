import numpy as np
from Layer import *
class RELU_activation(Layer):
    # self.prev = [c1]
    def fw(self):
        self.fw_result = self.prev[0].fw_result * (self.prev[0].fw_result > 0) # apply a mask
    def bw(self):
        self.bw_result = self.nxt[0].bw_result * (self.fw_result >= 0)

class Linear_activation(Layer):
    def fw(self):
        self.fw_result = self.prev[0].fw_result
    def bw(self):
        self.bw_result = self.nxt[0].bw_result