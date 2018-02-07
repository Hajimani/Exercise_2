from collections import deque
from Layer import *
from FC import *
from Input import *
from Loss import *
from Conv2D import *
from Pooling import *
import numpy as np
class Model:
    #output = loss
    
    def __init__(self,output,X,Y,lr=0.001):
        print("Initing")
        self.node_list = [output]
        self.lr = lr
        index = 0
        while(True):
            node = self.node_list[index]
            #node.build(batch_size)
            for n in node.prev:
                self.node_list.append(n)
            index += 1
            if len(self.node_list) == index:
                self.node_list[-1](X,Y)# set input
                break    
        self.node_list = [x for x in reversed(self.node_list)]
        
    def forward_pass(self):
       [n.fw() for n in self.node_list]
    def backward_pass(self):
        [n.bw() for n in reversed(self.node_list)]
    def update(self):
        [n.update(self.lr) for n in self.node_list]