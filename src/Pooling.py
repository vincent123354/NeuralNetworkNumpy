from abc import ABC, abstractmethod
from .Utils import Utils

import copy
import numpy as np

class Pooling(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def backward(self):
        pass
    
    def __str__(self):
        return 'Pooling Layer'

class MaxPool(Pooling):
    def __init__(self, k, p=0, s=None):
        super().__init__()
        self.cache = {}
        self.k = k
        self.p = p
        if s == None:
            self.s = k
        else:
            self.s = s
        
    def __call__(self, x):
        m, c, h, w = x.shape
        x_pad = Utils.pad(x, self.p)

        output_width = int(np.floor((w + self.p*2 - self.k) / self.s + 1))
        output_height = int(np.floor((h + self.p*2 - self.k) / self.s + 1))

        output_shape = (m, c, output_width, output_height) + (self.k, self.k)
        itemsize = x_pad.itemsize
        stride = x_pad.strides[0], itemsize*(w+2*self.p)*(h+2*self.p), itemsize*self.s*(w+2*self.p), itemsize*self.s, itemsize*(w+2*self.p), itemsize
        subarr = np.lib.stride_tricks.as_strided(x_pad, output_shape, stride)
        
        output = np.max(np.max(subarr, axis=4), axis=4)
        
        self.cache['x'] = x
        self.cache['submat'] = subarr
        
        return output
    
    def backward(self, d):
        mask = (self.cache['submat'].max(axis=4, keepdims=True).max(axis=5, keepdims=True) == self.cache['submat']).astype(int)
        
        d = np.repeat(np.repeat(d[:,:,:,:,np.newaxis,np.newaxis], self.k, axis=4), self.k, axis=5)
        masked_d = d * mask
        
        da_prev = Utils.pad(np.zeros(self.cache['x'].shape), self.p)
        oas = np.lib.stride_tricks.as_strided(da_prev, self.cache['submat'].shape, self.cache['submat'].strides)
        np.add.at(oas, ([i for i in range(self.cache['x'].shape[0])]), masked_d)
        da_prev = Utils.unpad(da_prev, self.p)
        
        return da_prev

class AvgPool(Pooling):
    def __init__(self, k, p=0, s=None):
        super().__init__()
        self.cache = {}
        self.k = k
        self.p = p
        if s == None:
            self.s = k
        else:
            self.s = s
        
    def __call__(self, x):
        m, c, h, w = x.shape
        x_pad = Utils.pad(x, self.p)

        output_width = int(np.floor((w + self.p*2 - self.k) / self.s + 1))
        output_height = int(np.floor((h + self.p*2 - self.k) / self.s + 1))

        output_shape = (m, c, output_width, output_height) + (self.k, self.k)
        itemsize = x_pad.itemsize
        stride = x_pad.strides[0], itemsize*(w+2*self.p)*(h+2*self.p), itemsize*self.s*(w+2*self.p), itemsize*self.s, itemsize*(w+2*self.p), itemsize

        subarr = np.lib.stride_tricks.as_strided(x_pad, output_shape, stride)
        
        output = np.mean(np.mean(subarr, axis=4), axis=4)
        
        self.cache['x'] = x
        self.cache['submat'] = subarr
        
        return output
    
    def backward(self, d):
        d /= (self.k * self.k)
        
        temp = np.repeat(np.repeat(d[:,:,:,:,np.newaxis, np.newaxis], self.k, axis=4), self.k, axis=5)
        da_prev = Utils.pad(np.zeros(self.cache['x'].shape), self.p)
        oas = np.lib.stride_tricks.as_strided(da_prev, self.cache['submat'].shape, self.cache['submat'].strides)
        np.add.at(oas, ([i for i in range(self.cache['x'].shape[0])]), temp)
        da_prev = Utils.unpad(da_prev, self.p)
        
        return da_prev