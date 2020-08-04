from abc import ABC, abstractmethod
from .WeightInitializer import WeightInitializer
from .Utils import Utils

import copy
import numpy as np

class NN(ABC):
    def __init__(self):
        super().__init__()
        
    @property
    def weight(self):
        return self._w.values
    
    @property
    def bias(self):
        return self._b.values
    
    def parameters(self):
        return self._w, self._b

    def init_params(self, in_features, out_features, k=None):
        self._w, self._b = self.initializer(in_features, out_features, k)   

    def update_params(self, params):
        self._w.values = copy.deepcopy(params[0].values)
        self._b.values = copy.deepcopy(params[1].values)
        
    def __repr__(self):
        return self._w.__repr__() + self._b.__repr__()
    
    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def backward(self):
        pass

class Conv2d(NN):
    def __init__(self, in_features, out_features, k, p=0, s=1, initializer=WeightInitializer()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.p = p
        self.s = s
        
        self.initializer = initializer
        self.init_params(in_features, out_features, k)
        self.cache = {}
    
    
    def __call__(self, x):
        m, channel, height, width = x.shape
        x_pad = Utils.pad(x, self.p)
        submat_shape = tuple((m, int(np.floor((height + 2*self.p - self.k) / self.s + 1)), int(np.floor((width + 2*self.p - self.k) / self.s + 1)))) + (self.in_features, self.k, self.k)
        itemsize = x_pad.itemsize
        stride = (x_pad.strides[0], self.s*itemsize*(width+2*self.p), self.s*itemsize) + x_pad.strides[1:]
        sub_matric = np.lib.stride_tricks.as_strided(x_pad, submat_shape, stride)
        output = np.einsum('lijk,mnoijk->mlno', self.weight, sub_matric)
        
        self.cache['x'] = x
        self.cache['submat'] = sub_matric
        
        return output + self.bias
    
    def backward(self, dz):
        dw = np.einsum('moni,mnijkl->ojkl', dz, self.cache['submat']) / (dz.shape[0] * self.cache['x'].shape[2] * self.cache['x'].shape[3])
        db = dz.sum(axis=2,keepdims=True).sum(axis=3, keepdims=True).sum(axis=0, keepdims=True) / (dz.shape[0] * self.cache['x'].shape[2] * self.cache['x'].shape[3])
        
        temp = np.multiply(np.expand_dims(dz, [4,5,6]), np.repeat(np.repeat(self.weight[:, np.newaxis, np.newaxis, :, :, :], dz.shape[2], axis=1), dz.shape[3], axis=2)).sum(axis=1)
        da_prev = Utils.pad(np.zeros(self.cache['x'].shape), self.p)
        oas = np.lib.stride_tricks.as_strided(da_prev, self.cache['submat'].shape, self.cache['submat'].strides)
        np.add.at(oas, ([i for i in range(self.cache['x'].shape[0])]), temp)
        da_prev = Utils.unpad(da_prev, self.p)

        average = dz.shape[1] * dz.shape[2] * dz.shape[3]

        da_prev *= average 

        self._w.grad = dw
        self._b.grad = db

        return da_prev

class Linear(NN):
    def __init__(self, in_features, out_features, initializer=WeightInitializer()):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        
        self.initializer = initializer
        
        self.init_params(in_features, out_features)
        
        self.cache = None
   
    def __call__(self, x):
        z = x.dot(self.weight.T) + self.bias.T
        self.cache = {'x':x}
        return z

    def backward(self, dz):
        dw = dz.T.dot(self.cache['x']) / self.cache['x'].shape[0] / self.in_features
        db = np.sum(dz, axis=0, keepdims=True).T / self.cache['x'].shape[0] / self.in_features
        da = dz.dot(self.weight)

        self._w.grad = dw
        self._b.grad = db
        return da