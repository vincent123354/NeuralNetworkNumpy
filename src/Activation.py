from abc import ABC, abstractmethod
from .Utils import Utils

import copy
import numpy as np

class ActivationLayer(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def backward(self, d):
        pass
    
    def __str__(self):
        return "Activation Layer"


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__()
        self.x = None
        
    def __call__(self,x):
        self.x = x
        return self.sigmoid(x)
    
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def backward(self):
        s = self.sigmoid(self.x)
        return np.expand_dims(s * (1-s), -1)

class Softmax(ActivationLayer):
    def __init__(self):
        super().__init__()
        self.x = None
        
    def __call__(self, x):
        self.x = x
        return self.softmax(x)
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def backward(self):
        a = self.softmax(self.x)
        return np.transpose(Utils.bmm(np.ones(a.shape+tuple((1,))), a[:, np.newaxis, :]), [0,2,1]) * (np.eye(a.shape[1]) - Utils.bmm(np.ones(a.shape+tuple((1,))), a[:, np.newaxis, :]))
#         return Utils.bmm(np.ones(a.shape+tuple((1,))), a[:, np.newaxis, :]) * (np.eye(a.shape[1]) - np.transpose(Utils.bmm(np.ones(a.shape+tuple((1,))), a[:, np.newaxis, :]), [0,2,1]))

class Relu(ActivationLayer):
    def __init__(self):
        super().__init__()
        self.x = None
    
    def __call__(self, x):
        self.x = x
        return np.where(x>=0, x, 0)
    
    def backward(self):
        return (self.x > 0).astype(int)
#         return np.where(self.x>0, np.where(self.x<=0, self.x, 1), 0)

class Tanh:
    def __init__(self):
        self.x = None
        
    def __call__(self, x):
        self.x = x
        return np.tanh(x)
    
    def backward(self):
        return (1-np.power(np.tanh(self.x), 2))