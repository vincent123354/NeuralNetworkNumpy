from .Variable import Variable

import numpy as np

class WeightInitializer:
    def __init__(self, method='he'):
        self.method = method
    
    def __call__(self, in_features, out_features, k=None):
        if self.method == 'normal':
            return self.normal_init(in_features, out_features, k)
        elif self.method == 'xavier_normal':
            return self.xavier_init(in_features, out_features, normal=True, k=k)
        elif self.method == 'xavier_standard':
            return self.xavier_init(in_features, out_features, k)
        elif self.method == 'he':
            return self.he_init(in_features, out_features, k)
    
    def normal_init(self, in_features, out_features, k=None):
        if k == None:
            w = np.random.randn(out_features, in_features)
            b = np.random.randn(out_features, 1)
            return Variable(w), Variable(b)
        else:
            w = np.random.randn(in_features, out_features, k, k)
            b = np.random.randn(1, out_features, 1, 1)
            return Variable(w), Variable(b)
    
    def xavier_init(self, in_features, out_features, normal=False, k=None):
        if k == None:
            if normal:
                limit = np.sqrt(2 / (in_features + out_features))
                w = np.random.normal(0, 1, size=(out_features, in_features))  * limit
            else:
                limit = np.sqrt(6 / (in_features + out_features))
                w = np.random.uniform(-limit, limit, size=(out_features, in_features)) 
            b = np.random.randn(out_features, 1)
            return Variable(w), Variable(b)
        else:
            if normal:
                limit = np.sqrt(2 / (in_features + out_features))
                w = np.random.normal(0, 1, size=(out_features, in_features, k, k))  * limit
            else:
                limit = np.sqrt(6 / (in_features + out_features))
                w = np.random.uniform(-limit, limit, size=(out_features, in_features, k, k)) 
            b = np.random.randn(1, out_features, 1, 1)
            return Variable(w), Variable(b)
    
    def he_init(self, in_features, out_features, k=None):
        if k == None:
            limit = np.sqrt(2 / in_features)
            w = np.random.normal(0, 1, size=(out_features, in_features)) * limit
            b = np.zeros((out_features, 1))
            return Variable(w), Variable(b)
        else:
            limit = np.sqrt(2 / in_features)
            w = np.random.normal(0, 1, size=(out_features, in_features, k, k)) * limit
            b = np.zeros((1, out_features, 1, 1))
            return Variable(w), Variable(b)