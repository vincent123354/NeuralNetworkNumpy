import numpy as np

class Utils:
    @staticmethod
    def pad(x, p):
        return np.pad(x, ((0,0),(0,0),(p,p),(p,p)))
    
    @staticmethod
    def unpad(x, p):
        m,c,h,w = x.shape
        return x[:,:,p:h-p,p:w-p]
    
    @staticmethod
    def bmm(x, y):
        result = None
        if len(x.shape) == 2:
            result = np.einsum('ij,ij->ij', x, y)
        elif len(x.shape) == 3:
            result = np.einsum('ijk,ikl->ijl', x, y)
        if result.shape[-1] == 1:
            result = result.squeeze(-1)
        return result
    
    @staticmethod
    def onehot(x):
        n_values = np.max(x) + 1
        return np.eye(n_values)[x].squeeze()

class Flatten:
    def __init__(self):
        self.x = None
        
    def __call__(self, x):
        self.x = x
        return x.reshape(x.shape[0], -1)
    
    def backward(self, d):
        return d.reshape(self.x.shape)