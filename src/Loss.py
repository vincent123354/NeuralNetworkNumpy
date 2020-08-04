import numpy as np

class BinaryCrossEntropyLoss:
    def __init__(self):
        self.a = None
        self.y = None
    
    def __call__(self, a, y, eps=1e-7):
        self.a = a
        self.y = y
        return -1 * np.mean(np.multiply(y, np.log(a+eps)) + np.multiply((1-y), np.log(1-a+eps)))
    
    def backward(self, eps=1e-7):
        return np.expand_dims(-(self.y/(self.a+eps) - (1-self.y)/(1-self.a+eps)), -1)

# http://saitcelebi.com/tut/output/part2.html#numerical_stability_of_the_loss_function
class CrossEntropyLoss:
    def __init__(self):
        self.a = None
        self.y = None
        
    def __call__(self, a, y, eps=1e-7):
        self.a = a
        self.y = y
        K = - np.max(a, axis=1, keepdims=True)
        return - (y * (a + K - np.log(np.exp(a + K).sum(axis=1, keepdims=True)))).sum(axis=1).mean()
#         return np.mean(y * np.log(a+eps))
    
    def backward(self, eps=1e-7):
        return np.expand_dims(- self.y / (self.a+eps), -1)