from .Utils import Utils
from .Variable import Variable
from .Activation import ActivationLayer
from .Pooling import Pooling
from .NN import NN

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.append(layer.parameters())
        return params
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, criterion):
        d = criterion.backward()
        idx = len(self.layers) + 1
        if isinstance(self.layers[-1], ActivationLayer):
            d = Utils.bmm(self.layers[-1].backward(), d)
            idx = -1
        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, NN):
                d = layer.backward(d)
            elif isinstance(layer, Pooling):
                d = layer.backward(d)
            elif isinstance(layer, ActivationLayer):
                d = layer.backward() * d
            else:
                d = layer.backward(d)

    def set_params(self, params):
        idx = 0
        for i in range(self.n_layers):
            if hasattr(self.layers[i], 'parameters'):
                if isinstance(params[idx], tuple):
                    self.layers[i].update_params(Variable(params[idx][0]), Variable(params[idx][1]))
                    idx += 1
                elif isinstance(params[idx], Variable):
                    self.layers[i].update_params(params[idx])
                    idx += 1
                    
    def __getitem__(self, idx):
        return self.layers[idx]