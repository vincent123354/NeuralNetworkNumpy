# https://stackoverflow.com/questions/40583131/python-deepcopy-with-custom-getattr-and-setattr
class Variable(object):
    
    def __init__(self, x):
        super().__setattr__('__dict__', {})
        self.values = x
        self.grad = None
        
    def __getattr__(self, key):
        try:
            return super().__getattribute__('__dict__')[key]
        except KeyError:
            raise AttributeError(key)
    
    def __setattr__(self, key, value):
        if key in self.__dict__:
            super().__setattr__(key, value)
        else:
            self.__dict__[key] = value
    
    def __repr__(self):
        return str(self.values)
    