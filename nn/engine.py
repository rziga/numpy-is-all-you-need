import numpy as np


def check_grad(module, input, h=1e-6):
    #TODO: gradient checking
    pass

def search_params(item):
    # deep search , returns flattened list of parameters
    if isinstance(item, Parameter):
        return [item]
    elif isinstance(item, list):
        iterator = item
    elif isinstance(item, Module):
        iterator = vars(item).values()
    else:
        return []
    
    return [
        p for item in iterator 
        for p in search_params(item)
        ]

def search_submodules(module):
    # shallow searc - first level submodules only
    submodules = []
    for var in vars(module).values():
        if isinstance(var, Module):
            submodules.append(var)
        elif isinstance(var, list):
            submodules.extend([el for el in var if isinstance(el, Module)])
    return submodules


class Parameter():

    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(self.data)
        self.n_params = np.prod(self.data.shape)


class Module():

    def __init__(self):
        self._eval = False
        self._persistent = False

    def forward(self, input):
        raise NotImplementedError

    def backward(self, next):
        # define jacobian * next
        # next is d loss / output
        # update parameters
        # return d output / d input * d loss / d output = d loss / d input
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)

    def parameters(self):
        return search_params(self)
    
    def n_params(self):
        return sum([p.n_params for p in self.parameters()])
    
    def submodules(self):
        return search_submodules(self)
    
    @property
    def mode(self):
        return "eval" if self._eval else "train"
    
    def eval(self, verbose=False):
        self._eval = True
        if verbose: 
            print(f"{self} set to eval")
        for sm in self.submodules():
            sm.eval(verbose)
    
    def train(self, verbose=False):
        self._eval = False
        if verbose:
            print(f"{self} set to train")
        for sm in self.submodules():
            sm.train(verbose)

    def _persist(self, status=True):
        self._persistent = True
        for sm in self.submodules():
            sm.persist(status)
            