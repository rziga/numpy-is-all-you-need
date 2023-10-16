import numpy as np
from ._base import Module


class ReLU(Module):

    def __init__(self):
        super().__init__()
        self.M = None

    def forward(self, input):
        self.M = input > 0
        return input * self.M
    
    def backward(self, next):
        return next * self.M


class Sigmoid(Module):

    def __init__(self):
        super().__init__()
        self.O = None
    
    def forward(self, input):
        self.O = 1 / (1 + np.exp(-input))
        return self.O
    
    def backward(self, next):
        return (self.O * (1 - self.O)) * next

"""
class Softmax(Module):
    
    def __init__(self):
        self.O = None

    def forward(self, input):
        expx = np.exp(input - input.max(axis=-1, keepdims=True))
        self.O = expx / expx.sum(axis=-1, keepdims=True)
        return self.O
    
    def backward(self, next):
        #flatten
        #diag matrix
        #add the offdiag elements
        #mask with kronecker product of ones matrix and diag matrix
        O_flat = np.atleast_2d(self.O.ravel())
        D = -O_flat * O_flat.T
        D += np.diag(O_flat.squeeze(-2))
        D *= np.kron(np.eye(self.O.shape[-2]), np.ones([self.O.shape[-1], self.O.shape[-1]]))

        d = next.reshape(*next.shape[:-2], -1)
        d = d @ D
        return d.reshape(next.shape)
"""
        
class Softmax(Module):
    
    def __init__(self):
        super().__init__()
        self.O = None

    def forward(self, input):
        expx = np.exp(input - input.max(axis=-1, keepdims=True))
        self.O = expx / expx.sum(axis=-1, keepdims=True)
        return self.O
    
    def backward(self, next):
        O = self.O[..., np.newaxis] # [..., examples, features, 1]
        O_T = O.swapaxes(-1, -2) # [..., examples, 1, features]
        D = -O * O_T # [..., examples, features, features]
        D[..., range(D.shape[-2]), range(D.shape[-1])] += self.O # add over diagonal
        return (D @ next[..., np.newaxis]).squeeze(-1)
    