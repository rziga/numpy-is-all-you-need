import numpy as np
from ._base import Module
from .basic import Linear, MatMul
from .activations import Softmax


class AttentionHead(Module):

    def __init__(self, d_model, d_hidden, mask_fcn=None):
        super().__init__()
        self.lin_Q = Linear(d_model, d_hidden, bias=False)
        self.lin_K = Linear(d_model, d_hidden, bias=False)
        self.lin_V = Linear(d_model, d_hidden, bias=False)
        self.mm1 = MatMul()
        self.mm2 = MatMul()
        self.sm = Softmax()
        self.mask_fcn = mask_fcn
        self.mask = None
        self.scale = d_hidden ** 0.5

    def forward(self, input):
        Q, K, V = self.lin_Q(input[0]), self.lin_K(input[1]), self.lin_V(input[2])
        x = self.mm1([Q, K.swapaxes(-1, -2)]) / self.scale
        if self.mask_fcn is not None:
            self.mask = self.mask_fcn(x.shape)
            x[self.mask == 0] = -1e-11
        x = self.sm(x)
        return self.mm2([x, V])
    
    def backward(self, next):
        d, d_V = self.mm2.backward(next)
        d = self.sm.backward(d) / self.scale
        if self.mask is not None:
            d = self.mask * d
        d_Q, d_K_T = self.mm1.backward(d)
        return (
            self.lin_Q.backward(d_Q), 
            self.lin_K.backward(d_K_T.swapaxes(-1, -2)), 
            self.lin_V.backward(d_V)
        )
    

class MultiHeadAttention(Module):

    def __init__(self, n_heads, d_model, d_hidden, mask_fcn=None):
        super().__init__()
        self.heads = [AttentionHead(d_model, d_hidden, mask_fcn) for _ in range(n_heads)]
        self.lin = Linear(n_heads*d_hidden, d_model, bias=False)

    def forward(self, input):
        outs = [head(input) for head in self.heads]
        x = np.concatenate(outs, axis=-1)
        return self.lin(x)
    
    def backward(self, next):
        d = self.lin.backward(next)
        ds = np.split(d, len(self.heads), axis=-1)
        ds = [head.backward(d) for head, d in zip(self.heads, ds)]
        d = np.sum(ds, axis=0)
        return d
    

class SelfAttention(MultiHeadAttention):

    def forward(self, input):
        return super().forward([input, input, input])

    def backward(self, next):
        return np.sum(super().backward(next), axis=0)