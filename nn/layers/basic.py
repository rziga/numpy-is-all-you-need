import numpy as np
from ..engine import Parameter, Module


class MatMul(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.first, self.second = input
        return np.matmul(*input)
    
    def backward(self, next):
        return (
            next @ self.second.swapaxes(-1, -2), 
            self.first.swapaxes(-1, -2) @ next
            )


class Linear(Module):

    def __init__(self, in_chan, out_chan, bias=True):
        super().__init__()
        self.W = Parameter(2*np.random.rand(in_chan, out_chan)-1)
        self.b = Parameter(2*np.random.rand(1, out_chan)-1) if bias else None
        self.mm = MatMul()

    def forward(self, input):
        return self.mm([input, self.W.data]) +\
            (self.b.data if self.b is not None else 0)
    
    def backward(self, next):
        d_X, d_W = self.mm.backward(next)
        self.W.grad += d_W.reshape(-1, *self.W.grad.shape).sum(axis=0) # accumulate over batches
        if self.b:
            self.b.grad += next.reshape(-1, *self.b.grad.shape).sum(axis=0)
        return d_X
    

class LayerNorm(Module):

    def __init__(self):
        super().__init__()
        self.sigma = None
        self.X_shift = None

    def forward(self, input):
        mu = input.mean(axis=-1, keepdims=True)
        self.sigma = input.std(axis=-1, keepdims=True)
        self.X_shift = input - mu
        return self.X_shift / self.sigma
    
    def backward(self, next):
        # neural-threads medium post
        # simplified form and changed for batch processing
        N = self.X_shift.shape[-1]
        sigma = self.sigma[..., np.newaxis]
        X_s = self.X_shift[..., np.newaxis]
        X_s_T = X_s.swapaxes(-1, -2)
        D = -(X_s * X_s_T + sigma**2) / sigma**3 / N
        D[..., range(D.shape[-2]), range(D.shape[-1])] += 1 / sigma.squeeze(-1)
        return (D @ next[..., np.newaxis]).squeeze(-1)
    

class Embedding(Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeds = Parameter(np.random.rand(vocab_size, embedding_dim))
        self.embedding_dim = embedding_dim
        self.selector = None

    def forward(self, input):
        self.selector = input.ravel()
        return self.embeds.data[self.selector].reshape(*input.shape, self.embedding_dim)
    
    def backward(self, next):
        d = next.reshape(-1, next.shape[-1])
        np.add.at(self.embeds.grad, self.selector, d)


class Dropout(Module):

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, input):
        if self.mode == "eval":
            return input
        if not self._persistent or self.mask is None:
            self.mask = (np.random.rand(*input.shape) >= self.p).astype(input.dtype)
        return input * self.mask / (1-self.p)
    
    def backward(self, next):
        if self.mode == "eval":
            next
        return next * self.mask / (1-self.p)
    