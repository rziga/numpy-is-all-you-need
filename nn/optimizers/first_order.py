import numpy as np
from ._base import Optimizer


class GD(Optimizer):

    def __init__(self, parameters, learning_rate):
        super().__init__(parameters)
        self.lr = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad


class Adam(Optimizer):

    def __init__(self, parameters, learning_rate, beta_1, beta_2, eps=1e-8):
        super().__init__(parameters)
        self.lr = learning_rate
        self.b1 = beta_1
        self.b2 = beta_2
        self.eps = eps

        self.mom1 = [np.zeros_like(p.grad) for p in self.parameters]
        self.mom2 = [np.zeros_like(p.grad) for p in self.parameters]
        self.t = 0

    def step(self):
        for p, m1, m2 in zip(self.parameters, self.mom1, self.mom2):
            self._update_parameter(p, m1, m2)
    
    def _update_parameter(self, parameter, moment_1, moment_2):
        t = self.t + 1
        g = parameter.grad
        m = self.b1 * moment_1 + (1 - self.b1) * g
        v = self.b2 * moment_2 + (1 - self.b1) * g**2
        m_ = m / (1 - self.b1**t)
        v_ = v / (1 - self.b2**t)
        # update
        parameter.data -= self.lr * m_ / (np.sqrt(v_) + self.eps)
        moment_1[...] = m
        moment_2[...] = v
