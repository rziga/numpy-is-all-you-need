import numpy as np
from ._base import Loss

class CategoricalCrossentropy(Loss):

    def calculate(self, true, predicted):
        self.in_shape = predicted.shape
        self.true = true.ravel()
        predicted = predicted.reshape(-1, self.in_shape[-1])
        self.p = predicted[range(predicted.shape[-2]), self.true]
        return -np.log(self.p).mean()
    
    def backward(self):
        D = np.zeros((len(self.p), self.in_shape[-1]))
        D[range(D.shape[0]), self.true] = -1 / self.p / self.in_shape[0]
        return D.reshape(self.in_shape)