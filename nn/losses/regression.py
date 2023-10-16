from ._base import Loss

class MeanSquaredError(Loss):

    def calculate(self, true, pred):
        self.diff = pred - true
        return (self.diff ** 2).mean()

    def backward(self):
        return 2 * (self.diff) / len(self.diff)