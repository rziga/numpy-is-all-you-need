from ..engine import search_params, Module


class Sequential(Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, next):
        d = next
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d