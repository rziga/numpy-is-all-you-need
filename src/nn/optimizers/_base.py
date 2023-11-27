class Optimizer():

    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad[:] = 0.0