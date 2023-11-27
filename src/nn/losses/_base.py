class Loss():

    def calculate(self, true, predicted):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError