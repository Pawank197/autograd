from tensor import Tensor
import numpy as np
from nn import Module

class SGD(Module):
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, parameters, lr=0.01):
        super().__init__()
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if getattr(param, 'requires_grad', False):
                param.data -= self.lr * param.grad
                param.grad = np.zeros_like(param.data)
        
    def zero_grad(self):
        for param in self.parameters:
            if getattr(param, 'requires_grad', False):
                param.grad = np.zeros_like(param.data)