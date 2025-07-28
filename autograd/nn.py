from tensor import Tensor
import numpy as np
import activations

"""
This module provides classes for forming neural network layers
"""

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def add_module(self, name: str, module: "Module"):
        if not isinstance(module, Module):
            raise TypeError(f'Expected a Module, got {type(module).__name__}')
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def add_parameter(self, name: str, param: Tensor):
        if not isinstance(param, Tensor):
            raise TypeError(f"Expected a Tensor, got {type(param).__name__}")
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        for param in self.parameters():
            if getattr(param, 'requires_grad', False):
                param.grad = np.zeros_like(param.data)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class Sequential(Module):
    """
    A sequential container to hold multiple layers.
    Args:
        *layers: Variable number of layers to be added to the sequential container.
    """
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            if not isinstance(layer, Module):
                raise TypeError(f"EXpected a Module, got {type(layer).__name__}")
            self.add_module(f'layer_{i}', layer)
    
    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x

class Linear(Module):
    """
    A linear layer that applies a linear tranformation to the inputs.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    Outputs:
        Tensor: Output tensor after applying the linear transformation. (*, in_features) -> (*, out_features)
    """
    def __init__(self, in_features, out_features, requires_grad=True):
        super().__init__()
        w = Tensor(np.random.uniform(-1, 1, (out_features, in_features)), requires_grad=requires_grad, label=f'Linear weight')
        b   = Tensor(np.random.uniform(-1, 1, (out_features,)), requires_grad=requires_grad, label=f'linear bias')
        self.add_parameter('weight', w)
        self.add_parameter('bias', b)

    def forward(self, x):
        return x @ self.weight.T + self.bias
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return activations.relu(x)
