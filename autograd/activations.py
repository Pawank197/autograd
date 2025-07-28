from tensor import Tensor
import numpy as np
from helper import unbroadcast, make_tensor

"""
Firstly, Activation functions
"""

def relu(x: Tensor) -> Tensor:
    """
    Applies the ReLU activation function element-wise.
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Output tensor with ReLU applied.
    """
    out = Tensor(np.maximum(0, x.data), (x,), 'relu')
    out.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            x.grad += unbroadcast((out.data > 0) * out.grad, x.shape)
    out._backward = _backward
    return out

def leaky_relu(x: Tensor, alpha=0.01) -> Tensor:
    """
    Applies the Leaky ReLU activation function element-wise.
    Args:
        x (Tensor): Input tensor.
        alpha (float): Slope for negative values.
    Returns:
        Tensor: Output tensor with Leaky ReLU applied.
    """
    out = Tensor(np.where(x.data > 0, x.data, alpha * x.data), (x,), 'leaky_relu')
    out.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            grad = np.where(out.data > 0, 1.0, alpha)
            x.grad += unbroadcast(grad * out.grad, x.shape)
    out._backward = _backward
    return out

def prelu(x: Tensor, alpha: Tensor) -> Tensor:
    """
    Applies the Parametric ReLU activation function element-wise.
    Args:
        x (Tensor): Input tensor.
        alpha (Tensor): Learnable negative slope, shape-compatible with x.
    Returns:
        Tensor: Output tensor with PReLU applied.
    """
    out = Tensor(np.where(x.data > 0, x.data, alpha.data * x.data), (x, alpha), 'prelu')
    out.requires_grad = x.requires_grad or alpha.requires_grad

    def _backward():
        if x.requires_grad:
            grad = np.where(x.data > 0, 1.0, alpha.data)
            x.grad += unbroadcast(grad * out.grad, x.shape)
        if alpha.requires_grad:
            # Gradient w.r.t alpha: sum over the negative part
            alpha.grad += unbroadcast((x.data * (x.data <= 0)) * out.grad, alpha.shape)
    out._backward = _backward
    return out

def GELU(x: Tensor) -> Tensor:
    """
    Applies the GELU activation function element-wise.
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Output tensor with GELU applied.
    """
    out = Tensor(0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3))), (x,), 'gelu')
    out.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            sqrt_2_pi = np.sqrt(2 / np.pi)
            tanh_arg = sqrt_2_pi * (x.data + 0.044715 * x.data**3)
            tanh_val = np.tanh(tanh_arg)
            sech2_val = 1 - tanh_val**2
            # Derivative of GELU: 0.5 * (1 + tanh) + 0.5 * x * sech^2 * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
            derivative = 0.5 * (1 + tanh_val) + 0.5 * x.data * sech2_val * sqrt_2_pi * (1 + 3 * 0.044715 * x.data**2)
            x.grad += unbroadcast(derivative * out.grad, x.shape)
    out._backward = _backward
    return out

def sigmoid(x: Tensor) -> Tensor:
    """
    Applies the sigmoid activation function element-wise.
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Output tensor with sigmoid applied.    
    """
    sig = 1 / (1 + (-x).exp())
    out = Tensor(sig.data , (x,), 'sigmoid')
    out.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            # derivative of sigmoid is sigmoid * (1 - sigmoid)
            x.grad += unbroadcast(out.data * (1 - out.data) * out.grad, x.shape)
    out._backward = _backward
    return out

def tanh(x: Tensor) -> Tensor:
    """
    Applies the tanh activation function element-wise.
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Output tensor with tanh applied.
    """
    out = Tensor(np.tanh(x.data), (x,), 'tanh')
    out.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            # derivative of tanh is 1 - tanh^2
            x.grad += unbroadcast((1 - out.data**2) * out.grad, x.shape)
    out._backward = _backward
    return out

def softmax(x: Tensor, dim=-1) -> Tensor:
    """
    Applies Softmax on the tensor
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Output tensor with softamx applied.
    """
    shifted = x - x.max(axis=dim, keepdims=True)
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(axis=dim, keepdims=True)
    out_data = exp_shifted.data / sum_exp.data
    out = Tensor(out_data, (x,), 'softmax')
    out.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            g = out.grad
            dot = (out.data * g).sum(axis=dim, keepdims=True)
            grad_x = out.data * (g - dot)
            x.grad += unbroadcast(grad_x, x.shape)
    out._backward = _backward
    return out