"""
This file contains the implementation of a simple pytorch-like tensor library from scratch, using numpy.
It includes classes for representing values in a computational graph, performing a number of operations.
"""
import numpy as np
import math
from helper import unbroadcast, make_tensor


class Tensor:
    """
    This class represents a tensor in a computational graph, allowing for automatic differentiation(auto-grad).
    """
    def __init__(self, data, _children=(), _op='', label='', requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad}, label='{self.label}')"  
        
    # All the operations below are overloaded to work with both Tensor objects and numpy arrays, integers, or floats.

    def __add__(self, other):
        # If both are tensors, then add else show error
        if isinstance(other, (Tensor, np.ndarray, int, float)):
            other = make_tensor(other, Tensor)
            out = Tensor(self.data + other.data, (self, other), '+')
            out.requires_grad = self.requires_grad or other.requires_grad
            def _backward():
                if self.requires_grad:
                    self.grad += unbroadcast(1.0*out.grad, self.shape)
                if other.requires_grad:
                    other.grad += unbroadcast(1.0*out.grad, other.shape)
            out._backward = _backward
            return out
        raise TypeError(f"Unsupported operand type(s) for +: 'Tensor' and '{type(other).__name__}'")
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, (Tensor, np.ndarray, int, float)):
            other = make_tensor(other, Tensor)
            out = Tensor(self.data * other.data, (self, other), '*')
            out.requires_grad = self.requires_grad or other.requires_grad
            def _backward():
                if self.requires_grad:
                    self.grad  += unbroadcast(other.data*out.grad, self.data.shape)
                if other.requires_grad:
                    other.grad += unbroadcast(self.data*out.grad, other.data.shape)
            out._backward = _backward
            return out
        raise TypeError(f"Unsupported operand type(s) for *: 'Tensor' and '{type(other).__name__}'")
    
    def __rmul__(self, other):
        return self*other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f'^{other}')
        out.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast((other * self.data ** (other - 1)) * out.grad, self.shape)        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self*(-1.0)

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self*(other**-1)
    
    def __rtruediv__(self, other):
        return other*(self**-1)
    
    def __matmul__(self, other):
        if isinstance(other, (Tensor, np.ndarray)):
            other = make_tensor(other, Tensor)
            out = Tensor(self.data @ other.data, (self, other), '@')
            out.requires_grad = self.requires_grad or other.requires_grad
            def _backward():
                if self.requires_grad:
                    # dL/dA = dL/dC @ B.T
                    self.grad += unbroadcast(out.grad @ other.data.T, self.shape)
                if other.requires_grad:
                    # dL/dB = A.T @ dL/dC
                    other.grad += unbroadcast(self.data.T @ out.grad, other.shape)
            out._backward = _backward
            return out
        raise TypeError(f"Unsupported operand type(s) for @: 'Tensor' and '{type(other).__name__}'")
    
    # Some other important methods and operations
    
    def exp(self):
    # Exponential function
        out = Tensor(np.exp(self.data), (self,), 'exp')
        out.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                # The gradient of exp(x) is exp(x)
                self.grad += unbroadcast(out.data * out.grad, self.shape)
        out._backward = _backward
        return out
    
    def log(self):
    # Natural logarithm
        out = Tensor(np.log(self.data), (self,), 'log')
        out.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                # The gradient of log(x) is 1/x
                self.grad += unbroadcast(out.grad / self.data, self.shape)
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = (self,)
        out._op = 'sum'
        def _backward():
            if not self.requires_grad:
                return
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad
            else:
                grad = out.grad
                if not keepdims:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                    for ax in axes:
                        grad = np.expand_dims(grad, ax)
                self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        # Too lazy to write its backward pass, so its using the sum's + division's backward passes
        summed = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = 1
            for ax in axes:
                count *= self.shape[ax]
        out = summed * (1.0 / count)
        out.requires_grad = self.requires_grad
        out._prev = (summed,)
        out._op = 'mean'
        return out                        

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        out.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out
    
    def gather(self, indices, dim=0):
        if not isinstance(indices, (Tensor, np.ndarray)):
            raise TypeError(f"Expected indices to be a Tensor or np.ndarray, got {type(indices).__name__}")
        indices = make_tensor(indices, Tensor)

        # Forward pass using np.take_along_axis
        out_data = np.take_along_axis(self.data, indices.data, axis=dim)
        out = Tensor(out_data, (self, indices), 'gather')
        out.requires_grad = self.requires_grad

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            grad_input = np.zeros_like(self.data, dtype=np.float32)

            idxs = np.indices(indices.data.shape)
            idxs = list(idxs)
            idxs.insert(dim, indices.data)
            idxs = tuple(idxs)

            np.add.at(grad_input, idxs, grad)
            self.grad += grad_input

        out._backward = _backward
        return out

    @property
    def T(self):
        # Transpose operation for Tensor
        out = Tensor(self.data.T, (self,), 'transpose')
        out.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad.T, self.shape)
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        max_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_data, (self,), 'max')
        out.requires_grad = self.requires_grad
        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            max_keepdims = np.max(self.data, axis=axis, keepdims=True)
            mask = (self.data == max_keepdims).astype(np.float32)
            grad_input = mask * grad
            self.grad += unbroadcast(grad_input, self.shape)
        out._backward = _backward
        return out if axis is None or keepdims else out.reshape(max_data.shape)
    
    # Backward pass for autograd
    def backward(self, grad=None):
        # no grad is only allowed for scalar backwards
        if grad is None:
            if self.shape != ():
                raise ValueError("Gradients must be provided for non-scalar tensors.")
            grad = np.ones_like(self.data, dtype=np.float32)

        self.grad = grad.copy()
        # First, we need to get the topological order of the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # Now we can compute the gradients in reverse order
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        # Reset the gradients to zero
        self.grad = np.zeros_like(self.data, dtype=np.float32)