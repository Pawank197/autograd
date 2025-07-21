import numpy as np

"""
This module provides utility functions for broadcasting and unbroadcasting gradients in a computational graph.
"""

def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Unbroadcast a gradient to match the target shape.
    Args:
        grad (np.ndarray): The gradient to unbroadcast.
        target_shape (tuple): The target shape to match.
    Returns:
        np.ndarray: The unbroadcasted gradient.
    """
    #1) Prepend 1's to the gradient shape to match the target shape
    ndim_diff = grad.ndim - len(target_shape)
    shape = (1,)*ndim_diff + target_shape

    #2) identify the axes where target_dim is 1 but grad_dim > 1\
    sum_axes = [
        i for i , (g_dim, t_dim) in enumerate(zip(grad.shape, shape)) if t_dim == 1 and g_dim > 1
    ]

    #3) Sum over those axes
    if sum_axes:
        grad = grad.sum(axis=tuple(sum_axes), keepdims=True)

    # 4) Finally reshape the gradient to the target shape
    return grad.reshape(target_shape)

def make_tensor(tensor, Tensor):
    """
    checks if the input is a Tensor, if not converts it to a Tensor.
    Args:
        tensor: The input to convert.
        Tensor: The Tensor class to use for conversion.
    Returns:
        Tensor: The converted tensor.
    """
    return tensor if isinstance(tensor, Tensor) else Tensor(tensor)