from tensor import Tensor
import numpy as np
from helper import unbroadcast, make_tensor

def MSELoss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Computes the mean squared error (scalar) between true and predicted values.
    Args:
        y_true (Tensor): Shape (B, C)
        y_pred (Tensor): Shape (B, C)
    Returns:
        Tensor: Scalar loss
    """
    diff = y_true - y_pred                      # shape (B, C)
    mse = (diff ** 2).sum() / diff.data.size    # scalar
    mse.requires_grad = True

    def _backward():
        if y_pred.requires_grad:
            y_pred.grad += unbroadcast((-2 * diff.data / diff.data.size) * mse.grad, y_pred.shape)
    mse._backward = _backward
    return mse

def CrossEntropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    pass