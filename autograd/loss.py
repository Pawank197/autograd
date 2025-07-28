from tensor import Tensor
import numpy as np
from helper import unbroadcast
import activations

def MSELoss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Computes the mean squared error (MSE) between predicted and true values:
    MSE = mean((y_true - y_pred)²), averaged over all elements.
    Make sure both y_true and y_pred are of the same shape.
    Args:
        y_true (Tensor): Shape (*)
        y_pred (Tensor): Shape (*)
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

def CrossEntropy(logits: Tensor, target: Tensor) -> Tensor:
    """
    Computes the cross-entropy loss of logits and target.
    Used for multi-class classification tasks.
    The expected shape of input logits is (C) or (N, C) and of target (), (N).
    Args:
        logits (Tensor)
        target (Tensor) 
    Returns:
        Tensor: () or (N) shaped loss respectively.
    """
    # Check for shape compatibility
    if len(logits.shape) < 1:
        raise ValueError("y_pred must have at least 1 dimension (class logits)")

    if logits.shape[:-1] != target.shape:
        raise ValueError(f"Expected logits of shape (*, C) and target of shape (*) but got {logits.shape} and {target.shape}")

    """
    How does CrossEntropy works?
    So we have an input tensor logits of shape (*, C), where C is the number of classes.
    Now, firstly we apply softmax to y_pred to get the probabilities for each class.
    Then we take the log of these probabilities.
    Finally, we compute the negative log likelihood of the true class labels.
    """
    # apply softmax
    probs = activations.softmax(logits, dim=-1)
    # collect the probabilities of the true classes
    true_probs = probs.gather(target, dim=-1)
    # take the log of the probabilities
    log_probs = true_probs.log()
    # compute the negative log likelihood
    loss = -log_probs.mean()
    loss.requires_grad = True

    return loss