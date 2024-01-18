import math
import numpy as np
import torch as t

def softmax(x):
    # x: [N, c]
    max_x, _ = x.max(dim=-1, keepdim=True)
    shift_x = x - max_x
    exp_x = shift_x.exp()
    softmax = exp_x / t.sum(exp_x, dim=-1, keepdim=True)
    return softmax

def softmax_np(x):
    # x: [N, c]
    exp_x = np.exp(x-np.max(x, axis=-1, keepdims=True))
    softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return softmax

x = t.randn(2,3)
x_softmax = softmax(x)
x_softmax_np = softmax_np(x.numpy())
print(x)
print(x_softmax)
print(x_softmax_np)
