import numpy as np
import torch as t

def CrossEntropy(pred, targ):
    # pred: [N, c]
    # targ: [N, c]
    eps=1e-12
    pred = np.clip(pred, eps, 1-eps)
    ce = -np.sum(targ*np.log(pred),axis=-1)
    return ce

x = np.random.randn(2,3)
y = np.random.randn(2,3)
print(CrossEntropy(x,y))