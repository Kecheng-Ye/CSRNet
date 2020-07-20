import numpy as np
from math import sqrt

def MAE(x : list, y : list, factor = 1):
    _sum = 0

    for i in range(len(x)):
        _sum += abs(np.sum(x[i]/factor) - np.sum(y[i]))
    
    return _sum/len(x)


def MSE(x : list, y : list, factor = 1):
    _sum = 0

    for i in range(len(x)):
        _sum += abs(np.sum(x[i]/factor) - np.sum(y[i]))**2
    
    return sqrt(_sum/len(x))


def PSNR(x : list, y : list):
    mse = MSE(x, y)
    EPS = 1e-8
    score = -10 * np.log10(mse + EPS)

    return score

