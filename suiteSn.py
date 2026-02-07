import numpy as np
from math import e


def SuiteSn(n):
    S = np.array([e-1])
    for k in range(1, n + 1):
        Sk = e - k * S[k - 1]
        S = np.append(S, Sk)

    return S
