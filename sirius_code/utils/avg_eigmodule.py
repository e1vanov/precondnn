import numpy as np
import scipy as sp

def avg_eigmodule(X, C):
    # X shape: (N, 1, L)
    # C shape: (N, 1, L)

    ans = []
    for i in range(X.shape[0]):
        A = sp.linalg.toeplitz(X[i,0])
        C_inv = sp.linalg.circulant(C[i,0])
        ans.append(np.max(np.abs(np.linalg.eigvalsh(A @ C_inv) - 1.)))

    return ans
