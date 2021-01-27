import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

# My implementation
def cross_entropy(Y, P):
    ret = 0
    for i in range(len(Y)):
        if Y[i] == 1:
            ret += np.log(P[i])
        else:
            ret += np.log(1-P[i])
    return -ret

# Udacity Solution
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))