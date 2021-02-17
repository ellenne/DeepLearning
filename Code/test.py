import numpy as np

def tanh (x):
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

a = [[0]]
print (tanh(a))