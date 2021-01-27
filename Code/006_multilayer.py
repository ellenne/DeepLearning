import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data

# 4 x 1
X = np.random.randn(4)


# 1 x 4 
X_T = X[None, :]

#  4 x 3
weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))

# 3 x 2
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network
#   1 x 3 = 1 x 4 * 4 x 3 
hidden_layer_in = np.matmul(X_T, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)


# 1 x 2 = 1 x 3 * 3 x 2
output_layer_in = np.matmul(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)