import numpy as np
#                      401x3     401x1
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

# 401           3
n_records, n_features = features.shape
last_loss = None
# Initialize weights
# 3x2
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
# 2
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        #   1x2           3x1   * 3x2 
        hidden_input = np.dot(x, weights_input_hidden)
        #   1x2
        hidden_output = sigmoid(hidden_input)
        #   1x1                      1x2 * 2x1                  
        output = sigmoid(np.dot(hidden_output, 
                                weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        # 1x1 = 1x1 - 1x1
        error = y - output

        # TODO: Calculate error term for the output unit - delta Kappa with a_k being _output_
        #   1x1         =       1x1 *       1x1
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        #  1x1     = 1x1 * 1x1
        hidden_error =  np.dot(output_error_term, weights_hidden_output)
        
        # TODO: Calculate the error term for the hidden layer
        #    2x1                  1x1    *           2x1           *       1x1
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
        
        # TODO: Update the change in weights
        #     1x2                          2x1    *     1x2
        del_w_hidden_output += output_error_term * hidden_output
        #                            2x1       *     1x2
        del_w_input_hidden += hidden_error_term * x[:, None]

    # TODO: Update weights  (don't forget to division by n_records or number of samples)
    weights_input_hidden += learnrate / n_records * del_w_input_hidden 
    weights_hidden_output += learnrate / n_records * del_w_hidden_output 

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))