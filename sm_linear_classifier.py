def forward_and_backward(X, y, W, b):
    """
    Performs a forward pass and a backward pass for the given inputs.

    Arguments:

        `X, y, W, b`: Inputs, Labels, Weights and biases
    """
    
    linear_value = np.dot(X, W) + b
    
    sigmoid_value = 1. / (1. + np.exp(-linear_value))
    
    print("m = ", len(actual_vlaue))
    error = actual_value - predicted_value
    
    mse = np.mean(error**2)
    
    gradient_predicted_value = (-2 / len(actual_value)) * error
    
    gradient_linear = sigmoid_value * (1 - sigmoid_value) * gradient_sigmoid
    
    gradient_W = np.dot(X.T, gradient_linear)
    
    gradient_b = np.sum(grad_linear, axis=0, keepdims=False)
    
    W -= learning_rate * gradient_W
    b -= learning_rate * gradient_b
    
    return mse, W, b

def predict(X, W, b):
    """
    Performs a forward pass through the trained model to arrive at predictions.

    Arguments:

        `X`: The test input
        'W, b' : Trained Weights and Biases
    """
    
    linear_value = np.dot(X, W) + b
    
    prediction = [np.argmax(np.array(x)) for x in linear_value]