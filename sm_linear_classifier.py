import numpy as np
from numba import jit

@jit
def forward_and_backward(X, y, W, b, learning_rate):
    """
    Performs a forward pass and a backward pass for the given inputs.

    Arguments:

        `X, y, W, b`: Inputs, Labels, Weights and biases
        'learning_rate': Hyper parameter that controls rate of learning
    """
    
    linear_value = np.dot(X, W) + b
    
    sigmoid_value = 1. / (1. + np.exp(-linear_value))
    
    error = y - sigmoid_value
    
    mse = np.mean(error**2)
    
    gradient_sigmoid = (-2 / len(y)) * error
    
    gradient_linear = sigmoid_value * (1 - sigmoid_value) * gradient_sigmoid
    
    gradient_W = np.dot(X.T, gradient_linear)
    
    gradient_b = np.sum(gradient_linear, axis=0, keepdims=False)
    
    W -= learning_rate * gradient_W
    b -= learning_rate * gradient_b

    """
    print("X =", X)
    print("y =", y)
    print("W =", W)
    print("b =", b)
    print("linear value =", linear_value)
    print("sigmoid_value =", sigmoid_value)
    print("error =", error)
    print("new W =", W)
    print("new b =", b)
    print("mse =", mse)
    print("gradient_sigmoid =", gradient_sigmoid)
    print("gradient_linear =", gradient_linear)
    print("gradient_W =", gradient_W)
    print("gradient_b =", gradient_b)

    print("X shape =", X.shape)
    print("y shape =", y.shape)
    print("W shape =", W.shape)
    print("b shape =", b.shape)
    print("linear_value shape =", linear_value.shape)
    print("sigmoid_value shape =", sigmoid_value.shape)
    print("m = ", len(y))
    print("error shape =", error.shape)
    print("mse shape =", mse.shape)
    print("gradient_sigmoid shape =", gradient_sigmoid.shape)
    print("gradient_linear shape = ", gradient_linear.shape)
    print("gradient_W shape = ", gradient_W.shape)
    print("gradient_b shape = ", gradient_b.shape)
    """
    
    return mse, W, b

def predict(X, W, b):
    """
    Performs a forward pass through the trained model to arrive at predictions.

    Arguments:

        `X`: The test input
        'W, b' : Trained Weights and Biases
    """
    
    linear_value = np.dot(X, W) + b
    
    predictions = [np.argmax(np.array(x)) for x in linear_value]
    
    """
    print("predicted linear_value = ", linear_value)
    print("prediction = ", predictions)
    """
    
    return predictions