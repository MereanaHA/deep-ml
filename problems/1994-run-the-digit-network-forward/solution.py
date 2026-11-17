import numpy as np


def digit_network_forward(image, weights, biases):
    """Run an image through a fully connected sigmoid network.

    Returns (output activations rounded to 4 decimals, predicted digit).
    """
    def sigmoid(z):
        return 1.0/(1.0+ np.exp(-z))
    
    activations = np.asarray(image)
    activations = activations.reshape(-1)

    for i in range(len(weights)) :
        activations = sigmoid(np.dot(weights[i],activations) + biases[i])
    
    activations = np.round(activations,4)
    
    digit = 0
    
    for i in range(len(activations)):
        if activations[i]>activations[digit]:
            digit = i

    return [float(x) for x in activations],digit
    pass
