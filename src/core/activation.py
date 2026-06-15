"""
    All activation layers are subclass of class Layer. The forward pass is their main function logic
    on the other hand, the backward pass is their derivative of the function

    Activation function are used to introduce non-linearity in the system for neural network to learn complexity.

    Following activation functions are implemented and their more description can be found in document.pdf
    * Binary Step Function
    * Linear Function
    * Sigmoid Activation Function
    * Tanh
    * Rectified Linear Unit (ReLU)
    * Leaky ReLU
    * Parameterised ReLU
    * Exponential Linear Unit
    * Softmax
"""
import numpy as np
from .layers import Layer

class BSF(Layer):
    def forward(self, X : np.ndarray):
        return 1 if X >= 0 else 0
    
    def backward(self, output_grad : np.ndarray):
        return 0

class LinearFunction(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, X : np.ndarray):
        return -self.alpha * X

    def backward(self, output_grad : np.ndarray):
        return self.alpha * output_grad

class sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X : np.ndarray):
        self.X = 1/(1 + np.exp(-X))
        return self.X

    def backward(self, output_grad : np.ndarray):
        return (self.X * (1 - self.X)) * output_grad

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X : np.ndarray):
        self.X = 2 * (1/(1 + np.exp(-2 * X))) -1
        return self.X

    def backward(self, output_grad : np.ndarray):
        return (1 - (self.X * self.X)) * output_grad

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X : np.ndarray):
        self.X = X
        return np.where(X >= 0, X, 0)

    def backward(self, output_grad : np.ndarray):
        return np.where(X >= 0, 1, 0) * output_grad

class LeakyReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X : np.ndarray):
        self.X = X
        return np.where(X >= 0, X, 0.01 * X)

    def backward(self, output_grad : np.ndarray):
        return np.where(X >= 0, 1, 0.01) * output_grad

class Parameterised_ReLU(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, X : np.ndarray):
        self.X = X
        return np.where(X >= 0, X, self.alpha * X)

    def backward(self, output_grad : np.ndarray):
        return np.where(self.X >= 0, 1, self.alpha) * output_grad

class EPU(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, X : np.ndarray):
        self.X = X
        return np.where(X >= 0, X, self.alpha * (np.exp(X) - 1))

    def backward(self, output_grad : np.ndarray):
        return np.where(self.X >= 0, self.X, self.alpha * np.exp(X)) * output_grad

class softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X : np.ndarray):
        # X - max(X) is for numerical stability
        Z = X - np.max(X, axis=-1, keepdims=True)
        self.X = np.exp(Z)/np.sum(np.exp(Z), axis=-1, keepdims=True)
        return self.X

    def backward(self, output_grad : np.ndarray):
        return self.X * (output_grad - np.sum(output_grad * self.X, axis=-1, keepdims=True))
