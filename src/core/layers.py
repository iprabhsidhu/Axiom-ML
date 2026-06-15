"""
    Layer class is the base class for Neural Network operations on neurons.
    Desne Layers, Activations, Normalisations are all subclass of Layer.

    Layer consist of three main fucntion
    forward(self, X : ndarray) - takes the neurons forward in the neural network and is the main math logic of the respective layer
        parameters of the function:
        * X -> it is the feature matrix on which the defined logic will operate on. It must be a numpy array.

    backward(self, output_grad : ndarray, lr) - backpropogation derivative of the respective function with learning rate for function learn.
        backward(), can also be stated as a method that implements the chain rule.
        parameters of the functions:
        * output_grad -> It is the total error gadient flowing back from the subsequent layers.
        * lr -> also called learning rate. It is a small positive, which updates the internal parameters (weights and bais) of the model/layer.
    
    Dense Layer which is a linear transformation layer and child class of Layer.
    
    __init__(self, input_dim, output_dim) - Initializes the linear transformation layer with weight of shape (input_dim, output_dim) and bias off
        shape (1, output_dim).
    
    forward(self, X : ndarray) - defines the forward pass with mathematical formula Z = XW + B
        where, W and B are the respective weights and bias.
    
    backward(self, output_grad, lr=0.1) - calculates the weight and bias gradient passed from subsequent forward layer and updates internal params.
        It also passes the input gradient to next backward layer.
"""
import numpy as np

class Layer():
    def __init__(self):
        self.X = 0 # initialize the feature matrix later in subclasses
   
    # takes the neuron forward in neural network
    def forward(self, X : np.ndarray):
        raise NotImplementedError("Forward method must be defined")
    
    # takes the neuron backward in neural network
    def backward(self, output_grad : np.ndarray):
        raise NotImplementedError("Backward method must be defined")
    
    def __call__(self, X : np.ndarray):
        return self.forward(X)

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        limit = 1.0 / input_dim
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.random.uniform(-limit, limit, (1, output_dim))
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)


    def forward(self, X : np.ndarray):
        self.X = X
        return np.dot(self.X, self.weights) + self.bias
        
    def backward(self, output_grad : np.ndarray, lr=0.1):
        # Calculate weight and bias gradient
        self.weights_grad[:] = self.X.T @ output_grad
        self.bias_grad[:] = np.sum(output_grad, axis=0, keepdims=True)

        # Calculates the input gradient
        input_grad = np.dot(output_grad, self.weights.T)
        
        # Send the error backward
        return input_grad

    def parameters(self):
        return (self.weights, self.weights_grad), (self.bias, self.bias_grad)
