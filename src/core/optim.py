"""
    All the optimizers are implemented here. Optimizer takes the loss gradient and updates parameter of the model accordingly.
    Optimizers are algorithms that minimize a loss function. They are dependent on model's lernable parameter {weights, biases}

    There are different type of optimizers, which are implemented here
    * Gradient Descent
    * Stochastic Gradient Descent (SGD)
    * Adaptive Momentum Estimation (Adam)

    all optimizers are subclass of class optimizer which has two main methods
    * zero_grad() - must be called before next optimize step to set old gradients to zero.
    * step() - optimizes the parameter based on new loss gradient
"""
from .layers import Layer

class Optimizer():
    def __init__(self, model : Layer, lr=0.1):
        self.lr = lr
        self.output_gradient = None
        self.model = model

    def step(self):
        raise NotImplementedError("Must be overriden in subclass.")

    def zero_grad(self):
        for param, grad in self.model.parameters():
            if grad is not None:
                grad.fill(0)

class SGD(Optimizer):
    def step(self):
        for param, grad in self.model.parameters():
            param[:] -= self.lr * grad  
