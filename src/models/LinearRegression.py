"""
    Linear Regression model build using Layer class.
"""
import numpy as np
from core.layers import Layer, DenseLayer
from core.activation import ReLU

class LinearRegression(Layer):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseLayer(input_dim=1, output_dim=10)
        self.layer_2 = DenseLayer(input_dim=10, output_dim=10)
        self.layer_3 = DenseLayer(input_dim=10, output_dim=1)
        
    def forward(self, X : np.ndarray):
        self.X = X
        return self.layer_3.forward(self.layer_2.forward(self.layer_1.forward(X)))

    def backward(self, output_grad : np.ndarray):
        grad = self.layer_3.backward(output_grad)
        grad = self.layer_2.backward(grad)
        grad = self.layer_1.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in [self.layer_1, self.layer_2, self.layer_3]:
            params.extend(layer.parameters())
        return params
