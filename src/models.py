"""
    models.py contains all the Deep learning model in ready to use state.
    The following models are implemented.
    * Linear Regression
    * MultiLayer Perceptron (MLP)

    The unit tests for the models are written in `tests` directory
"""
import numpy as np
from core.layers import Layer, DenseLayer
from core.activation import ReLU

# Linear Regression Model
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

class MultiLayerPerceptron(Layer):
    def __init__(self):
        super().__init__()
        self.layer_0 = DenseLayer(input_dim=784, output_dim=128)
        self.activation_1 = ReLU()
        self.layer_2 = DenseLayer(input_dim=128, output_dim=256)
        self.activation_3 = ReLU()
        self.layer_4 = DenseLayer(input_dim=256, output_dim=128)
        self.activation_5 = ReLU()
        self.output_layer = DenseLayer(input_dim=128, output_dim=10)

    def forward(self, X : np.ndarray):
        self.X = X
        return self.output_layer(self.activation_5(self.layer_4(self.activation_3(self.layer_2(self.activation_1(self.layer_0(X)))))))

    def backward(self, output_grad : np.ndarray):
        grad = self.output_layer.backward(output_grad)
        grad = self.activation_5.backward(grad)
        grad = self.layer_4.backward(grad)
        grad = self.activation_3.backward(grad)
        grad = self.layer_2.backward(grad)
        grad = self.activation_1.backward(grad)
        grad = self.layer_0.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in [self.layer_0, self.layer_2, self.layer_4, self.output_layer]:
            params.extend(layer.parameters())
        return params
