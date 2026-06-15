"""
    class Test_Layer is a group of tests performed on DenseLayer.
"""
import numpy as np
import pytest
from core.layers import DenseLayer

class TestDenseLayer:
    @pytest.fixture(autouse=True)
    def sample_data(self):
        self.X = np.random.randn(3,4)
        self.layer = DenseLayer(input_dim=4, output_dim=10)

    def test_initialize(self):
        assert self.layer.weights.shape == (4, 10)
        assert self.layer.bias.shape == (1, 10)

    def test_forward_shape(self):
        output = self.layer.forward(self.X)
        assert output.shape == (3, 10)

    def test_backward_shape(self):
        self.layer.forward(self.X)
        output_grad = np.ones((3, 10), dtype=float)
        
        input_grad = self.layer.backward(output_grad)
        
        assert input_grad.shape == (3,4)
