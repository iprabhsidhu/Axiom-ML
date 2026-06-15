"""
    Test models
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from utils.plots import plot_predictions
from models.LinearRegression import LinearRegression
from core.loss import MAE
from core.optim import SGD

class TestLinearRegression:
    @pytest.fixture(autouse=True)
    def sample_data(self):
        self.weights = 0.7
        self.bias = 0.3
        start = 0
        end = 1
        step = 0.01
        np.random.seed(42)
        # Create dummy data
        self.X = np.arange(start, end, step).reshape(-1, 1)
        self.y = self.weights * self.X + self.bias

    def test_model(self):
        train_split = int(0.8 * len(self.X))
        X_train, y_train = self.X[:train_split], self.y[:train_split]
        X_test, y_test = self.X[train_split:], self.y[train_split:]

        # Model
        model = LinearRegression()

        # Loss and Optimizers
        loss_fn = MAE()
        optimizer = SGD(model, lr=0.1)

        # Trainig loop
        epochs = 100

        for epoch in range(epochs):
            y_preds = model(X_train)
            loss = loss_fn(y_preds, y_train)

            optimizer.zero_grad()
            loss_fn.backward(model)
            optimizer.step()

            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)

            if epoch % 10 == 0:
                print(f"Epoch : {epoch} | Training loss : {loss} | Test loss: {test_loss}")
                print(f"weigh : {model.parameters()[0]} | bias : {model.parameters()[1]}")

        test_preds = model(X_test)
        plot_predictions(X_train, y_train, X_test, y_test, test_preds)
