# This code implements a simple feedforward neural network with custom activation functions and L2 regularization.
# The key components are:

# 1. **Activation Class**: An abstract base class for activation functions with two static methods:
#    - `function(x)`: Computes the activation value for the input `x`.
#    - `derivative(x)`: Computes the derivative of the activation function at `x`.

# 2. **SafeSigmoid Class**: A subclass of the `Activation` class implementing the Sigmoid activation function:
#    - The `function(x)` method safely computes the Sigmoid by clipping the input values to avoid overflow.
#    - The `derivative(x)` method computes the Sigmoid derivative, also ensuring safe output with a small epsilon.

# 3. **ClippedReLU Class**: A subclass of the `Activation` class implementing the ReLU (Rectified Linear Unit) activation function:
#    - The `function(x)` method clips the output to the range [0, 20].
#    - The `derivative(x)` method computes the ReLU derivative, returning 1 for positive values and 0 for others, with clipping at the defined bounds.

# 4. **Layer Class**: Represents a layer in the neural network, with methods for:
#    - `initialize_parameters()`: Initializes weights (`W`) and biases (`b`) using He or Xavier initialization depending on the chosen method.
#    - `forward(X)`: Computes the forward pass through the layer (input to output).
#    - `backward(grad, prev_a, learning_rate)`: Performs the backward pass (gradient computation and weight update), using gradient clipping and L2 regularization.

# 5. **NeuralNetwork Class**: Represents the entire neural network with methods for:
#    - `add(layer)`: Adds a layer to the network.
#    - `compile()`: Initializes the parameters of each layer based on the input dimensions.
#    - `forward(X)`: Propagates the input `X` through all layers.
#    - `backward(X, y, learning_rate)`: Performs the backpropagation algorithm to update the weights and biases.
#    - `train(X, y, epochs, learning_rate, verbose)`: Trains the network using the provided data `X` and labels `y` for a specified number of epochs with gradient updates.
#    - `plot_loss()`: Plots the training loss over time.
#    - `predict(X)`: Predicts the output for input data `X` after training, using a threshold of 0.5 for binary classification.

# The neural network can be trained on a small dataset `X` with corresponding binary labels `y`, and the loss is tracked over epochs.
# After training, predictions can be made using the trained model.

# Example usage:
# A small dataset `X` and labels `y` are defined. The network is built with a Layer having 4 units and ClippedReLU activation,
# followed by a Layer with 1 unit and SafeSigmoid activation.
# The network is trained for 10000 epochs, and the training loss is plotted. Predictions are made on new test data.


import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Activation(ABC):
    @staticmethod
    @abstractmethod
    def function(x):
        pass
    
    @staticmethod
    @abstractmethod
    def derivative(x):
        pass

class SafeSigmoid(Activation):
    @staticmethod
    def function(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        x = np.clip(x, 1e-12, 1-1e-12)
        return x * (1 - x)

class ClippedReLU(Activation):
    @staticmethod
    def function(x):
        return np.clip(x, 0, 20)
    
    @staticmethod
    def derivative(x):
        return np.where((x > 0) & (x < 20), 1, 0)
    
class Layer:
    def __init__(self, units, activation, input_dim=None, initialization='he', l2=0.01):
        self.units = units
        self.activation = activation
        self.input_dim = input_dim
        self.initialization = initialization
        self.l2 = l2  # L2 regularization
        self.W = None
        self.b = None
        self.initialize_parameters()
        
    def initialize_parameters(self):
        if self.input_dim is not None:
            scale = {
                'he': np.sqrt(2. / self.input_dim),
                'xavier': np.sqrt(2. / (self.input_dim + self.units)),
                'glorot': np.sqrt(6. / (self.input_dim + self.units))
            }.get(self.initialization, 0.01)
            
            self.W = np.random.randn(self.input_dim, self.units) * scale
            self.b = np.zeros((1, self.units))
    
    def forward(self, X):
        self.z = np.dot(X, self.W) + self.b
        self.a = self.activation.function(self.z)
        return self.a
    
    def backward(self, grad, prev_a, learning_rate):
        m = prev_a.shape[0]
        delta = grad * self.activation.derivative(self.a if isinstance(self.activation, SafeSigmoid) else self.z)
        
        # Gradient clipping
        delta = np.clip(delta, -5, 5)
        
        dW = np.dot(prev_a.T, delta)/m + (self.l2 * self.W)/m  # L2 regularization
        db = np.sum(delta, axis=0, keepdims=True)/m
        grad_back = np.dot(delta, self.W.T)
        
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return grad_back

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []

    def add(self, layer):
        if len(self.layers) > 0:
            layer.input_dim = self.layers[-1].units
        self.layers.append(layer)

    def compile(self):
        for i in range(1, len(self.layers)):
            self.layers[i].input_dim = self.layers[i - 1].units
            self.layers[i].initialize_parameters()

    def forward(self, X):
        a = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, X, y, learning_rate):
        self.y_pred = self.forward(X)
        grad = np.clip(self.y_pred - y.reshape(-1, 1), -1e8, 1e8)

        for i in reversed(range(1, len(self.layers))):
            prev_layer = self.layers[i - 1]
            grad = self.layers[i].backward(grad, prev_layer.a, learning_rate)

        # Handle the first layer separately
        self.layers[0].backward(grad, X, learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        for epoch in range(epochs):
            y_pred = np.clip(self.forward(X), 1e-12, 1 - 1e-12)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

            # Add L2 regularization
            for layer in self.layers:
                if layer.W is not None:
                    loss += 0.5 * layer.l2 * np.sum(layer.W ** 2)

            self.loss_history.append(loss)
            self.backward(X, y, learning_rate)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def predict(self, X):
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)

# Usage example
if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 0, 1, 1])

    model = NeuralNetwork()
    model.add(Layer(units=4, activation=ClippedReLU, input_dim=3, l2=0.1))
    model.add(Layer(units=1, activation=SafeSigmoid, l2=0.1))
    model.compile()

    model.train(X, y, epochs=10000, learning_rate=0.01)
    model.plot_loss()
    
    X_test = np.array([[1, 2, 3], [4, 5, 6]])
    print("Predictions:", model.predict(X_test))













  