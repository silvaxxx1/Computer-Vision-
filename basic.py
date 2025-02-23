"""
Neural Network Implementation from Scratch using NumPy

### Approach:
1. **Data Preparation:**
   - We define a simple dataset `X` (input features) and `y` (binary labels).
   - The dataset consists of 4 samples, each with 3 features.

2. **Activation Functions:**
   - We use ReLU for hidden layers to introduce non-linearity.
   - We use Sigmoid for the output layer to map values between 0 and 1.

3. **Loss Function:**
   - We use Binary Cross-Entropy (BCE) since it's a classification task with binary labels.
   - The BCE gradient is computed for backpropagation.

4. **Forward Propagation:**
   - Computes weighted sums (`Z`) and applies activation functions (`A`).
   - The results are stored for later use in backpropagation.

5. **Backward Propagation:**
   - Computes gradients using the chain rule.
   - Updates weights and biases using gradient descent.

6. **Training Process:**
   - Iterates over epochs to adjust weights and minimize loss.
   - Stores loss history for visualization.

7. **Prediction & Evaluation:**
   - Uses trained weights to predict outputs for test samples.
   - Applies a threshold (0.5) to classify predictions as 0 or 1.

### Next Steps:
- Convert this into an object-oriented class for better modularity.
- Add batch training instead of full-batch gradient descent.
- Introduce additional layers and dropout for robustness.

"""


import numpy as np
import matplotlib.pyplot as plt

# Define the mock dataset
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

y = np.array([0, 0, 1, 1])

# Check the shape
m, n = X.shape
num_classes = len(np.unique(y))

print("the number of samples is : ", m)
print("the number of features is : ", n)
print("the number of classes is : ", num_classes)

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def back_sigmoid(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def back_relu(x):
    return np.where(x >= 0, 1, 0)

def loss_fn(y_true, y_pred, eps=1e-6):
    """ Compute the binary cross-entropy loss """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0) issues
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def loss_derivative(y_true, y_pred, eps=1e-15):
    """ Compute the gradient of BCE loss w.r.t y_pred """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def initialize_parameters(layer_dims, init_type='he'):
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        prev = layer_dims[l - 1]
        curr = layer_dims[l]

        if init_type == 'he':
            params[f'W{l}'] = np.random.randn(prev, curr) * np.sqrt(2. / prev)
        elif init_type == 'xavier':
            params[f'W{l}'] = np.random.randn(prev, curr) * np.sqrt(2. / (prev + curr))
        else:
            params[f'W{l}'] = np.random.randn(prev, curr) * 0.01
        params[f'b{l}'] = np.zeros((1, curr))

    return params

def forward(X, params, activation=['relu', 'sigmoid']):
    L = len(params) // 2  # Number of layers
    A = X

    activation_functions = {'relu': relu, 'sigmoid': sigmoid}
    A_dict = {0: X}
    Z_dict = {}

    for l in range(1, L + 1):
        W = params[f'W{l}']
        b = params[f'b{l}']
        Z = np.dot(A, W) + b
        activation_func = activation[l - 1] if l - 1 < len(activation) else activation[-1]
        A = activation_functions[activation_func](Z)
        A_dict[l] = A
        Z_dict[l] = Z

    return A, Z, A_dict, Z_dict

def backward(X, y, params, activation=['relu', 'sigmoid'], learning_rate=0.01):
    L = len(params) // 2  # Number of layers
    m = X.shape[0]  # Number of training examples

    # Forward pass to get activations and pre-activations
    _, _, A_dict, Z_dict = forward(X, params, activation)
    A = A_dict
    Z = Z_dict

    # Initialize gradients
    dW = {}
    db = {}

    # Output layer gradient
    dZ = A[L] - y.reshape(-1, 1)  # Ensure y is correctly shaped
    dW[L] = np.dot(A[L - 1].T, dZ) / m
    db[L] = np.sum(dZ, axis=0, keepdims=True) / m

    # Backpropagate through hidden layers
    for l in range(L - 1, 0, -1):
        # Determine activation function
        activation_func = activation[l - 1] if (l - 1) < len(activation) else activation[-1]
        # Compute derivative based on activation
        if activation_func == 'relu':
            derivative = back_relu(Z[l])
        elif activation_func == 'sigmoid':
            derivative = back_sigmoid(A[l])
        else:
            raise ValueError(f"Unsupported activation function: {activation_func}")

        dZ = np.dot(dZ, params[f'W{l + 1}'].T) * derivative
        dW[l] = np.dot(A[l - 1].T, dZ) / m
        db[l] = np.sum(dZ, axis=0, keepdims=True) / m

    # Update parameters
    for l in range(1, L + 1):
        params[f'W{l}'] -= learning_rate * dW[l]
        params[f'b{l}'] -= learning_rate * db[l]

    return params

def train(X, y, params, activation=['relu', 'sigmoid'], learning_rate=0.01, epochs=1000, verbose=True):
    loss_history = []
    for epoch in range(epochs):
        params = backward(X, y, params, activation, learning_rate)
        y_pred, _, _, _ = forward(X, params, activation)
        loss = loss_fn(y.reshape(-1, 1), y_pred)
        loss_history.append(loss)
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    return params, loss_history

def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.title("Loss over training epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def predict(X, params, activation=['relu', 'sigmoid']):
    y_pred, _, _, _ = forward(X, params, activation)
    return (y_pred > 0.5).astype(int)

# Initialize parameters and train the model
params = initialize_parameters([3, 4, 1])  # Layer sizes: 3 inputs, 4 hidden neurons, 1 output
final_params, loss_history = train(X, y, params, activation=['relu', 'sigmoid'], learning_rate=0.01, epochs=1000)

# Plot the loss history
plot_loss(loss_history)

# Test predictions
X_test = np.array([[1, 2, 3], [4, 5, 6]])
predictions = predict(X_test, final_params)
print("Predictions on new data:", predictions)