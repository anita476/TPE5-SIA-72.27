import numpy as np
from autoencoders.Autoencoder import Autoencoder, mse_grad, sigmoid, sigmoid_grad


class SimpleAutoencoder(Autoencoder):
    def __init__(self, layer_dims, activation, seed):
        super().__init__(layer_dims, activation, seed)

    def forward(self, x):
        self.z = []       # pre-activations at each layer
        self.a = [x]      # activations; a[0] = raw input

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.a[-1] @ W + b
            self.z.append(z)

            is_last = (i == len(self.weights) - 1)
            # Output layer always uses sigmoid to bound values in [0, 1]
            a = sigmoid(z) if is_last else self.act(z)
            self.a.append(a)

        return self.a[-1]

    def backward(self, y_true, lr):
        m = y_true.shape[0]
        dW_list = [None] * len(self.weights)
        db_list = [None] * len(self.biases)

        # Start delta at the output layer (sigmoid + MSE)
        delta = mse_grad(self.a[-1], y_true) * sigmoid_grad(self.z[-1])

        for i in reversed(range(len(self.weights))):
            dW_list[i] = self.a[i].T @ delta / m
            db_list[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.act_grad(self.z[i - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= lr * dW_list[i]
            self.biases[i]  -= lr * db_list[i]

    def encode(self, x):
        a = x
        for i in range(self.bottleneck_idx):
            a = self.act(a @ self.weights[i] + self.biases[i])
        return a

    def decode(self, z):
        a = z
        for i in range(self.bottleneck_idx, len(self.weights)):
            is_last = (i == len(self.weights) - 1)
            a = sigmoid(a @ self.weights[i] + self.biases[i]) if is_last \
                else self.act(a @ self.weights[i] + self.biases[i])
        return a


