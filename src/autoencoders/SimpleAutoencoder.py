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

    def _compute_grads(self, y_true):
        """Backprop, returns (dW_list, db_list) without touching weights."""
        dW_list = [None] * len(self.weights)
        db_list = [None] * len(self.biases)

        # Output layer: sigmoid + MSE combined gradient
        delta = mse_grad(self.a[-1], y_true) * sigmoid_grad(self.z[-1])

        for i in reversed(range(len(self.weights))):
            dW_list[i] = self.a[i].T @ delta
            db_list[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.act_grad(self.z[i - 1])

        return dW_list, db_list

    def backward(self, y_true, lr):
        """SGD backward pass (kept for compatibility)."""
        dW_list, db_list = self._compute_grads(y_true)
        self._apply_sgd(dW_list, db_list, lr)

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