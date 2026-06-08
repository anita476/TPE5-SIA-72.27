import numpy as np
from utils.activations import relu, tanh, tanh_grad, sigmoid, sigmoid_grad,relu_grad


ACTIVATIONS = {
    "relu":    (relu,    relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh":    (tanh,    tanh_grad),
}

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size


# @todo maybe other weight inits
def init_weights(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)



class Autoencoder:
    def __init__(self, layer_dims, activation, seed=None):
        """
        layer_dims : list of int, the middle value is the bottleneck
        """
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(ACTIVATIONS)}")

        self.layer_dims = layer_dims
        self.activation_name = activation
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.bottleneck_idx = len(layer_dims) // 2

        self._init_weights()

    def _init_weights(self):
        np.random.seed(self.seed)  # for reproducible He init
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_dims) - 1):
            self.weights.append(init_weights(self.layer_dims[i], self.layer_dims[i + 1]))
            self.biases.append(np.zeros((1, self.layer_dims[i + 1])))

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward().")

    def backward(self, y_true, lr):
        raise NotImplementedError("Subclasses must implement backward().")

    def encode(self, x):
        raise NotImplementedError("Subclasses must implement encode().")

    def decode(self, z):
        raise NotImplementedError("Subclasses must implement decode().")

    def train(self, X, epochs, lr, batch_size, log_every):
        """
        Train the autoencoder in-place.

        Parameters
        ----------
        X          : ndarray (n_samples, n_features)
        epochs     : number of full passes over the data
        lr         : learning rate
        batch_size : mini-batch size
        log_every  : print loss every N epochs (0 = silent)
        """
        for epoch in range(epochs):
            # Seeded shuffle — reproducible across runs with the same seed
            idx = self.rng.permutation(len(X))
            X_shuffled = X[idx]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X), batch_size):
                batch = X_shuffled[start:start + batch_size]
                out = self.forward(batch)
                epoch_loss += mse(out, batch)
                self.backward(batch, lr)
                n_batches += 1

            if log_every and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f}")