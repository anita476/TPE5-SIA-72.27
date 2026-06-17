import numpy as np
from utils.activations import relu, tanh, tanh_grad, sigmoid, sigmoid_grad, relu_grad


ACTIVATIONS = {
    "relu":    (relu,    relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh":    (tanh,    tanh_grad),
}

OPTIMIZERS = ("sgd", "adam")

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
        self._init_adam_state()

    def _init_weights(self):
        np.random.seed(self.seed)
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_dims) - 1):
            self.weights.append(init_weights(self.layer_dims[i], self.layer_dims[i + 1]))
            self.biases.append(np.zeros((1, self.layer_dims[i + 1])))

    def _init_adam_state(self):
        """Initialise first/second moment buffers for Adam."""
        self._adam_t  = 0
        self._adam_mw = [np.zeros_like(w) for w in self.weights]
        self._adam_vw = [np.zeros_like(w) for w in self.weights]
        self._adam_mb = [np.zeros_like(b) for b in self.biases]
        self._adam_vb = [np.zeros_like(b) for b in self.biases]

    def _corrupt_input(self, batch):
        return batch

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward().")

    def _compute_grads(self, y_true):
        """Return (dW_list, db_list) without modifying weights.
        Requires forward() to have been called first (populates self.a, self.z).
        Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _compute_grads().")

    def backward(self, y_true, lr):
        raise NotImplementedError("Subclasses must implement backward().")

    def encode(self, x):
        raise NotImplementedError("Subclasses must implement encode().")

    def decode(self, z):
        raise NotImplementedError("Subclasses must implement decode().")

    def _apply_sgd(self, dW_list, db_list, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * dW_list[i]
            self.biases[i]  -= lr * db_list[i]

    def _apply_adam(self, dW_list, db_list, lr,
                    beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        t = self._adam_t
        for i in range(len(self.weights)):
            # ── weights ──────────────────────────────────────────────────────
            self._adam_mw[i] = beta1 * self._adam_mw[i] + (1 - beta1) * dW_list[i]
            self._adam_vw[i] = beta2 * self._adam_vw[i] + (1 - beta2) * dW_list[i] ** 2
            mw_hat = self._adam_mw[i] / (1 - beta1 ** t)
            vw_hat = self._adam_vw[i] / (1 - beta2 ** t)
            self.weights[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)

            # ── biases ───────────────────────────────────────────────────────
            self._adam_mb[i] = beta1 * self._adam_mb[i] + (1 - beta1) * db_list[i]
            self._adam_vb[i] = beta2 * self._adam_vb[i] + (1 - beta2) * db_list[i] ** 2
            mb_hat = self._adam_mb[i] / (1 - beta1 ** t)
            vb_hat = self._adam_vb[i] / (1 - beta2 ** t)
            self.biases[i]  -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def train(self, X, epochs, lr, batch_size, log_every, optimizer="adam"):
        """
        Train the autoencoder in-place
        """
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: {OPTIMIZERS}")

        # Reset Adam state at the start of each train() call
        if optimizer == "adam":
            self._init_adam_state()

        for epoch in range(epochs):
            idx = self.rng.permutation(len(X))
            X_shuffled = X[idx]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, len(X), batch_size):
                batch = X_shuffled[start:start + batch_size]
                net_input = self._corrupt_input(batch)
                out   = self.forward(net_input)
                epoch_loss += mse(out, batch)

                dW_list, db_list = self._compute_grads(batch)

                if optimizer == "adam":
                    self._apply_adam(dW_list, db_list, lr)
                else:
                    self._apply_sgd(dW_list, db_list, lr)

                n_batches += 1

            if log_every and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f}")

    # new Autoencoder.train that captures per-epoch loss
    def train_and_collect(self,X, epochs, lr, batch_size, log_every, optimizer) -> list[float]:
        """Run training and return the mean batch loss for every epoch.

        We replicate the training loop here rather than monkey-patching so that
        the original ``Autoencoder.train`` stays untouched.
        """
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: {OPTIMIZERS}")

        if optimizer == "adam":
            self._init_adam_state()

        epoch_losses: list[float] = []

        for epoch in range(epochs):
            idx = self.rng.permutation(len(X))
            X_shuffled = X[idx]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X), batch_size):
                batch = X_shuffled[start:start + batch_size]
                net_input = self._corrupt_input(batch)
                out = self.forward(net_input)
                epoch_loss += mse(out, batch)

                dW_list, db_list = self._compute_grads(batch)

                if optimizer == "adam":
                    self._apply_adam(dW_list, db_list, lr)
                else:
                    self._apply_sgd(dW_list, db_list, lr)

                n_batches += 1

            mean_loss = epoch_loss / n_batches
            epoch_losses.append(mean_loss)

            if log_every and (epoch + 1) % log_every == 0:
                print(f"  [run] Epoch {epoch + 1}/{epochs} | Loss: {mean_loss:.10f}")

        return epoch_losses