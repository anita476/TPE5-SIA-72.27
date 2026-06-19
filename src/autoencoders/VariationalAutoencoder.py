import numpy as np
from autoencoders.Autoencoder import (
    Autoencoder,
    init_weights,
    sigmoid,
    sigmoid_grad,
)


class VariationalAutoencoder(Autoencoder):
    """Dense variational autoencoder, flat layout:
        [ enc_0 ... enc_{k-2} | W_mu | W_logvar | dec_0 ... dec_last ]
    Standard VAE loss = recon + KL (Kingma & Welling, 2014).
    """

    def __init__(self, layer_dims, activation, seed=None):
        super().__init__(layer_dims, activation, seed)

    # Weight initialisation                                              #
    def _add_layer(self, fan_in, fan_out):
        """Append one (weight, bias) pair to the flat parameter lists."""
        self.weights.append(init_weights(fan_in, fan_out, self.rng))
        self.biases.append(np.zeros((1, fan_out)))

    def _init_weights(self):
        dims = self.layer_dims
        k = self.bottleneck_idx
        lat = dims[k]  # latent dim
        h = dims[k - 1]  # last encoder-hidden dim

        self.weights, self.biases = [], []

        # encoder body: dims[0..k-1]
        for i in range(k - 1):
            self._add_layer(dims[i], dims[i + 1])
        self.n_enc = k - 1

        # two parallel heads  h_enc -> lat
        self.mu_idx = len(self.weights)
        self._add_layer(h, lat)
        self.logvar_idx = len(self.weights)
        self._add_layer(h, lat)

        # decoder body: dims[k..end]
        self.dec_start = len(self.weights)
        for i in range(k, len(dims) - 1):
            self._add_layer(dims[i], dims[i + 1])

    # Forward                                                            #
    def forward(self, x):
        # encoder body
        self._enc_z, self._enc_a = [], [x]
        a = x
        for i in range(self.n_enc):
            z = a @ self.weights[i] + self.biases[i]
            a = self.act(z)
            self._enc_z.append(z)
            self._enc_a.append(a)
        self._h_enc = a

        self._mu = a @ self.weights[self.mu_idx] + self.biases[self.mu_idx]
        self._logvar = a @ self.weights[self.logvar_idx] + self.biases[self.logvar_idx]

        # reparameterise: z = mu + std * eps,  eps ~ N(0, I)
        self._std = np.exp(0.5 * self._logvar)
        self._eps = self.rng.standard_normal(self._mu.shape)
        self._z = self._mu + self._std * self._eps

        # decoder body (sigmoid on the output layer)
        self._dec_z, self._dec_a = [], [self._z]
        a, n = self._z, len(self.weights)
        for i in range(self.dec_start, n):
            zi = a @ self.weights[i] + self.biases[i]
            a = sigmoid(zi) if i == n - 1 else self.act(zi)
            self._dec_z.append(zi)
            self._dec_a.append(a)
        return a

    def encode(self, x):
        a = x
        for i in range(self.n_enc):
            a = self.act(a @ self.weights[i] + self.biases[i])
        return a @ self.weights[self.mu_idx] + self.biases[self.mu_idx]

    def decode(self, z):
        a, n = z, len(self.weights)
        for i in range(self.dec_start, n):
            zi = a @ self.weights[i] + self.biases[i]
            a = sigmoid(zi) if i == n - 1 else self.act(zi)
        return a

    # Loss                                                               #
    @staticmethod
    def kl_divergence(mu, logvar):
        per_sample = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1)
        return np.mean(per_sample)

    def _loss(self, out, batch):
        N = batch.shape[0]
        recon = np.sum((out - batch) ** 2) / N
        return recon + self.kl_divergence(self._mu, self._logvar)

    # Backward (recon + KL)                                   #
    def _compute_grads(self, y_true):
        N = y_true.shape[0]
        n = len(self.weights)
        dW = [None] * n
        db = [None] * len(self.biases)

        # out -> z_sample
        delta = 2 * (self._dec_a[-1] - y_true) * sigmoid_grad(self._dec_z[-1])
        dz = None
        for local in reversed(range(n - self.dec_start)):
            gi = self.dec_start + local
            dW[gi] = self._dec_a[local].T @ delta / N
            db[gi] = np.mean(delta, axis=0, keepdims=True)
            if local > 0:
                delta = (delta @ self.weights[gi].T) * self.act_grad(self._dec_z[local - 1])
            else:
                dz = delta @ self.weights[gi].T  # into z_sample: no act grad

        # reparam trick (z = mu + std*eps) and add KL.
        # dKL/dmu = mu
        # dKL/dlogvar = 0.5 * (exp(logvar) - 1)
        d_mu = dz + self._mu
        d_logvar = dz * 0.5 * self._std * self._eps + 0.5 * (np.exp(self._logvar) - 1)

        # head gradients
        dW[self.mu_idx] = self._h_enc.T @ d_mu / N
        db[self.mu_idx] = np.mean(d_mu, axis=0, keepdims=True)
        dW[self.logvar_idx] = self._h_enc.T @ d_logvar / N
        db[self.logvar_idx] = np.mean(d_logvar, axis=0, keepdims=True)

        # encoder body backward
        delta = d_mu @ self.weights[self.mu_idx].T + d_logvar @ self.weights[self.logvar_idx].T
        for local in reversed(range(self.n_enc)):
            delta = delta * self.act_grad(self._enc_z[local])
            dW[local] = self._enc_a[local].T @ delta / N
            db[local] = np.mean(delta, axis=0, keepdims=True)
            if local > 0:
                delta = delta @ self.weights[local].T

        return dW, db

    def backward(self, y_true, lr):
        dW, db = self._compute_grads(y_true)
        self._apply_sgd(dW, db, lr)
