import numpy as np

from autoencoders.SimpleAutoencoder import SimpleAutoencoder
from utils.noise import add_noise


class DenoisingAutoencoder(SimpleAutoencoder):
    def __init__(
        self,
        layer_dims,
        activation,
        seed,
        noise_type: str = "gaussian",
        noise_level: float = 0.3,
        noise_seed=None,
        loss: str = "mse",
        weight_init: str = "he",
    ):
        super().__init__(layer_dims, activation, seed, loss=loss, weight_init=weight_init)
        self.noise_type = noise_type
        self.noise_level = noise_level
        self._noise_rng = np.random.default_rng(
            noise_seed if noise_seed is not None else seed
        )

    def _corrupt_input(self, batch):
        return add_noise(
            batch,
            level=self.noise_level,
            noise_type=self.noise_type,
            rng=self._noise_rng,
        )
