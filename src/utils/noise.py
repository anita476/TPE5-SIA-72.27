from __future__ import annotations

import numpy as np

NOISE_TYPES = ("gaussian", "salt_pepper", "masking")


def add_noise(
    X: np.ndarray,
    level: float,
    noise_type: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a corrupted copy of *X* (values in [0, 1]) without mutating it."""
    if rng is None:
        rng = np.random.default_rng()

    if level <= 0:
        return X.copy()

    if noise_type == "gaussian":
        noisy = X + rng.normal(0.0, level, size=X.shape)
        return np.clip(noisy, 0.0, 1.0)

    if noise_type == "salt_pepper":
        noisy = X.copy()
        flip = rng.random(X.shape) < level
        noisy[flip] = 1.0 - noisy[flip]
        return noisy

    if noise_type == "masking":
        noisy = X.copy()
        drop = rng.random(X.shape) < level
        noisy[drop] = 0.0
        return noisy

    raise ValueError(f"Unknown noise type '{noise_type}'. Options: {NOISE_TYPES}")
