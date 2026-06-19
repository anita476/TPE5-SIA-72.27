"""Olivetti faces loader.

Uses scikit-learn's fetch_olivetti_faces to load the dataset (downloads once
and caches automatically via sklearn).

    X:           (400, 4096)  float32 in [0, 1]   (64x64 grayscale, flattened)
    labels:      (400,)       int     subject ids 0..39
    label_names: list of 40 str

Requires: scikit-learn  (pip install scikit-learn)
"""
from __future__ import annotations

import numpy as np

ROWS, COLS = 64, 64

LABEL_NAMES = [f"subject_{i}" for i in range(40)]


def load_olivetti(seed: int = 0):
    """Load the full Olivetti faces dataset (400 images, 40 subjects).

    Returns (X, labels, label_names) where
        X:           (400, 4096) float32 in [0, 1]
        labels:      (400,)     int subject ids
        label_names: list of 40 str
    """
    try:
        from sklearn.datasets import fetch_olivetti_faces
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for the Olivetti dataset. "
            "Install it with: pip install scikit-learn"
        ) from e

    data = fetch_olivetti_faces(shuffle=True, random_state=seed)
    X = data.data.astype(np.float32)       # already (400, 4096) in [0, 1]
    labels = data.target                    # (400,) int 0..39

    return X, labels, LABEL_NAMES
