"""Fashion-MNIST loader.

Reads local IDX .gz files from data/fashion/. If missing, attempts to
download them (tries two mirrors). Place files manually if download fails:

    data/fashion/train-images-idx3-ubyte.gz
    data/fashion/train-labels-idx1-ubyte.gz

Returns a subsampled dataset in the same shape the rest of the pipeline expects.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

ROWS, COLS = 28, 28

_MIRRORS = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
]
_FILES = {
    "images": "train-images-idx3-ubyte.gz",
    "labels": "train-labels-idx1-ubyte.gz",
}

LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fashion")


def _download_file(fname: str, dest: str) -> None:
    """Try each mirror until one succeeds."""
    for base_url in _MIRRORS:
        url = base_url + fname
        try:
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            print(f"  Mirror failed ({e}), trying next...")
    raise RuntimeError(
        f"Could not download {fname}. Place it manually in {os.path.dirname(dest)}"
    )


def _ensure_downloaded(cache_dir: str = _CACHE_DIR) -> tuple[str, str]:
    """Return paths to (images_gz, labels_gz), downloading if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for key, fname in _FILES.items():
        dest = os.path.join(cache_dir, fname)
        if not os.path.isfile(dest):
            _download_file(fname, dest)
        paths[key] = dest
    return paths["images"], paths["labels"]


def _parse_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _parse_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _magic, _n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_fashion_mnist(
    n_samples: int = 4000,
    seed: int = 0,
    cache_dir: str = _CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a stratified subsample of Fashion-MNIST.

    Returns (X, labels, label_names) where
        X:           (n_samples, 784) float32 in [0, 1]
        labels:      (n_samples,)     int class ids
        label_names: list of 10 str
    """
    img_path, lbl_path = _ensure_downloaded(cache_dir)
    X_all = _parse_idx_images(img_path)
    y_all = _parse_idx_labels(lbl_path)

    rng = np.random.default_rng(seed)
    n_classes = 10
    per_class = n_samples // n_classes

    chosen = []
    for c in range(n_classes):
        idxs = np.where(y_all == c)[0]
        pick = rng.choice(idxs, size=min(per_class, len(idxs)), replace=False)
        chosen.append(pick)
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)

    return X_all[chosen], y_all[chosen], LABEL_NAMES
