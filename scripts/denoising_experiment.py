"""denoising_experiment.py
Train a Denoising Autoencoder and study its noise-removal capability.

Usage:
    python scripts/denoising_experiment.py --config configs/default_denoising.json

Extra config keys: noise_type, noise_level (training), noise_levels (eval list).
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import plot_style
from autoencoders.DenoisingAutoencoder import DenoisingAutoencoder
from utils.config_loader import load_config
from utils.denoising_eval import run_denoising_study
from utils.font_loader import load_font


def _plot_loss(losses, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(losses, linewidth=0.8, color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (vs clean input)")
    ax.set_title("Denoising Autoencoder training")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder experiment")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))

    out_dir = resolve(cfg.get("out", "output/denoising"))
    os.makedirs(out_dir, exist_ok=True)

    layer_dims = cfg["layer_dims"]
    activation = cfg["activation"]
    optimizer = cfg["optimizer"]
    seed = cfg["seed"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    log_every = cfg.get("log_every", 0)
    threshold = cfg.get("threshold", 0.5)
    max_errors = cfg.get("max_errors", 1)
    patience = cfg.get("patience")
    min_delta = cfg.get("min_delta", 1e-6)

    noise_type = cfg.get("noise_type", "gaussian")
    noise_level = cfg.get("noise_level", 0.3)
    noise_levels = cfg.get("noise_levels", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))
    print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")
    print(f"Architecture : {layer_dims}")
    print(f"Train noise  : {noise_type} @ {noise_level}")

    ae = DenoisingAutoencoder(
        layer_dims=layer_dims,
        activation=activation,
        seed=seed,
        noise_type=noise_type,
        noise_level=noise_level,
    )

    losses = ae.train_and_collect(
        X, epochs, lr, batch_size, log_every, optimizer,
        patience=patience, min_delta=min_delta,
    )
    loss_path = _plot_loss(losses, out_dir)
    print(f"\nBest loss  : {min(losses):.6f}  (restored model)")
    print(f"Loss plot  -> {loss_path}")

    run_denoising_study(
        ae=ae,
        X=X,
        labels=list(labels),
        noise_levels=noise_levels,
        noise_type=noise_type,
        out_dir=out_dir,
        seed=seed,
        threshold=threshold,
        max_errors=max_errors,
    )


if __name__ == "__main__":
    main()
