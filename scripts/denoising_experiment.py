"""denoising_experiment.py
Train a Denoising Autoencoder and study its noise-removal capability.

By default the noise-vs-error study is aggregated over several seeds
(mean ± std) so the conclusions rest on a good sample. A single representative
seed is also trained to produce the qualitative loss curve and
Clean/Noisy/Reconstructed example images. The per-noise-type comparison lives
in its own script (``compare_denoising.py``).

Usage:
    python scripts/denoising_experiment.py --config configs/default_denoising.json \
        --workers 8

Config keys: seeds (list), noise_type, noise_level (training),
noise_levels (eval list).
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import plot_style
from utils import multiseed
from autoencoders.DenoisingAutoencoder import DenoisingAutoencoder
from utils.config_loader import load_config
from utils.denoising_eval import run_denoising_study
from utils.font_loader import load_font


def _plot_loss(losses, out_dir, loss_name="mse"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(losses, linewidth=0.8, color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{loss_name.upper()} (vs clean input)")
    ax.set_title("Denoising Autoencoder training (seed representativa)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder experiment")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Fallback seed count [1..N] when the config has no "
                             "\"seeds\" list.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Simultaneous runs (1 = sequential).")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    out_dir = resolve(cfg.get("out", "output/denoising"))
    os.makedirs(out_dir, exist_ok=True)

    base = {
        "layer_dims": cfg["layer_dims"], "activation": cfg["activation"],
        "optimizer": cfg["optimizer"], "epochs": cfg["epochs"], "lr": cfg["lr"],
        "batch_size": cfg["batch_size"], "patience": cfg.get("patience"),
        "min_delta": cfg.get("min_delta", 1e-6), "threshold": cfg.get("threshold", 0.5),
        "max_errors": cfg.get("max_errors", 1), "noise_level": cfg.get("noise_level", 0.3),
        "loss": cfg.get("loss", "mse"),
    }
    noise_type = cfg.get("noise_type", "gaussian")
    noise_levels = cfg.get("noise_levels", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    seed_list = cfg.get("seeds") or list(range(1, args.seeds + 1))

    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))
    print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")
    print(f"Architecture : {base['layer_dims']}")
    print(f"Train noise  : {noise_type} @ {base['noise_level']}")
    print(f"Seeds={seed_list} | workers={args.workers}")

    # --- Qualitative: one representative seed for loss + example images -------
    # (run first: run_denoising_study writes its own single-seed curve which the
    # aggregated band below intentionally overwrites as the canonical figure.)
    ae = DenoisingAutoencoder(
        base["layer_dims"], base["activation"], seed_list[0],
        noise_type=noise_type, noise_level=base["noise_level"],
        loss=base["loss"])
    losses = ae.train_and_collect(
        X, base["epochs"], base["lr"], base["batch_size"], 0, base["optimizer"],
        patience=base["patience"], min_delta=base["min_delta"])
    loss_path = _plot_loss(losses, out_dir, base["loss"])
    print(f"Loss plot      -> {loss_path}")
    run_denoising_study(
        ae=ae, X=X, labels=list(labels), noise_levels=noise_levels,
        noise_type=noise_type, out_dir=out_dir, seed=seed_list[0],
        threshold=base["threshold"], max_errors=base["max_errors"])

    # --- Quantitative: denoising-vs-noise aggregated over seeds (default) -----
    per_seed = multiseed.run_denoising_seeds(
        base, X, noise_levels, noise_type, seed_list, workers=args.workers)
    band_path = multiseed.plot_denoising_band(
        per_seed, os.path.join(out_dir, "denoising_vs_noise.png"),
        title=f"Denoising capability — media ± desv. ({len(seed_list)} seeds)")
    print(f"\nDenoising band -> {band_path}")
    print("(noise-type comparison: run scripts/compare_denoising.py)")


if __name__ == "__main__":
    main()
