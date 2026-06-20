"""compare_denoising.py
Compare denoising robustness across noise types (gaussian / salt_pepper /
masking). Trains one DAE per noise type (same budget) and overlays the
reconstruction-error-vs-noise curves.

Usage:
    python scripts/compare_denoising.py --config configs/default_denoising.json
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_style
from autoencoders.DenoisingAutoencoder import DenoisingAutoencoder
from utils.config_loader import load_config
from utils.denoising_eval import evaluate_at_level
from utils.font_loader import load_font
from utils.noise import NOISE_TYPES


def main():
    parser = argparse.ArgumentParser(description="Denoising noise-type comparison")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    out_dir = os.path.join(resolve(cfg.get("out", "output/denoising")), "noise_compare")
    os.makedirs(out_dir, exist_ok=True)

    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))
    levels = cfg.get("noise_levels", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    train_level = cfg.get("noise_level", 0.2)
    threshold = cfg.get("threshold", 0.5)
    max_errors = cfg.get("max_errors", 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for noise_type in NOISE_TYPES:
        print(f"\n=== noise type: {noise_type} ===")
        ae = DenoisingAutoencoder(
            cfg["layer_dims"], cfg["activation"], cfg["seed"],
            noise_type=noise_type, noise_level=train_level,
        )
        ae.train_and_collect(
            X, cfg["epochs"], cfg["lr"], cfg["batch_size"],
            cfg.get("log_every", 0), cfg["optimizer"],
            patience=cfg.get("patience"), min_delta=cfg.get("min_delta", 1e-6),
        )

        rng = np.random.default_rng(cfg["seed"])
        recon_err = []
        for lvl in levels:
            res = evaluate_at_level(ae, X, lvl, noise_type, rng, threshold, max_errors)
            recon_err.append(res["avg_err"])
            rows.append([noise_type, lvl, round(res["avg_noisy_err"], 4),
                         round(res["avg_err"], 4), res["passed"], len(X)])
            print(f"  level {lvl}: recon_err={res['avg_err']:.2f} "
                  f"passed={res['passed']}/{len(X)}")
        ax.plot(levels, recon_err, "o-", label=noise_type)

    ax.set_xlabel("Noise level")
    ax.set_ylabel("Mean recon pixel error (vs clean)")
    ax.set_title("Denoising by noise type (trained & tested per type)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    curve_path = os.path.join(out_dir, "denoising_noise_comparison.png")
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)

    csv_path = os.path.join(out_dir, "denoising_noise_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["noise_type", "noise_level", "avg_noisy_err",
                         "avg_recon_err", "passed", "total"])
        writer.writerows(rows)

    print(f"\nCurve -> {curve_path}")
    print(f"CSV   -> {csv_path}")


if __name__ == "__main__":
    main()
