"""compare_denoising.py
Compare denoising robustness across noise types (gaussian / salt_pepper /
masking), aggregated over several seeds (mean ± std). Trains one DAE per
(noise type, seed) under the same budget and overlays the reconstruction-error
-vs-noise curves with their dispersion band.

Seeds are taken from the config's ``"seeds"`` list (fallback: ``--seeds`` count
[1..N]); ``--workers`` controls how many runs execute simultaneously.

Usage:
    python scripts/compare_denoising.py --config configs/default_denoising.json \
        --workers 8
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import csv
import os

import numpy as np

from utils import multiseed
from utils.config_loader import load_config
from utils.font_loader import load_font
from utils.noise import NOISE_TYPES


def main():
    parser = argparse.ArgumentParser(description="Denoising noise-type comparison")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Fallback seed count [1..N] when the config has no "
                             "\"seeds\" list.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Simultaneous runs (1 = sequential).")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    out_dir = os.path.join(resolve(cfg.get("out", "output/denoising")), "noise_compare")
    os.makedirs(out_dir, exist_ok=True)

    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))
    levels = cfg.get("noise_levels", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    seed_list = cfg.get("seeds") or list(range(1, args.seeds + 1))

    base = {
        "layer_dims": cfg["layer_dims"], "activation": cfg["activation"],
        "optimizer": cfg["optimizer"], "epochs": cfg["epochs"], "lr": cfg["lr"],
        "batch_size": cfg["batch_size"], "patience": cfg.get("patience"),
        "min_delta": cfg.get("min_delta", 1e-6), "threshold": cfg.get("threshold", 0.5),
        "max_errors": cfg.get("max_errors", 1), "noise_level": cfg.get("noise_level", 0.2),
    }

    print(f"Loaded {len(X)} characters | noise types={list(NOISE_TYPES)}")
    print(f"Seeds={seed_list} | workers={args.workers}")

    grouped = multiseed.run_noise_type_seeds(
        base, X, levels, list(NOISE_TYPES), seed_list, workers=args.workers)

    curve_path = multiseed.plot_noise_compare_bands(
        grouped, os.path.join(out_dir, "denoising_noise_comparison.png"),
        title=f"Denoising by noise type — media ± desv. ({len(seed_list)} seeds)")

    # CSV with mean ± std per (noise_type, level).
    csv_path = os.path.join(out_dir, "denoising_noise_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["noise_type", "noise_level", "n_seeds",
                         "recon_err_mean", "recon_err_std",
                         "noisy_err_mean", "noisy_err_std"])
        for ntype, per_seed in grouped.items():
            recon = np.array([d["recon"] for d in per_seed])   # (seeds, levels)
            noisy = np.array([d["noisy"] for d in per_seed])
            for j, lvl in enumerate(levels):
                writer.writerow([
                    ntype, lvl, len(per_seed),
                    round(float(recon[:, j].mean()), 4), round(float(recon[:, j].std()), 4),
                    round(float(noisy[:, j].mean()), 4), round(float(noisy[:, j].std()), 4),
                ])

    print(f"\nCurve -> {curve_path}")
    print(f"CSV   -> {csv_path}")


if __name__ == "__main__":
    main()
