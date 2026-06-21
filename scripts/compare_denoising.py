"""compare_denoising.py
Comparison studies for the Denoising Autoencoder, aggregated over several seeds
(mean ± std). The study is chosen by the config (checked in this order):

* ``"cross_robustness": true`` → train on each noise type, test on every type
  (3x3 heatmap of reconstruction error). Shows how training noise generalises.
* ``"qualitative": true`` → clean / noisy / reconstructed panel for the same
  characters across all noise types (intuition). Optional ``"qual_chars"`` and
  ``"qual_level"``.
* ``"train_levels": [..]`` → one figure per noise type comparing the effect of
  the training-noise level.
* ``"arch_variants": [{name, layer_dims}, ..]`` → compare bottleneck widths
  under a fixed noise type (justifies the chosen architecture).
* otherwise (default) → noise-type comparison: error-vs-noise curve per type,
  plus the "fair" plots (error vs actual damage, fraction of noise removed).

Seeds come from the config's ``"seeds"`` list (fallback ``--seeds`` → [1..N]);
``--workers`` controls how many runs execute simultaneously.

Usage:
    python scripts/compare_denoising.py --config configs/default_denoising.json    --workers 8
    python scripts/compare_denoising.py --config configs/denoising_arch.json        --workers 8
    python scripts/compare_denoising.py --config configs/denoising_trainlevel.json  --workers 8
    python scripts/compare_denoising.py --config configs/denoising_qualitative.json --workers 8
    python scripts/compare_denoising.py --config configs/denoising_crossrobust.json --workers 8
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


def _write_csv(grouped, levels, label_col, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([label_col, "noise_level", "n_seeds",
                         "recon_err_mean", "recon_err_std",
                         "noisy_err_mean", "noisy_err_std"])
        for label, per_seed in grouped.items():
            recon = np.array([d["recon"] for d in per_seed])   # (seeds, levels)
            noisy = np.array([d["noisy"] for d in per_seed])
            for j, lvl in enumerate(levels):
                writer.writerow([
                    label, lvl, len(per_seed),
                    round(float(recon[:, j].mean()), 4), round(float(recon[:, j].std()), 4),
                    round(float(noisy[:, j].mean()), 4), round(float(noisy[:, j].std()), 4),
                ])


def main():
    parser = argparse.ArgumentParser(description="Denoising comparison study")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Fallback seed count [1..N] when the config has no "
                             "\"seeds\" list.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Simultaneous runs (1 = sequential).")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))
    levels = cfg.get("noise_levels", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    seed_list = cfg.get("seeds") or list(range(1, args.seeds + 1))

    base = {
        "layer_dims": cfg["layer_dims"], "activation": cfg["activation"],
        "optimizer": cfg["optimizer"], "epochs": cfg["epochs"], "lr": cfg["lr"],
        "batch_size": cfg["batch_size"], "patience": cfg.get("patience"),
        "min_delta": cfg.get("min_delta", 1e-6), "threshold": cfg.get("threshold", 0.5),
        "max_errors": cfg.get("max_errors", 1), "noise_level": cfg.get("noise_level", 0.2),
        "loss": cfg.get("loss", "mse"),
    }

    arch_variants = cfg.get("arch_variants")
    train_levels = cfg.get("train_levels")
    out_root = resolve(cfg.get("out", "output/denoising"))
    out_nc = os.path.join(out_root, "noise_compare")

    if cfg.get("cross_robustness"):
        # --- Train on type A, test on type B (3x3 heatmap) -------------------
        os.makedirs(out_nc, exist_ok=True)
        level = cfg.get("noise_level", 0.2)
        print(f"Cross-robustness | test@{level} | seeds={seed_list} "
              f"| workers={args.workers}")
        matrix = multiseed.run_cross_robustness(
            base, X, list(NOISE_TYPES), level, seed_list, workers=args.workers)
        path = multiseed.plot_cross_robustness(
            matrix, list(NOISE_TYPES), level,
            os.path.join(out_nc, "cross_robustness.png"))
        print(f"\nHeatmap -> {path}")
        return

    if cfg.get("qualitative"):
        # --- Side-by-side clean/noisy/recon for every noise type -------------
        os.makedirs(out_nc, exist_ok=True)
        level = cfg.get("qual_level", cfg.get("noise_level", 0.2))
        rep_seed = seed_list[0]
        wanted = cfg.get("qual_chars", ["a", "e", "g", "m", "s", "w"])
        char_idx = [labels.index(c) for c in wanted if c in labels] \
            or list(range(min(6, len(X))))
        print(f"Qualitative panel | level={level} | seed={rep_seed} "
              f"| workers={args.workers}")
        models = multiseed.run_denoise_models(
            base, X, list(NOISE_TYPES), rep_seed, workers=args.workers)
        path = multiseed.plot_noise_qualitative(
            models, X, labels, level, char_idx,
            os.path.join(out_nc, "noise_qualitative.png"),
            noise_seed=rep_seed, threshold=base["threshold"])
        print(f"\nPanel -> {path}")
        return

    if train_levels:
        # --- Training-noise-level comparison, one figure per noise type ------
        out_dir = os.path.join(out_root, "trainlevel_compare")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Train-level comparison | levels={train_levels} | "
              f"types={list(NOISE_TYPES)} | seeds={seed_list} | workers={args.workers}")

        for noise_type in NOISE_TYPES:
            grouped = {}
            for tl in train_levels:
                b = dict(base)
                b["noise_level"] = tl
                print(f"  {noise_type} | train@{tl}")
                grouped[f"train@{tl}"] = multiseed.run_denoising_seeds(
                    b, X, levels, noise_type, seed_list, workers=args.workers)
            curve_path = multiseed.plot_noise_compare_bands(
                grouped, os.path.join(out_dir, f"trainlevel_{noise_type}.png"),
                title=f"{noise_type}: efecto del ruido de entrenamiento — "
                      f"media ± desv. ({len(seed_list)} seeds)")
            csv_path = os.path.join(out_dir, f"trainlevel_{noise_type}.csv")
            _write_csv(grouped, levels, "train_level", csv_path)
            print(f"  -> {curve_path}")
        print("\nDone (trainlevel_compare/).")
        return

    if arch_variants:
        # --- Architecture comparison (fixed noise type) ----------------------
        noise_type = cfg.get("noise_type", "gaussian")
        out_dir = os.path.join(out_root, "arch_compare")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Arch comparison | noise={noise_type} | seeds={seed_list} "
              f"| workers={args.workers}")

        grouped = {}
        for v in arch_variants:
            name = v["name"]
            b = dict(base)
            b["layer_dims"] = v["layer_dims"]
            print(f"  -> {name}: {v['layer_dims']}")
            grouped[name] = multiseed.run_denoising_seeds(
                b, X, levels, noise_type, seed_list, workers=args.workers)

        curve_path = multiseed.plot_noise_compare_bands(
            grouped, os.path.join(out_dir, "denoising_arch_comparison.png"),
            title=f"Denoising por arquitectura ({noise_type}) — media ± desv. "
                  f"({len(seed_list)} seeds)")
        csv_path = os.path.join(out_dir, "denoising_arch_comparison.csv")
        _write_csv(grouped, levels, "architecture", csv_path)
    else:
        # --- Noise-type comparison (default) ---------------------------------
        out_dir = out_nc
        os.makedirs(out_dir, exist_ok=True)
        print(f"Noise-type comparison | types={list(NOISE_TYPES)} "
              f"| seeds={seed_list} | workers={args.workers}")

        grouped = multiseed.run_noise_type_seeds(
            base, X, levels, list(NOISE_TYPES), seed_list, workers=args.workers)
        curve_path = multiseed.plot_noise_compare_bands(
            grouped, os.path.join(out_dir, "denoising_noise_comparison.png"),
            title=f"Denoising by noise type — media ± desv. ({len(seed_list)} seeds)")
        csv_path = os.path.join(out_dir, "denoising_noise_comparison.csv")
        _write_csv(grouped, levels, "noise_type", csv_path)
        # Fair comparison (normalises away the apples-to-oranges nominal level).
        multiseed.plot_recon_vs_actual(
            grouped, os.path.join(out_dir, "recon_vs_actual.png"))
        multiseed.plot_fraction_removed(
            grouped, levels, os.path.join(out_dir, "fraction_removed.png"))

    print(f"\nCurve -> {curve_path}")
    print(f"CSV   -> {csv_path}")


if __name__ == "__main__":
    main()
