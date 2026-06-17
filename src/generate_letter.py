"""Train the basic Autoencoder on font.h and generate new characters from its
2-D latent space.

Usage:
    python generate_letter.py --config ../configs/default_simple.json

Outputs (in "out"):
    latent_grid.png            - decoded grid sweeping the latent plane
    latent_interpolation.png   - new letters between two known letters
    latent_generated_point.png - latent scatter + one generated glyph
"""
from __future__ import annotations

import argparse
import os

from autoencoders.SimpleAutoencoder import SimpleAutoencoder
from utils.config_loader import load_config
from utils.font_loader import load_font
from utils.latent_generate import (
    interpolate,
    latent_grid,
    latent_scatter_with_point,
)


def main():
    parser = argparse.ArgumentParser(description="Generate letters from latent space")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--from", dest="src", default="c",
                        help="Source character for interpolation.")
    parser.add_argument("--to", dest="dst", default="e",
                        help="Target character for interpolation.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("out", "output/simple")
    os.makedirs(out_dir, exist_ok=True)

    X, _, labels = load_font(cfg.get("font", "../data/font.h"))
    if cfg["layer_dims"][len(cfg["layer_dims"]) // 2] != 2:
        raise ValueError("Generation requires a 2-D latent (bottleneck = 2).")

    print(f"Loaded {len(X)} characters  |  architecture {cfg['layer_dims']}")

    ae = SimpleAutoencoder(cfg["layer_dims"], cfg["activation"], cfg["seed"])
    ae.train_and_collect(
        X, cfg["epochs"], cfg["lr"], cfg["batch_size"],
        cfg.get("log_every", 0), cfg["optimizer"],
    )

    threshold = cfg.get("threshold", 0.5)

    grid_path = latent_grid(ae, X, out_dir, threshold=threshold)
    print(f"Latent grid          -> {grid_path}")

    # Interpolate between two known letters; midpoints are unseen characters.
    idx_a = labels.index(args.src) if args.src in labels else 0
    idx_b = labels.index(args.dst) if args.dst in labels else len(labels) - 1
    interp_path = interpolate(ae, X, labels, idx_a, idx_b, out_dir,
                              threshold=threshold)
    print(f"Interpolation        -> {interp_path}")

    # Generate one glyph from the latent centroid (a point with no real letter).
    z = ae.encode(X)
    centroid = z.mean(axis=0)
    point_path = latent_scatter_with_point(ae, X, labels, centroid, out_dir,
                                           threshold=threshold)
    print(f"Generated point      -> {point_path}")


if __name__ == "__main__":
    main()
