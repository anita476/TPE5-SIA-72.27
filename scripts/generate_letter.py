"""generate_letter.py
Train the basic Autoencoder on font.h and generate new characters from its
2-D latent space, shown across several seeds so the behaviour is not tied to a
single run.

Usage:
    python scripts/generate_letter.py --config configs/default_simple.json \
        --from c --to e --seeds 6 --workers 6
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import os

from utils.config_loader import load_config
from utils.font_loader import load_font
from utils import multiseed


def main():
    parser = argparse.ArgumentParser(description="Generate letters from latent space")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--from", dest="src", default="c",
                        help="Source character for interpolation.")
    parser.add_argument("--to", dest="dst", default="e",
                        help="Target character for interpolation.")
    parser.add_argument("--seeds", type=int, default=6,
                        help="Fallback seed count [1..N] when the config has no "
                             "\"seeds\" list.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Simultaneous runs (1 = sequential).")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    out_dir = resolve(cfg.get("out", "output/simple"))
    os.makedirs(out_dir, exist_ok=True)

    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))
    labels = list(labels)
    if cfg["layer_dims"][len(cfg["layer_dims"]) // 2] != 2:
        raise ValueError("Generation requires a 2-D latent (bottleneck = 2).")

    base = {k: v for k, v in cfg.items()
            if k not in ("font", "out", "grid", "seeds")}
    seed_list = cfg.get("seeds") or list(range(1, args.seeds + 1))
    threshold = cfg.get("threshold", 0.5)
    print(f"Loaded {len(X)} characters | architecture {cfg['layer_dims']}")
    print(f"Seeds={seed_list} | workers={args.workers}")

    models = multiseed.run_model_seeds(base, X, seed_list, workers=args.workers)

    idx_a = labels.index(args.src) if args.src in labels else 0
    idx_b = labels.index(args.dst) if args.dst in labels else len(labels) - 1

    interp_path = multiseed.plot_interpolation_seeds(
        models, X, labels, idx_a, idx_b,
        os.path.join(out_dir, "latent_interpolation_seeds.png"),
        threshold=threshold)
    print(f"Interpolation (seeds) -> {interp_path}")

    point_path = multiseed.plot_generated_point_seeds(
        models, X, os.path.join(out_dir, "latent_generated_point_seeds.png"),
        threshold=threshold)
    print(f"Generated point (seeds) -> {point_path}")


if __name__ == "__main__":
    main()
