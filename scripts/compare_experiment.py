"""compare_experiment.py
Compare autoencoder architectures / optimizers under the same budget.

Usage:
    python scripts/compare_experiment.py --config configs/compare_simple.json

The config holds common hyperparameters plus a "variants" list; each variant is
a dict with a "name" and any overrides (e.g. layer_dims, optimizer, lr).
"""

import _bootstrap
from _bootstrap import resolve

import argparse

from utils.comparison import run_comparison
from utils.config_loader import load_config
from utils.font_loader import load_font


def main():
    parser = argparse.ArgumentParser(description="Autoencoder comparison study")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    variants = cfg.get("variants")
    if not variants:
        raise ValueError("Config must define a non-empty 'variants' list.")

    out_dir = resolve(cfg.get("out", "output/compare"))
    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))

    base = {k: v for k, v in cfg.items()
            if k not in ("variants", "font", "out", "grid")}

    print(f"Loaded {len(X)} characters  |  {len(variants)} variant(s)")
    run_comparison(variants, base, X, list(labels), out_dir)


if __name__ == "__main__":
    main()
