"""test_autoencoder.py
Entry point for training and evaluating autoencoders on a bitmap font.

Modes
-----
--config CONFIG_JSON
    Load all hyperparameters from a JSON file.  If the file contains a
    ``"grid"`` section, every combination is run (possibly in parallel via
    ``--workers``).
"""

import _bootstrap
from _bootstrap import resolve

import argparse

from utils.config_loader import load_config, expand_grid
from utils.font_loader import load_font
from utils.grid_runner import run_grid

# defaults for the no-config single run
ACTIVATION  = "tanh"
OPTIMIZER   = "adam"
SEED        = 42
THRESHOLD   = 0.5
MAX_ERRORS  = 1
AUTOENCODER = "simple"
LAYER_DIMS  = [35, 24, 8, 2, 8, 24, 35]
EPOCHS      = 100_000
LR          = 1e-4
BATCH_SIZE  = 4
LOG_EVERY   = 1_000
OUT_DIR     = "output"
FONT_FILE   = "data/font.h"
WORKERS     = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate autoencoders on a bitmap font file.  "
            "Use --config to load hyperparameters (and optional grid search) "
            "from JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", metavar="CONFIG_JSON", default=None,
        help="Path to a JSON config file. Supports an optional \"grid\" section.",
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS,
        help="Max parallel worker processes for grid search (1 = sequential).",
    )
    parser.add_argument("--font", default=FONT_FILE,
                        help="Path to the font bitmap file.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Binarisation threshold.")
    parser.add_argument("--out", default=OUT_DIR, dest="out_dir",
                        help="Directory where output files are written.")
    return parser


def main():
    args = _build_parser().parse_args()

    if args.config is not None:
        print(f"Loading config from {args.config!r} ...")
        cfg = load_config(resolve(args.config))
        combinations = expand_grid(cfg)
        print(f"Expanded grid -> {len(combinations)} combination(s)")

        X, _, labels = load_font(resolve(cfg.get("font", FONT_FILE)))
        out_dir = resolve(cfg.get("out", OUT_DIR))
    else:
        X, _, labels = load_font(resolve(args.font))
        combinations = [dict(
            autoencoder_type=AUTOENCODER, layer_dims=LAYER_DIMS,
            activation=ACTIVATION, optimizer=OPTIMIZER, seed=SEED,
            epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, log_every=LOG_EVERY,
            threshold=args.threshold, max_errors=MAX_ERRORS,
        )]
        out_dir = resolve(args.out_dir)

    print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")
    run_grid(
        combinations=combinations,
        X=X,
        labels=list(labels),
        out_dir=out_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
