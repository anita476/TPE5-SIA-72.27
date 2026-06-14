"""test_autoencoder.py
Entry point for training and evaluating autoencoders on a bitmap font

Modes
-----
--config CONFIG_JSON
    Load all hyperparameters from a JSON file.  If the file contains a
    ``"grid"`` section, every combination is run (possibly in parallel via
    ``--workers``).

Output (per combination when using a grid)
------------------------------------------
<out_dir>/reconstruction_<id>.txt   — bitmap comparison
<out_dir>/autoencoder_results_<id>.png — loss curve + error bar chart
<out_dir>/grid_results.csv          — hyperparams + per-epoch MSE for all runs
"""

import argparse
import numpy as np

from utils.config_loader import load_config, expand_grid, resolve_autoencoder
from utils.font_loader import load_font
from utils.grid_runner import run_grid

# default hyperparameters
LAYER_DIMS  = [35, 24, 8, 2, 8, 24, 35]
ACTIVATION  = "tanh"
OPTIMIZER   = "adam"
SEED        = 42
EPOCHS      = 100_000
LR          = 0.0001
BATCH_SIZE  = 4
LOG_EVERY   = 1_000
THRESHOLD   = 0.5
MAX_ERRORS  = 1
AUTOENCODER = "simple"
OUT_DIR     = "results"
FONT_FILE   = "font.h"
WORKERS     = 1


def binarise(output: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Convert sigmoid outputs → binary {0, 1}."""
    return (output >= threshold).astype(np.float32)


def pixel_errors(pred_bin: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Return number of wrong pixels per sample."""
    return np.sum(pred_bin != y_true, axis=1)


def format_bitmap_side_by_side(
    original:      np.ndarray,
    reconstructed: np.ndarray,
    label:         str,
    errors:        int,
    rows:          int = 7,
    cols:          int = 5,
) -> str:
    """Return a printable string showing original and reconstructed bitmaps."""
    orig_grid = original.reshape(rows, cols)
    rec_grid  = reconstructed.reshape(rows, cols)
    lines = [f"[ {label} ]  —  {errors} pixel error(s)"]
    lines.append(f"  {'Original':<13}  {'Reconstructed'}")
    lines.append(f"  {'-'*11}  {'-'*13}")
    for r in range(rows):
        orig_row = " ".join("█" if p else "·" for p in orig_grid[r])
        rec_row  = " ".join("█" if p else "·" for p in rec_grid[r])
        diff     = "  ✗" if not np.array_equal(orig_grid[r], rec_grid[r]) else ""
        lines.append(f"  {orig_row}   {rec_row}{diff}")
    lines.append("")
    return "\n".join(lines)


def run_test(
    font_path:        str,
    autoencoder_type: str   = AUTOENCODER,
    layer_dims:       list  = LAYER_DIMS,
    activation:       str   = ACTIVATION,
    optimizer:        str   = OPTIMIZER,
    seed:             int   = SEED,
    epochs:           int   = EPOCHS,
    lr:               float = LR,
    batch_size:       int   = BATCH_SIZE,
    log_every:        int   = LOG_EVERY,
    threshold:        float = THRESHOLD,
    max_errors:       int   = MAX_ERRORS,
    out_dir:          str   = OUT_DIR,
    workers:          int   = WORKERS,
) -> tuple:
    """Train and evaluate one (or more, if a grid) autoencoder configuration.

    Returns
    -------
    tuple
        ``(passed, failed, errors)`` for the single run.
    """
    X, _, labels = load_font(font_path)
    print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")

    params = dict(
        autoencoder_type = autoencoder_type,
        layer_dims       = layer_dims,
        activation       = activation,
        optimizer        = optimizer,
        seed             = seed,
        epochs           = epochs,
        lr               = lr,
        batch_size       = batch_size,
        log_every        = log_every,
        threshold        = threshold,
        max_errors       = max_errors,
    )

    results = run_grid(
        combinations = [params],
        X            = X,
        labels       = list(labels),
        out_dir      = out_dir,
        workers      = workers,
    )

    r = results[0]
    return r.passed, r.failed, r.per_char_errors



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate autoencoders on a bitmap font file.  "
            "Use --config to load hyperparameters (and optional grid search) "
            "from JSON, or pass individual flags for a single run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        metavar="CONFIG_JSON",
        default=None,
        help=(
            "Path to a JSON config file.  When provided, ALL hyperparameters "
            "are read from it (the individual flags below are ignored).  "
            "If the file contains a \"grid\" section, all combinations are run."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=(
            "Maximum number of parallel worker processes for grid search.  "
            "Use 1 for sequential execution."
        ),
    )

    parser.add_argument("--font",        default=FONT_FILE,
                        help="Path to the font bitmap file.")
    parser.add_argument("--autoencoder", default=AUTOENCODER,
                        dest="autoencoder_type",
                        help="Autoencoder architecture to use (e.g. 'simple').")
    parser.add_argument("--threshold",   type=float, default=THRESHOLD,
                        help="Binarisation threshold.")
    parser.add_argument(
        "--out",
        default=OUT_DIR,
        dest="out_dir",
        help=(
            "Directory where output files are written "
            "(reconstruction_*.txt, autoencoder_results_*.png, grid_results.csv). "
            "Created automatically if it does not exist."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if args.config is not None:
        print(f"Loading config from {args.config!r} …")
        cfg          = load_config(args.config)
        combinations = expand_grid(cfg)

        print(f"Expanded grid → {len(combinations)} combination(s)")

        X, _, labels = load_font(cfg.get("font", FONT_FILE))
        print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")

        run_grid(
            combinations = combinations,
            X            = X,
            labels       = list(labels),
            out_dir      = cfg.get("out", OUT_DIR),
            workers      = args.workers,
        )

    else:
        run_test(
            font_path        = args.font,
            autoencoder_type = args.autoencoder_type,
            threshold        = args.threshold,
            out_dir          = args.out_dir,
            workers          = args.workers,
        )