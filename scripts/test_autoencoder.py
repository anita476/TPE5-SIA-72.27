"""test_autoencoder.py
Entry point for training and evaluating the basic autoencoder on a bitmap font.

By default the experiment is run over **several random seeds** and the plots
report mean ± std, so any claim rests on a good sample rather than a single
lucky/unlucky run. Use ``--seeds`` to choose how many seeds and ``--workers``
to choose how many runs execute simultaneously (kept small by default so it is
portable to machines with few CPUs).

Modes
-----
--config CONFIG_JSON
    Load all hyperparameters from a JSON file. If the file contains a
    ``"grid"`` section with more than one combination, the classic
    hyperparameter grid search is run instead (one set of plots per run).
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import csv
import os

from utils.config_loader import load_config, expand_grid
from utils.font_loader import load_font
from utils.grid_runner import run_grid
from utils import multiseed

# defaults for the no-config single run
ACTIVATION  = "tanh"
OPTIMIZER   = "adam"
THRESHOLD   = 0.5
MAX_ERRORS  = 1
AUTOENCODER = "simple"
LAYER_DIMS  = [35, 30, 15, 2, 15, 30, 35]
EPOCHS      = 70_000
LR          = 1e-3
BATCH_SIZE  = 8
OUT_DIR     = "output/simple"
FONT_FILE   = "data/font.h"
WORKERS     = 1
SEEDS       = 10


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the basic autoencoder on a bitmap font, "
            "aggregated over several seeds (mean ± std) by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", metavar="CONFIG_JSON", default=None,
                        help="Path to a JSON config file. A multi-combination "
                             "\"grid\" section triggers grid search instead.")
    parser.add_argument("--seeds", type=int, default=SEEDS,
                        help="Fallback seed count [1..N] when the config has no "
                             "\"seeds\" list. The config's \"seeds\" takes priority.")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Simultaneous runs (1 = sequential). Choose to fit "
                             "the host's CPU count.")
    parser.add_argument("--latent-show", type=int, default=6,
                        help="How many seeds to display in the latent-space grid.")
    parser.add_argument("--font", default=FONT_FILE,
                        help="Path to the font bitmap file.")
    parser.add_argument("--out", default=None, dest="out_dir",
                        help="Output directory (defaults to the config's 'out').")
    return parser


def _write_summary_csv(results, out_dir):
    summary = multiseed.summarise(results)
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["seed", "passed", "avg_pixel_errors", "max_pixel_errors"])
        for r in results:
            w.writerow([r.params.get("seed"), r.passed,
                        round(r.avg_pixel_errors, 4), r.max_pixel_errors])
    return summary, path


def _run_multiseed(base, X, labels, out_dir, seed_list, workers, latent_show):
    max_errors = base.get("max_errors", MAX_ERRORS)
    n = len(seed_list)
    print(f"Multi-seed run: seeds={seed_list} | workers={workers}")

    results = multiseed.run_seeds(base, X, list(labels), seed_list, out_dir,
                                  workers=workers)

    loss_path = multiseed.plot_loss_band(
        results, os.path.join(out_dir, "loss.png"),
        title=f"Training Loss — media ± desv. ({n} seeds)")
    err_path = multiseed.plot_per_char_band(
        results, max_errors, os.path.join(out_dir, "errors.png"),
        title=f"Errores por carácter — media ± desv. ({n} seeds)")
    latent_path = multiseed.plot_latent_seeds(
        results, os.path.join(out_dir, "latent_seeds.png"), n_show=latent_show)
    summary, csv_path = _write_summary_csv(results, out_dir)

    print("\n=== Resumen multi-seed ===")
    print(f"  passed : {summary['passed_mean']:.1f} ± {summary['passed_std']:.1f}"
          f"  (min {summary['passed_min']:.0f}, max {summary['passed_max']:.0f})"
          f" / {summary['total_chars']}")
    print(f"  avg_err: {summary['avg_err_mean']:.3f} ± {summary['avg_err_std']:.3f}")
    print(f"\nLoss   -> {loss_path}")
    print(f"Errors -> {err_path}")
    print(f"Latent -> {latent_path}")
    print(f"CSV    -> {csv_path}")


def main():
    args = _build_parser().parse_args()

    if args.config is not None:
        print(f"Loading config from {args.config!r} ...")
        cfg = load_config(resolve(args.config))
        combinations = expand_grid(cfg)
        X, _, labels = load_font(resolve(cfg.get("font", FONT_FILE)))
        out_dir = resolve(args.out_dir or cfg.get("out", OUT_DIR))
        base = {k: v for k, v in cfg.items()
                if k not in ("font", "out", "grid", "seeds")}
        seed_list = cfg.get("seeds") or list(range(1, args.seeds + 1))
    else:
        X, _, labels = load_font(resolve(args.font))
        base = dict(autoencoder_type=AUTOENCODER, layer_dims=LAYER_DIMS,
                    activation=ACTIVATION, optimizer=OPTIMIZER, epochs=EPOCHS,
                    lr=LR, batch_size=BATCH_SIZE, threshold=THRESHOLD,
                    max_errors=MAX_ERRORS)
        combinations = [base]
        out_dir = resolve(args.out_dir or OUT_DIR)
        seed_list = list(range(1, args.seeds + 1))

    os.makedirs(out_dir, exist_ok=True)
    print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")

    # A genuine multi-combination grid -> classic hyperparameter search.
    if len(combinations) > 1:
        print(f"Grid with {len(combinations)} combinations -> grid search mode")
        run_grid(combinations=combinations, X=X, labels=list(labels),
                 out_dir=out_dir, workers=args.workers)
        return

    # Single combination -> multi-seed aggregated study (the default).
    _run_multiseed(base, X, labels, out_dir, seed_list, args.workers,
                   args.latent_show)


if __name__ == "__main__":
    main()
