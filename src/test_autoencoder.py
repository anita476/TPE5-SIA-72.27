import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.config_loader import load_config, resolve_autoencoder
from utils.font_loader import load_font

# DEFAULT HYPERPARAMETERS
LAYER_DIMS  = [35, 24, 8, 2, 8, 24, 35]
ACTIVATION  = "tanh"
OPTIMIZER   = "adam"
SEED        = 42
EPOCHS      = 100_000
LR          = 0.0001
BATCH_SIZE  = 4
LOG_EVERY   = 1_000
THRESHOLD   = 0.5          # binarisation threshold
MAX_ERRORS  = 1
AUTOENCODER = "simple"
OUT_DIR     = "results"    # default output directory
FONT_FILE = "font.h"



def binarise(output: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Convert sigmoid outputs → binary {0, 1}."""
    return (output >= threshold).astype(np.float32)


def pixel_errors(pred_bin: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Return number of wrong pixels for each sample."""
    return np.sum(pred_bin != y_true, axis=1)


def format_bitmap_side_by_side(
    original: np.ndarray,
    reconstructed: np.ndarray,
    label: str,
    errors: int,
    rows: int = 7,
    cols: int = 5,
) -> str:
    """Return a string showing original and reconstructed bitmaps side by side."""
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
):
    # Ensure the output directory exists before writing anything.
    os.makedirs(out_dir, exist_ok=True)
    reconstruction_path = os.path.join(out_dir, "reconstruction.txt")
    plot_path           = os.path.join(out_dir, "autoencoder_results.png")
    print(f"Output directory : {os.path.abspath(out_dir)}")

    X, bitmaps, labels = load_font(font_path)
    print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")

    AEClass = resolve_autoencoder(autoencoder_type)
    ae = AEClass(layer_dims, activation, seed)

    print(f"\nAutoencoder  : {autoencoder_type} ({AEClass.__name__})")
    print(f"Architecture : {layer_dims}")
    print(f"Activation   : {activation}")
    print(f"Training for {epochs} epochs …\n")
    ae.train(X, epochs=epochs, lr=lr, batch_size=batch_size,
             log_every=log_every, optimizer=optimizer)

    latent_all        = ae.encode(X)
    reconstructed_raw = ae.decode(latent_all)
    reconstructed_bin = binarise(reconstructed_raw, threshold=threshold)

    errors = pixel_errors(reconstructed_bin, X)

    print("\n" + "="*55)
    print(f"{'Idx':>4}  {'Label':<20}  {'Errors':>6}  {'Pass':>5}")
    print("-"*55)
    for i, (label, err) in enumerate(zip(labels, errors)):
        status = "PASSED" if err <= max_errors else "FAILED"
        print(f"{i:>4}  {label:<20}  {err:>6}  {status:>5}")

    passed  = np.sum(errors <= max_errors)
    failed  = len(errors) - passed
    avg_err = errors.mean()
    max_err = errors.max()

    print("="*55)
    print(f"\nSummary")
    print(f"  Characters passed (≤{max_errors} wrong pixel) : {passed}/{len(X)}")
    print(f"  Characters failed                    : {failed}/{len(X)}")
    print(f"  Average pixel errors per character   : {avg_err:.2f}")
    print(f"  Worst-case pixel errors              : {max_err}")

    goal_met = (failed == 0)
    print(
        f"\n{f"GOAL MET, all characters reconstructed within tolerance of {max_errors}!" if goal_met else f"GOAL NOT MET, some characters exceed the error threshold of {max_errors}"}"
    )

    with open(reconstruction_path, "w", encoding="utf-8") as f:
        f.write("Autoencoder Reconstruction Report\n")
        f.write(f"Autoencoder  : {autoencoder_type} ({AEClass.__name__})\n")
        f.write(f"Architecture : {layer_dims}\n")
        f.write(f"Activation   : {activation}  |  Threshold : {threshold}\n")
        f.write(f"Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}\n")
        f.write(f"Passed: {passed}/{len(X)}  |  Avg errors: {avg_err:.2f}  |  Max errors: {max_err}\n")
        f.write("=" * 50 + "\n\n")
        for i, (label, err) in enumerate(zip(labels, errors)):
            block = format_bitmap_side_by_side(X[i], reconstructed_bin[i], label, err)
            f.write(block + "\n")
    print(f"Character comparison saved to {reconstruction_path}")

    latent = latent_all
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(latent[:, 0], latent[:, 1],
               c=np.arange(len(X)), cmap="tab20", s=80, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (latent[i, 0], latent[i, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_title("2-D Latent Space")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    colours = ["green" if e <= max_errors else "red" for e in errors]
    ax2.bar(range(len(errors)), errors, color=colours, edgecolor="black", linewidth=0.5)
    ax2.axhline(max_errors, color="red", linestyle="--",
                label=f"Threshold ({max_errors})")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_ylabel("Pixel errors")
    ax2.set_title("Reconstruction errors per character")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.show()

    return passed, failed, errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test an autoencoder on a font.h bitmap file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        metavar="CONFIG_JSON",
        default=None,
        help=(
            "Path to a JSON config file. When provided, ALL hyperparameters "
            "are read from it and the commandline flags are ignored"
        ),
    )

    parser.add_argument("--font",      default="font.h",
                        help="Path to the font bitmap file.")
    parser.add_argument("--autoencoder", default=AUTOENCODER,
                        dest="autoencoder_type",
                        help="Autoencoder architecture to use (e.g. 'simple').")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Binarisation threshold.")
    parser.add_argument("--out",       default=OUT_DIR,
                        dest="out_dir",
                        help=(
                            "Directory where output files are written "
                            "(reconstruction.txt and autoencoder_results.png). "
                            "Created automatically if it does not exist."
                        ))

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if args.config is not None:
        print(f"Loading config from {args.config!r} …")
        cfg = load_config(args.config)

        run_test(
            font_path        = cfg.get("font",             FONT_FILE),
            autoencoder_type = cfg.get("autoencoder_type", AUTOENCODER),
            layer_dims       = cfg.get("layer_dims",       LAYER_DIMS),
            activation       = cfg.get("activation",       ACTIVATION),
            optimizer        = cfg.get("optimizer",        OPTIMIZER),
            seed             = cfg.get("seed",             SEED),
            epochs           = cfg.get("epochs",           EPOCHS),
            lr               = cfg.get("lr",               LR),
            batch_size       = cfg.get("batch_size",       BATCH_SIZE),
            log_every        = cfg.get("log_every",        LOG_EVERY),
            threshold        = cfg.get("threshold",        THRESHOLD),
            max_errors       = cfg.get("max_errors",       MAX_ERRORS),
            out_dir          = cfg.get("out",              OUT_DIR),
        )
    else:
        run_test(
            font_path        = args.font,
            autoencoder_type = args.autoencoder_type,
            threshold        = args.threshold,
            out_dir          = args.out_dir,
        )