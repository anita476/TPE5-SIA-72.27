import argparse
import numpy as np
import matplotlib.pyplot as plt
from autoencoders.SimpleAutoencoder import SimpleAutoencoder
from utils.font_loader import load_font

# hyperparameters - @todo config file
LAYER_DIMS  = [35, 24,8, 2, 8, 24,35]
ACTIVATION  = "tanh"
OPTMIZER = "adam"
SEED        = 42
EPOCHS      = 100_000
LR          = 0.0001
BATCH_SIZE  = 4
LOG_EVERY   = 1000
THRESHOLD   = 0.5                   # binarisation threshold
MAX_ERRORS  = 1


def binarise(output: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Convert sigmoid outputs → binary {0, 1}."""
    return (output >= threshold).astype(np.float32)


def pixel_errors(pred_bin: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Return number of wrong pixels for each sample."""
    return np.sum(pred_bin != y_true, axis=1)


def format_bitmap_side_by_side(original: np.ndarray, reconstructed: np.ndarray,
                               label: str, errors: int, rows=7, cols=5) -> str:
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


def run_test(font_path: str, threshold: float = THRESHOLD, out_path: str = "reconstruction.txt"):
    X, bitmaps, labels = load_font(font_path)
    print(f"Loaded {len(X)} characters  |  input dim = {X.shape[1]}")

    ae = SimpleAutoencoder(LAYER_DIMS, ACTIVATION, SEED)
    print(f"\nArchitecture : {LAYER_DIMS}")
    print(f"Activation   : {ACTIVATION}")
    print(f"Training for {EPOCHS} epochs …\n")
    ae.train(X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, log_every=LOG_EVERY, optimizer="adam")


    latent_all        = ae.encode(X)               # (n, 2) — reused for scatter plot
    reconstructed_raw = ae.decode(latent_all)      # continuous in (0, 1)
    reconstructed_bin = binarise(reconstructed_raw, threshold=threshold)

    errors = pixel_errors(reconstructed_bin, X)

    print("\n" + "="*55)
    print(f"{'Idx':>4}  {'Label':<20}  {'Errors':>6}  {'Pass':>5}")
    print("-"*55)
    for i, (label, err) in enumerate(zip(labels, errors)):
        status = "✓" if err <= MAX_ERRORS else "✗"
        print(f"{i:>4}  {label:<20}  {err:>6}  {status:>5}")

    passed   = np.sum(errors <= MAX_ERRORS)
    failed   = len(errors) - passed
    avg_err  = errors.mean()
    max_err  = errors.max()

    print("="*55)
    print(f"\nSummary")
    print(f"  Characters passed (≤{MAX_ERRORS} wrong pixel) : {passed}/{len(X)}")
    print(f"  Characters failed                    : {failed}/{len(X)}")
    print(f"  Average pixel errors per character   : {avg_err:.2f}")
    print(f"  Worst-case pixel errors              : {max_err}")

    goal_met = (failed == 0)
    print(f"\n{'GOAL MET — all characters reconstructed within tolerance!' if goal_met else 'GOAL NOT MET — some characters exceed the error threshold.'}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Autoencoder Reconstruction Report\n")
        f.write(f"Architecture : {LAYER_DIMS}\n")
        f.write(f"Activation   : {ACTIVATION}  |  Threshold : {threshold}\n")
        f.write(f"Epochs: {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}\n")
        f.write(f"Passed: {passed}/{len(X)}  |  Avg errors: {avg_err:.2f}  |  Max errors: {max_err}\n")
        f.write("=" * 50 + "\n\n")
        for i, (label, err) in enumerate(zip(labels, errors)):
            block = format_bitmap_side_by_side(X[i], reconstructed_bin[i], label, err)
            f.write(block + "\n")
    print(f"Character comparison saved to {out_path}")

    latent = latent_all
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    sc = ax.scatter(latent[:, 0], latent[:, 1], c=np.arange(len(X)), cmap="tab20", s=80, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (latent[i, 0], latent[i, 1]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.set_title("2-D Latent Space")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    colours = ["green" if e <= MAX_ERRORS else "red" for e in errors]
    ax2.bar(range(len(errors)), errors, color=colours, edgecolor="black", linewidth=0.5)
    ax2.axhline(MAX_ERRORS, color="red", linestyle="--", label=f"Threshold ({MAX_ERRORS})")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_ylabel("Pixel errors")
    ax2.set_title("Reconstruction errors per character")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("autoencoder_results.png", dpi=150)
    print("\nPlot saved to autoencoder_results.png")
    plt.show()


    return passed, failed, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SimpleAutoencoder on font.h")
    parser.add_argument("--font",      default="font.h",           help="Path to font.h")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Binarisation threshold (default: {THRESHOLD})")
    parser.add_argument("--out",       default="reconstruction.txt",
                        help="Output txt file for character comparison (default: reconstruction.txt)")
    args = parser.parse_args()
    run_test(args.font, threshold=args.threshold, out_path=args.out)