import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from fashion_mnist_loader import load_fashion_mnist, LABEL_NAMES, ROWS, COLS


def render_grid(n_per_class=2, out_path=None, seed=0):
    """Show Fashion-MNIST samples in rows: two per class."""
    X, labels, _ = load_fashion_mnist(n_samples=10000, seed=seed)

    n_classes = len(LABEL_NAMES)
    fig, axes = plt.subplots(
        n_per_class, n_classes,
        figsize=(n_classes * 1.4, n_per_class * 1.8),
    )

    for ax in axes.ravel():
        ax.axis("off")

    for c in range(n_classes):
        idxs = np.where(labels == c)[0][:n_per_class]
        for j, idx in enumerate(idxs):
            img = X[idx].reshape(ROWS, COLS)
            axes[j, c].imshow(1.0 - img, cmap="gray", vmin=0, vmax=1,
                              interpolation="bilinear")
        axes[0, c].set_title(LABEL_NAMES[c], fontsize=10, pad=8)

    fig.suptitle(
        f"Fashion-MNIST dataset ({n_classes} classes, {ROWS}x{COLS})",
        fontsize=13,
    )
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=3, h_pad=2)

    if out_path is None:
        out_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "output", "fashion_mnist_grid.png")
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Wrote grid to {out_path}")
    return out_path


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    render_grid(out_path=out)



if __name__ == "__main__":
    main()
