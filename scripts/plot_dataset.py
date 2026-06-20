"""plot_dataset.py
Render the 32 input characters from font.h as a labeled grid (slide: dataset).

Usage:
    python scripts/plot_dataset.py
"""

import _bootstrap
from _bootstrap import resolve

import os
import matplotlib.pyplot as plt

from utils.font_loader import load_font


def main():
    X, bitmaps, labels = load_font(resolve("data/font.h"))
    n = len(bitmaps)
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.6))
    fig.suptitle("Dataset: 32 caracteres de font.h (7x5)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(bitmaps[i], cmap="binary", vmin=0, vmax=1)
            ax.set_title(labels[i], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")

    out_dir = resolve("output/simple")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_grid.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Dataset grid -> {out_path}")


if __name__ == "__main__":
    main()
