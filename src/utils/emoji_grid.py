import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import plot_style
from emoji_loader import load_emojis

EMOJIS_H = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "emojis.h")
)


def _short_label(label):
    """Drop the 'U+XXXX ' prefix and wrap long names for the title."""
    name = label.split("  ", 1)[-1]
    words = name.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 16:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def render_grid(emojis_h=EMOJIS_H, out_path=None, ncols=5):
    bm_color, bm_bw, labels = load_emojis(emojis_h)[2:]
    n = len(labels)
    cells_per_row = ncols
    nrows = (n + cells_per_row - 1) // cells_per_row

    fig, axes = plt.subplots(
        nrows, cells_per_row * 2,
        figsize=(cells_per_row * 2 * 1.4, nrows * 1.8),
    )
    axes = np.atleast_2d(axes)

    for ax in axes.ravel():
        ax.axis("off")

    for i in range(n):
        row = i // cells_per_row
        col = (i % cells_per_row) * 2
        ax_c = axes[row, col]
        ax_bw = axes[row, col + 1]

        ax_c.imshow(bm_color[i], interpolation="nearest")
        ax_c.set_title(_short_label(labels[i]), fontsize=6, pad=2)
        ax_c.axis("off")

        ax_bw.imshow(1.0 - bm_bw[i], cmap="gray", vmin=0, vmax=1,
                     interpolation="nearest")
        ax_bw.set_title("b&w", fontsize=6, pad=2)
        ax_bw.axis("off")

    fig.suptitle(
        f"Emoji dataset ({n} emojis, 20x20)  -  color | binary",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is None:
        out_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "output", "emoji_grid.png")
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote grid to {out_path}")
    return out_path


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    render_grid(out_path=out)


if __name__ == "__main__":
    main()
