"""olivetti_grid.py
Render a grid of all 400 Olivetti faces (40 subjects x 10 images each).

Usage:
    cd scripts && python ../src/utils/olivetti_grid.py [output_path]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from olivetti_loader import load_olivetti, ROWS, COLS


def render_grid(out_path=None, ncols=10):
    """Show all faces arranged as subjects (rows) x images (cols)."""
    X, subject_ids, _ = load_olivetti(seed=0)

    # Group by subject: dict {subject_id: [indices]}
    subjects = sorted(set(subject_ids))
    n_subjects = len(subjects)
    per_subject = {s: np.where(subject_ids == s)[0] for s in subjects}
    max_per_row = max(len(v) for v in per_subject.values())
    ncols = min(ncols, max_per_row)

    fig, axes = plt.subplots(n_subjects, ncols,
                             figsize=(ncols * 1.0, n_subjects * 1.1))
    for ax in axes.ravel():
        ax.axis("off")

    for row, subj in enumerate(subjects):
        idxs = per_subject[subj]
        for col in range(ncols):
            if col < len(idxs):
                img = X[idxs[col]].reshape(ROWS, COLS)
                axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1,
                                      interpolation="nearest")
            if col == 0:
                axes[row, col].set_ylabel(f"s{subj}", fontsize=6,
                                           rotation=0, labelpad=15, va="center")

    fig.suptitle(f"Olivetti Faces ({len(X)} images, {n_subjects} subjects, "
                 f"{ROWS}x{COLS})", fontsize=11)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))

    if out_path is None:
        out_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "output", "olivetti_grid.png")
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
