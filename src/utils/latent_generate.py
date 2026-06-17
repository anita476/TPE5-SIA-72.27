from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROWS, COLS = 7, 5


def _grid(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(ROWS, COLS)


def latent_grid(ae, X, labels, out_dir, n=12, binarise=True, threshold=0.5):
    """Decode a regular grid of latent points covering the data's latent range."""
    z = ae.encode(X)
    z1 = np.linspace(z[:, 0].min(), z[:, 0].max(), n)
    z2 = np.linspace(z[:, 1].max(), z[:, 1].min(), n)

    fig, axes = plt.subplots(n, n, figsize=(n * 0.6, n * 0.6))
    for i, b in enumerate(z2):
        for j, a in enumerate(z1):
            point = np.array([[a, b]], dtype=np.float32)
            out = ae.decode(point)[0]
            if binarise:
                out = (out >= threshold).astype(np.float32)
            ax = axes[i, j]
            ax.imshow(_grid(out), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Latent grid (decoded) - generated characters", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "latent_grid.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def interpolate(ae, X, labels, idx_a, idx_b, out_dir, steps=9, threshold=0.5):
    """Decode points along the segment between two letters' latent codes.

    The intermediate points produce new characters absent from the dataset.
    """
    z = ae.encode(X)
    za, zb = z[idx_a], z[idx_b]
    alphas = np.linspace(0.0, 1.0, steps)

    fig, axes = plt.subplots(2, steps, figsize=(steps * 1.1, 2.6))
    for k, alpha in enumerate(alphas):
        point = ((1 - alpha) * za + alpha * zb).reshape(1, -1).astype(np.float32)
        raw = ae.decode(point)[0]
        binary = (raw >= threshold).astype(np.float32)
        axes[0, k].imshow(_grid(raw), cmap="gray_r", vmin=0, vmax=1)
        axes[1, k].imshow(_grid(binary), cmap="gray_r", vmin=0, vmax=1)
        for r in range(2):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
        axes[0, k].set_title(f"{alpha:.2f}", fontsize=8)

    axes[0, 0].set_ylabel("raw", fontsize=9)
    axes[1, 0].set_ylabel("binary", fontsize=9)
    fig.suptitle(
        f"Interpolation '{labels[idx_a]}' -> '{labels[idx_b]}' (new letters)",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "latent_interpolation.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def latent_scatter_with_point(ae, X, labels, point, out_dir, threshold=0.5):
    """Scatter the dataset in latent space and mark a generated point + glyph."""
    z = ae.encode(X)
    point = np.asarray(point, dtype=np.float32).reshape(1, -1)
    raw = ae.decode(point)[0]
    binary = (raw >= threshold).astype(np.float32)

    fig, (ax, ax_g) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax.scatter(z[:, 0], z[:, 1], c=np.arange(len(X)), cmap="tab20", s=80, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (z[i, 0], z[i, 1]), fontsize=7, ha="center",
                    va="bottom", xytext=(0, 5), textcoords="offset points")
    ax.scatter(point[0, 0], point[0, 1], marker="*", s=320, color="red",
               edgecolor="black", zorder=5, label="generated")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Latent space + generated point")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_g.imshow(_grid(binary), cmap="gray_r", vmin=0, vmax=1)
    ax_g.set_title("Generated glyph")
    ax_g.set_xticks([])
    ax_g.set_yticks([])

    plt.tight_layout()
    path = os.path.join(out_dir, "latent_generated_point.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path
