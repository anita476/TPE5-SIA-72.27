"""seed_similarity.py
Structural-similarity study of the basic autoencoder's 2-D latent space across
random seeds.

For every seed the model is trained on font.h and the characters are embedded
in the 2-D latent space. We then compare *seeds* by how they arrange the
characters relative to one another — not by their absolute coordinates, which
are meaningless because the latent space can be rotated, reflected, translated
or rescaled without changing the model.

The comparison is a Representational Similarity Analysis (RSA):

1.  For each seed compute the matrix of pairwise distances between every pair
    of characters (the RDM). This matrix is invariant to rotation, reflection,
    translation and uniform scaling of the latent plane.
2.  For each pair of seeds correlate the upper triangles of their RDMs
    (Pearson). A high correlation means both seeds place the characters in the
    same relative geometry, regardless of how the plane is oriented.
3.  The mean correlation off the diagonal summarises how much structure is
    shared across independent runs (≈0 would mean the geometry is dictated by
    the random init; >0 means part of it is dictated by the data).

Usage:
    python scripts/seed_similarity.py --config configs/chosen_model.json --workers 10
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from utils import plot_style  # noqa: F401  (registers the shared style)
from utils.config_loader import load_config
from utils.font_loader import load_font
from utils import multiseed


def _rdm_condensed(z: np.ndarray) -> np.ndarray:
    """Upper-triangle of the pairwise-distance matrix for one seed's latent."""
    return pdist(z, metric="euclidean")


def _similarity_matrix(results) -> tuple[np.ndarray, list]:
    """Pearson correlation between every pair of seeds' RDMs."""
    rdms = [_rdm_condensed(r.latent) for r in results]
    seeds = [r.params.get("seed") for r in results]
    n = len(rdms)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = np.corrcoef(rdms[i], rdms[j])[0, 1]
            M[i, j] = M[j, i] = c
    return M, seeds


def _off_diagonal_mean(M: np.ndarray) -> float:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(M[mask].mean())


def _plot_heatmap(M, seeds, out_path):
    labels = [f"s{s}" for s in seeds]
    off_mean = _off_diagonal_mean(M)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(M, cmap="YlGn", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_title("Similitud estructural entre seeds (invariante a giros)\n"
                 f"media fuera de diagonal = {off_mean:.2f}", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlación de distancias por pares")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path, off_mean


def _write_csv(M, seeds, off_mean, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + [f"s{s}" for s in seeds])
        for i, s in enumerate(seeds):
            w.writerow([f"s{s}"] + [round(float(v), 4) for v in M[i]])
        w.writerow([])
        w.writerow(["off_diagonal_mean", round(off_mean, 4)])
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Cross-seed structural similarity of the latent space (RSA).")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Fallback seed count [1..N] when the config has no "
                             "\"seeds\" list.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Simultaneous runs (1 = sequential).")
    parser.add_argument("--out", default=None, dest="out_dir",
                        help="Output directory (defaults to the config's 'out').")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    out_dir = resolve(args.out_dir or cfg.get("out", "output/simple"))
    os.makedirs(out_dir, exist_ok=True)

    X, _, labels = load_font(resolve(cfg.get("font", "data/font.h")))
    if cfg["layer_dims"][len(cfg["layer_dims"]) // 2] != 2:
        raise ValueError("Structural-similarity study assumes a 2-D latent.")

    base = {k: v for k, v in cfg.items()
            if k not in ("font", "out", "grid", "seeds")}
    seed_list = cfg.get("seeds") or list(range(1, args.seeds + 1))
    print(f"Loaded {len(X)} characters | architecture {cfg['layer_dims']}")
    print(f"Seeds={seed_list} | workers={args.workers}")

    results = multiseed.run_seeds(base, X, list(labels), seed_list, out_dir,
                                  workers=args.workers)

    M, seeds = _similarity_matrix(results)
    png_path, off_mean = _plot_heatmap(
        M, seeds, os.path.join(out_dir, "latent_seed_similarity.png"))
    csv_path = _write_csv(
        M, seeds, off_mean,
        os.path.join(out_dir, "latent_seed_similarity.csv"))

    print(f"\nMedia fuera de diagonal = {off_mean:.3f}")
    print(f"Heatmap -> {png_path}")
    print(f"CSV     -> {csv_path}")


if __name__ == "__main__":
    main()
