"""generate_vae.py
Generate new images from a trained VAE.

Supports emoji (20x20), Fashion-MNIST (28x28), Olivetti faces (64x64), and
CelebA (20x20 grayscale).

Produces (with prefix):
  output/<prefix>vae_prior_samples.png
  output/<prefix>vae_traversal_*.png
  output/<prefix>vae_latent_grid.png

Run AFTER train_vae.py (uses the best config it saved).
"""
from __future__ import annotations

import _bootstrap
from _bootstrap import resolve

import argparse
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.emoji_loader import load_emojis
from autoencoders.VariationalAutoencoder import VariationalAutoencoder

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Module-level defaults; overridden in main() per dataset.
ROWS, COLS = 20, 20
GRAY_CMAP = "gray_r"


def _pick_olivetti_pairs(vae, X, class_ids, seed=42):
    """Pick expression (same subject) and identity (different subject) pairs.

    For expression pairs, picks the two same-subject images whose latent means
    are farthest apart — tends to be the biggest expression/pose change.

    Returns (pairs, titles) where pairs is a list of (idx1, idx2) and titles
    is a list of figure titles.
    """
    rng = np.random.default_rng(seed)
    class_ids = np.asarray(class_ids)
    mu_all = vae.encode(X)
    subjects = np.unique(class_ids)

    pairs, titles = [], []

    # --- Expression morphs: same subject, max latent distance ---
    for subj in rng.choice(subjects, size=2, replace=False):
        idxs = np.where(class_ids == subj)[0]
        mus = mu_all[idxs]
        # Find the pair with max Euclidean distance in latent space
        best_dist, best_i, best_j = -1, 0, 1
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                d = np.linalg.norm(mus[i] - mus[j])
                if d > best_dist:
                    best_dist = d
                    best_i, best_j = i, j
        a, b = int(idxs[best_i]), int(idxs[best_j])
        pairs.append((a, b))
        titles.append(f"Expression morph (same subject {subj})")
        print(f"  Expression pair: idx {a},{b} (subject {subj}), "
              f"latent dist={best_dist:.3f}")

    # --- Identity morphs: different subjects ---
    subj_pairs = rng.choice(subjects, size=(2, 2), replace=False)
    for s1, s2 in subj_pairs:
        i1 = int(rng.choice(np.where(class_ids == s1)[0]))
        i2 = int(rng.choice(np.where(class_ids == s2)[0]))
        pairs.append((i1, i2))
        titles.append(f"Identity morph (subject {s1} -> subject {s2})")
        print(f"  Identity pair: idx {i1} (subj {s1}) -> {i2} (subj {s2})")

    return pairs, titles


def _imshow(ax, flat, is_color=False):
    """Display a single flattened image."""
    if is_color:
        ax.imshow(np.clip(flat.reshape(ROWS, COLS, 3), 0, 1))
    else:
        ax.imshow(flat.reshape(ROWS, COLS), cmap=GRAY_CMAP, vmin=0, vmax=1)


# ── Prior sampling ───────────────────────────────────────────────────────────

def prior_sample_grid(vae, X, labels, n=16, seed=0, is_color=False, prefix="",
                      bitmaps_bw=None, bitmaps_color=None):
    rng = np.random.default_rng(seed)
    latent_dim = vae.layer_dims[vae.bottleneck_idx]
    z = rng.standard_normal((n, latent_dim))
    decoded = vae.decode(z)

    ncols = min(n, 8)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.3 * ncols, 1.3 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n:
            _imshow(ax, decoded[idx], is_color)
        ax.set_xticks([]); ax.set_yticks([])

    mode_label = "Color" if is_color else "B&W"
    fig.suptitle(f"VAE Prior Samples ({mode_label}, continuous)", fontsize=11)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, f"{prefix}vae_prior_samples.png")
    plt.savefig(p1, dpi=150)
    plt.close(fig)

    paths = [p1]

    # Binarised version only makes sense for B&W
    if not is_color:
        binary = (decoded >= 0.5).astype(np.float32)
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.3 * ncols, 1.3 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            if idx < n:
                _imshow(ax, binary[idx], False)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("VAE Prior Samples (binarised)", fontsize=11)
        plt.tight_layout()
        p2 = os.path.join(OUT_DIR, f"{prefix}vae_prior_samples_bin.png")
        plt.savefig(p2, dpi=150)
        plt.close(fig)
        paths.append(p2)

    # Latent space map showing where samples came from (2-D only)
    if latent_dim == 2:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        mu_all = vae.encode(X)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Training data: thumbnails for small datasets, scatter for large
        if bitmaps_bw is not None or bitmaps_color is not None:
            # Small dataset (emojis): show thumbnails
            if is_color and bitmaps_color is not None:
                bitmaps = bitmaps_color
                cmap_kw = {}
            else:
                bitmaps = bitmaps_bw
                cmap_kw = {"cmap": "gray_r"}
            for i in range(len(mu_all)):
                img = OffsetImage(bitmaps[i], zoom=1.0, **cmap_kw)
                ab = AnnotationBbox(img, (mu_all[i, 0], mu_all[i, 1]),
                                    frameon=True, pad=0.2,
                                    bboxprops=dict(edgecolor="lightgray",
                                                   linewidth=0.5, alpha=0.6))
                ax.add_artist(ab)
        else:
            # Large dataset (fashion): scatter dots
            ax.scatter(mu_all[:, 0], mu_all[:, 1], c="lightgray", s=6,
                       alpha=0.3, zorder=1, label="Training data")

        # Sampled z points
        ax.scatter(z[:, 0], z[:, 1], c="tab:red", s=120, marker="*",
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=f"Prior samples (n={n})")
        for i in range(n):
            ax.annotate(str(i+1), (z[i, 0], z[i, 1]),
                        fontsize=7, fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points", color="tab:red")

        # 1-sigma circle of N(0,I)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "--", color="steelblue",
                alpha=0.5, linewidth=1.0, label="1-sigma circle")

        # Set limits with margin
        all_pts = np.vstack([mu_all, z])
        for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
            lo, hi = all_pts[:, dim].min(), all_pts[:, dim].max()
            span = max(hi - lo, 1.0)
            setter(lo - 0.2 * span, hi + 0.2 * span)

        ax.set_title("Prior Sampling: where the new images come from", fontsize=11)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p3 = os.path.join(OUT_DIR, f"{prefix}vae_prior_samples_latent.png")
        plt.savefig(p3, dpi=150)
        plt.close(fig)
        paths.append(p3)

    return paths


# ── Latent traversal (annotated) ─────────────────────────────────────────────

def latent_traversal_annotated(vae, X, labels, idx1, idx2,
                               n_steps=9, out_dir=".", filename=None,
                               is_color=False, prefix="", title=None):
    latent_dim = vae.layer_dims[vae.bottleneck_idx]
    if latent_dim != 2:
        print(f"  annotated traversal needs latent_dim=2 (got {latent_dim}); "
              f"the decoded strip still works, but the 2-D map is skipped.")

    mu_all = vae.encode(X)
    mu1 = vae.encode(X[idx1:idx1 + 1])[0]
    mu2 = vae.encode(X[idx2:idx2 + 1])[0]

    ts = np.linspace(0.0, 1.0, n_steps)
    zs = np.array([(1.0 - t) * mu1 + t * mu2 for t in ts])
    frames = np.array([vae.decode(z[None, :])[0] for z in zs])

    fig = plt.figure(figsize=(max(9, 1.25 * n_steps), 5.0))
    gs = fig.add_gridspec(2, n_steps, height_ratios=[3.2, 1.3], hspace=0.35)

    if latent_dim == 2:
        ax = fig.add_subplot(gs[0, :])
        bg_s = 8 if len(mu_all) > 100 else 70
        ax.scatter(mu_all[:, 0], mu_all[:, 1], c="lightgray", s=bg_s,
                   alpha=0.4, zorder=1)
        ax.plot(zs[:, 0], zs[:, 1], "--", color="gray", lw=1.2, zorder=2)
        sc = ax.scatter(zs[:, 0], zs[:, 1], c=ts, cmap="viridis",
                        s=55, zorder=3, edgecolor="white", linewidth=0.5)
        ax.scatter(*mu1, c="tab:blue", s=170, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.scatter(*mu2, c="tab:red", s=170, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.annotate(labels[idx1], mu1, fontsize=8, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points", color="tab:blue")
        ax.annotate(labels[idx2], mu2, fontsize=8, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points", color="tab:red")
        ax.set_title("Latent space: traversal path", fontsize=10)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("t", fontsize=8)

    for i in range(n_steps):
        axi = fig.add_subplot(gs[1, i])
        _imshow(axi, frames[i], is_color)
        axi.set_title(f"t={ts[i]:.2f}", fontsize=7)
        axi.set_xticks([]); axi.set_yticks([])

    default_title = f"Latent traversal: {labels[idx1]}  ->  {labels[idx2]}"
    fig.suptitle(title or default_title, fontsize=12)
    filename = filename or f"{prefix}vae_traversal_annotated_{idx1}_{idx2}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 2-D latent grid ─────────────────────────────────────────────────────────

def latent_grid_decode(vae, grid_size=15, z_range=2.5, is_color=False, prefix=""):
    latent_dim = vae.layer_dims[vae.bottleneck_idx]
    if latent_dim != 2:
        print(f"  Skipping latent grid (latent_dim={latent_dim}, need 2)")
        return None

    z1 = np.linspace(-z_range, z_range, grid_size)
    z2 = np.linspace(-z_range, z_range, grid_size)

    if is_color:
        canvas = np.ones((grid_size * ROWS, grid_size * COLS, 3))
    else:
        canvas = np.zeros((grid_size * ROWS, grid_size * COLS))

    for i, zi in enumerate(reversed(z2)):
        for j, zj in enumerate(z1):
            z = np.array([[zj, zi]])
            decoded = vae.decode(z)[0]
            if is_color:
                img = np.clip(decoded.reshape(ROWS, COLS, 3), 0, 1)
            else:
                img = decoded.reshape(ROWS, COLS)
            canvas[i*ROWS:(i+1)*ROWS, j*COLS:(j+1)*COLS] = img

    fig, ax = plt.subplots(figsize=(10, 10))
    if is_color:
        ax.imshow(np.clip(canvas, 0, 1))
    else:
        ax.imshow(canvas, cmap=GRAY_CMAP, vmin=0, vmax=1)
    ax.set_title(f"Latent Space Grid ({grid_size}x{grid_size}, range=[-{z_range},{z_range}])")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{prefix}vae_latent_grid.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global ROWS, COLS, GRAY_CMAP

    parser = argparse.ArgumentParser(description="Generate images from trained VAE")
    parser.add_argument("--color", action="store_true",
                        help="Use color (RGB 1200-dim) mode [emoji only]")
    parser.add_argument("--dataset", type=str, default="emoji",
                        choices=["emoji", "fashion", "olivetti", "celeba"],
                        help="Dataset to use (default: emoji)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to load (fashion/celeba only)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Dataset loading ---------------------------------------------------
    if args.dataset == "olivetti":
        from utils.olivetti_loader import load_olivetti
        from utils.olivetti_loader import ROWS as O_ROWS, COLS as O_COLS
        ROWS, COLS = O_ROWS, O_COLS
        GRAY_CMAP = "gray"
        print("Loading Olivetti faces...")
        X, class_ids, label_names = load_olivetti(seed=0)
        labels = [f"s{c}" for c in class_ids]
        is_color = False
        bitmaps_bw = None
        bitmaps_color = None
        prefix = "olivetti_"
        print(f"  {len(X)} images, X shape = {X.shape}")

    elif args.dataset == "fashion":
        from utils.fashion_mnist_loader import load_fashion_mnist
        from utils.fashion_mnist_loader import ROWS as F_ROWS, COLS as F_COLS
        ROWS, COLS = F_ROWS, F_COLS
        GRAY_CMAP = "gray_r"
        print("Loading Fashion-MNIST...")
        X, class_ids, label_names = load_fashion_mnist(
            n_samples=args.n_samples or 4000, seed=0)
        labels = [label_names[c] for c in class_ids]
        is_color = False
        bitmaps_bw = None
        bitmaps_color = None
        prefix = "fashion_"
        print(f"  {len(X)} samples, X shape = {X.shape}")
    elif args.dataset == "celeba":
        from utils.celeba_loader import load_celeba
        from utils.celeba_loader import ROWS as C_ROWS, COLS as C_COLS
        ROWS, COLS = C_ROWS, C_COLS
        GRAY_CMAP = "gray"
        print("Loading CelebA...")
        X, class_ids, label_names = load_celeba(
            n_samples=args.n_samples or 2000, seed=0)
        labels = [label_names[c] for c in class_ids]
        is_color = False
        bitmaps_bw = None
        bitmaps_color = None
        prefix = "celeba_"
        print(f"  {len(X)} samples, X shape = {X.shape}")
    else:
        ROWS, COLS = 20, 20
        GRAY_CMAP = "gray_r"
        is_color = args.color
        prefix = "color_" if is_color else ""
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emojis.h")
        X_color, X_bw, bitmaps_color, bitmaps_bw, labels = load_emojis(data_path)
        X = X_color if is_color else X_bw
        class_ids = None
        label_names = None
        print(f"Loaded {len(labels)} emojis, X shape = {X.shape} "
              f"({'color' if is_color else 'bw'})")

    # Load best config or use defaults
    cfg_path = os.path.join(OUT_DIR, f"{prefix}best_vae_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"Using config from {cfg_path}")
    else:
        if args.dataset == "olivetti":
            cfg = {
                "layer_dims": [4096, 512, 2, 512, 4096],
                "lr": 1e-3, "epochs": 10000,
                "activation": "tanh", "seed": 42, "batch_size": 32,
                "recon_loss": "mse",
            }
        elif args.dataset == "fashion":
            cfg = {
                "layer_dims": [784, 256, 2, 256, 784],
                "lr": 1e-3, "epochs": 200,
                "activation": "tanh", "seed": 42, "batch_size": 64,
                "recon_loss": "mse",
            }
        elif args.dataset == "celeba":
            cfg = {
                "layer_dims": [4096, 512, 2, 512, 4096],
                "lr": 1e-6, "epochs": 10000,
                "activation": "tanh", "seed": 42, "batch_size": 64,
                "recon_loss": "mse",
            }
        elif is_color:
            cfg = {
                "layer_dims": [1200, 256, 2, 256, 1200],
                "lr": 1e-3, "epochs": 3000,
                "activation": "relu", "seed": 42, "batch_size": 20,
                "recon_loss": "mse",
            }
        else:
            cfg = {
                "layer_dims": [400, 128, 2, 128, 400],
                "lr": 1e-3, "epochs": 3000,
                "activation": "relu", "seed": 42, "batch_size": 20,
                "recon_loss": "bce",
            }
        print("No saved config found, using defaults")

    # Train
    print("\nTraining VAE...")
    vae = VariationalAutoencoder(
        cfg["layer_dims"], activation=cfg["activation"],
        seed=cfg["seed"], recon_loss=cfg.get("recon_loss", "mse"),
    )
    vae.train(
        X, cfg["epochs"], cfg["lr"],
        batch_size=cfg["batch_size"], log_every=500, patience=500
    )

    # 1) Prior sampling
    print("\n--- Prior Sampling ---")
    paths = prior_sample_grid(vae, X, labels, n=16, seed=123, is_color=is_color,
                              prefix=prefix,
                              bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color)
    for p in paths:
        print(f"  -> {p}")

    # 2) Latent traversals
    print("\n--- Latent Traversals ---")
    n = len(X)
    if args.dataset == "olivetti":
        pairs, pair_titles = _pick_olivetti_pairs(vae, X, class_ids)
    elif args.dataset == "fashion":
        rng = np.random.default_rng(42)
        pairs, pair_titles = [], None
        for c1, c2 in [(0, 6), (7, 5), (2, 4), (3, 0)]:
            i1 = rng.choice(np.where(class_ids == c1)[0])
            i2 = rng.choice(np.where(class_ids == c2)[0])
            pairs.append((int(i1), int(i2)))
    else:
        pair_titles = None
        pairs = [
            (0, n//2),
            (1, n-1),
            (0, n-1),
            (n//4, 3*n//4),
        ]
    for k, (idx1, idx2) in enumerate(pairs):
        if idx1 < n and idx2 < n:
            title = pair_titles[k] if pair_titles else None
            p = latent_traversal_annotated(vae, X, labels, idx1, idx2,
                                           n_steps=9, out_dir=OUT_DIR,
                                           is_color=is_color, prefix=prefix,
                                           title=title)
            print(f"  {labels[idx1]} -> {labels[idx2]}: {p}")

    # 3) 2-D latent grid
    print("\n--- Latent Grid ---")
    p = latent_grid_decode(vae, grid_size=15, z_range=2.5,
                           is_color=is_color, prefix=prefix)
    if p:
        print(f"  Grid -> {p}")

    # 4) Latent scatter (non-emoji datasets)
    latent_dim = vae.layer_dims[vae.bottleneck_idx]
    if latent_dim == 2 and args.dataset != "emoji":
        from train_vae import plot_latent_scatter_classes, plot_latent_scatter_colorbar
        print("\n--- Latent Scatter ---")
        latent = vae.encode(X)
        if args.dataset == "olivetti":
            p = plot_latent_scatter_colorbar(
                latent, class_ids,
                "Olivetti VAE Latent Space",
                out_dir=OUT_DIR,
                filename=f"{prefix}vae_latent_scatter_subjects.png")
        else:  # fashion / celeba
            title = {"fashion": "Fashion-MNIST", "celeba": "CelebA"}.get(
                args.dataset, args.dataset) + " VAE Latent Space"
            p = plot_latent_scatter_classes(
                latent, class_ids, label_names, title,
                out_dir=OUT_DIR,
                filename=f"{prefix}vae_latent_scatter_classes.png")
        print(f"  -> {p}")

    print("\n=== generate_vae.py complete ===")


if __name__ == "__main__":
    main()
