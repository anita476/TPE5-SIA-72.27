"""olivetti_vae.py
Single-run VAE on Olivetti faces: train once, generate all plots, done.

Usage:
    python scripts/olivetti_vae.py

All outputs saved to output/ with olivetti_ prefix.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.datasets import fetch_olivetti_faces
from autoencoders.VariationalAutoencoder import VariationalAutoencoder

# ── Tunables ────────────────────────────────────────────────────────────────
SEED       = 42
LAYER_DIMS = [4096, 256, 2, 256, 4096]
ACTIVATION = "relu"
LR         = 1e-3
EPOCHS     = 2500
BATCH_SIZE = 32
PATIENCE   = 80
LOG_EVERY  = 50

ROWS, COLS = 64, 64
CMAP       = "gray"
PREFIX     = "olivetti_"
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "output")

# ── Helpers ─────────────────────────────────────────────────────────────────

def _imshow(ax, flat):
    ax.imshow(flat.reshape(ROWS, COLS), cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def _save(fig, name):
    path = os.path.join(OUT_DIR, f"{PREFIX}{name}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")
    return path


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_loss_curves(total, recon, kl):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, color, title in [
        (axes[0], total,  "steelblue",  "Total Loss (recon + KL)"),
        (axes[1], recon,  "darkorange", "Reconstruction Loss (MSE)"),
        (axes[2], kl,     "green",      "KL Divergence"),
    ]:
        ax.plot(data, linewidth=0.8, color=color)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Olivetti VAE Training ({len(total)} epochs)", fontsize=11)
    plt.tight_layout()
    _save(fig, "loss_curves.png")


def plot_reconstructions(X, vae, n=10):
    latent = vae.encode(X[:n])
    recon = vae.decode(latent)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for i in range(n):
        _imshow(axes[0, i], X[i])
        _imshow(axes[1, i], recon[i])
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=9)
            axes[1, i].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle("Olivetti VAE Reconstructions", fontsize=11)
    plt.tight_layout()
    _save(fig, "reconstructions.png")


def plot_prior_samples(vae, n=16):
    rng = np.random.default_rng(123)
    latent_dim = LAYER_DIMS[len(LAYER_DIMS) // 2]
    z = rng.standard_normal((n, latent_dim))
    decoded = vae.decode(z)

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n:
            _imshow(ax, decoded[idx])
        else:
            ax.axis("off")
    fig.suptitle("VAE Prior Samples  z ~ N(0, I)", fontsize=11)
    plt.tight_layout()
    _save(fig, "prior_samples.png")


def plot_latent_scatter(X, vae, labels):
    mu = vae.encode(X)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(mu[:, 0], mu[:, 1], c=labels, cmap="tab20", s=22, alpha=0.75)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Subject ID", fontsize=9)
    ax.set_title("Olivetti VAE Latent Space (latent_dim=2)", fontsize=11)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "latent_scatter.png")


def plot_traversal(vae, X, labels, idx1, idx2, tag, n_steps=9):
    mu_all = vae.encode(X)
    mu1 = mu_all[idx1]
    mu2 = mu_all[idx2]

    ts = np.linspace(0.0, 1.0, n_steps)
    zs = np.array([(1.0 - t) * mu1 + t * mu2 for t in ts])
    frames = np.array([vae.decode(z[None, :])[0] for z in zs])

    fig = plt.figure(figsize=(max(10, 1.4 * n_steps), 6.0))
    gs = fig.add_gridspec(2, n_steps, height_ratios=[3.0, 1.4], hspace=0.35)

    # Top: latent scatter + interpolation line
    ax = fig.add_subplot(gs[0, :])
    ax.scatter(mu_all[:, 0], mu_all[:, 1], c="lightgray", s=12, alpha=0.4, zorder=1)
    ax.plot(zs[:, 0], zs[:, 1], "--", color="gray", lw=1.2, zorder=2)
    sc = ax.scatter(zs[:, 0], zs[:, 1], c=ts, cmap="viridis",
                    s=55, zorder=3, edgecolor="white", linewidth=0.5)
    ax.scatter(*mu1, c="tab:blue", s=170, zorder=5, edgecolor="black", linewidth=1.0)
    ax.scatter(*mu2, c="tab:red",  s=170, zorder=5, edgecolor="black", linewidth=1.0)
    ax.annotate(f"s{labels[idx1]}", mu1, fontsize=8, ha="center", va="bottom",
                xytext=(0, 10), textcoords="offset points", color="tab:blue")
    ax.annotate(f"s{labels[idx2]}", mu2, fontsize=8, ha="center", va="bottom",
                xytext=(0, 10), textcoords="offset points", color="tab:red")
    ax.set_title("Latent space: traversal path", fontsize=10)
    ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("t", fontsize=8)

    # Bottom: decoded face strip
    for i in range(n_steps):
        axi = fig.add_subplot(gs[1, i])
        _imshow(axi, frames[i])
        axi.set_title(f"t={ts[i]:.2f}", fontsize=7)

    fig.suptitle(f"Latent traversal: subject {labels[idx1]} -> subject {labels[idx2]}  ({tag})",
                 fontsize=11)
    _save(fig, f"traversal_{tag}_{idx1}_{idx2}.png")


def plot_latent_grid(vae, grid_size=15, z_range=2.5):
    z1 = np.linspace(-z_range, z_range, grid_size)
    z2 = np.linspace(-z_range, z_range, grid_size)

    canvas = np.zeros((grid_size * ROWS, grid_size * COLS))
    for i, zi in enumerate(reversed(z2)):
        for j, zj in enumerate(z1):
            decoded = vae.decode(np.array([[zj, zi]]))[0]
            canvas[i*ROWS:(i+1)*ROWS, j*COLS:(j+1)*COLS] = decoded.reshape(ROWS, COLS)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap=CMAP, vmin=0, vmax=1)
    ax.set_title(f"Latent Grid ({grid_size}x{grid_size}, z in [-{z_range}, {z_range}])")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    _save(fig, "latent_grid.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load data
    print("Loading Olivetti faces...")
    data = fetch_olivetti_faces(shuffle=True, random_state=SEED)
    X = data.data.astype(np.float32)
    labels = data.target
    print(f"  {X.shape[0]} images, {ROWS}x{COLS}, {len(set(labels))} subjects")

    # 2) Train
    print(f"\nTraining VAE: arch={LAYER_DIMS}, lr={LR}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}, patience={PATIENCE}")
    vae = VariationalAutoencoder(LAYER_DIMS, activation=ACTIVATION,
                                 seed=SEED, recon_loss="mse")
    t0 = time.time()
    total, recon, kl = vae.train(X, EPOCHS, LR, batch_size=BATCH_SIZE,
                                  log_every=LOG_EVERY, optimizer="adam",
                                  patience=PATIENCE)
    elapsed = time.time() - t0
    print(f"\nTraining done: {len(total)} epochs in {elapsed:.1f}s "
          f"({elapsed/len(total):.2f}s/epoch)")
    print(f"  Final loss: total={total[-1]:.4f}  recon={recon[-1]:.4f}  KL={kl[-1]:.4f}")

    # 3) Generate all outputs
    print("\n--- Loss Curves ---")
    plot_loss_curves(total, recon, kl)

    print("\n--- Reconstructions ---")
    plot_reconstructions(X, vae)

    print("\n--- Prior Samples ---")
    plot_prior_samples(vae)

    print("\n--- Latent Scatter ---")
    plot_latent_scatter(X, vae, labels)

    # Expression traversal: same subject, different images
    print("\n--- Traversal: expression ---")
    for subj in [0, 5]:
        idxs = np.where(labels == subj)[0]
        if len(idxs) >= 2:
            plot_traversal(vae, X, labels, int(idxs[0]), int(idxs[-1]), "expr")
            break

    # Identity traversal: different subjects
    print("\n--- Traversal: identity ---")
    for s1, s2 in [(0, 20), (5, 35), (1, 30)]:
        i1 = np.where(labels == s1)[0]
        i2 = np.where(labels == s2)[0]
        if len(i1) and len(i2):
            plot_traversal(vae, X, labels, int(i1[0]), int(i2[0]), "identity")
            break

    print("\n--- 2-D Latent Grid ---")
    plot_latent_grid(vae)

    print(f"\nAll done.  Total wall time: {time.time() - t0:.1f}s "
          f"(training: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
