"""generate_vae.py
Generate new emojis from a trained VAE.

Produces:
  output/vae_prior_samples.png      — grid of emojis from z ~ N(0, I)
  output/vae_traversal_*.png        — latent interpolation strips between emoji pairs
  output/vae_latent_grid.png        — decode a uniform grid over the 2-D latent space

Run AFTER train_vae.py (uses the best config it saved).
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.emoji_loader import load_emojis
from autoencoders.VariationalAutoencoder import VariationalAutoencoder
from train_vae import train_vae_with_logging

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
ROWS, COLS = 20, 20


def _show_grid(images, n_rows, n_cols, title, filename, labels=None):
    """Display a grid of 20x20 images."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        if idx < len(images):
            ax.imshow(images[idx].reshape(ROWS, COLS), cmap="gray_r", vmin=0, vmax=1)
            if labels and idx < len(labels):
                ax.set_title(labels[idx], fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Prior sampling ───────────────────────────────────────────────────────────

def prior_sample_grid(vae, n=16, seed=0):
    rng = np.random.default_rng(seed)
    latent_dim = vae.layer_dims[vae.bottleneck_idx]
    z = rng.standard_normal((n, latent_dim))
    decoded = vae.decode(z)
    # Also produce binarised versions
    binary = (decoded >= 0.5).astype(np.float32)

    ncols = min(n, 8)
    nrows = (n + ncols - 1) // ncols

    p1 = _show_grid(decoded, nrows, ncols,
                    "VAE Prior Samples (continuous)", "vae_prior_samples.png")
    p2 = _show_grid(binary, nrows, ncols,
                    "VAE Prior Samples (binarised)", "vae_prior_samples_bin.png")
    return p1, p2


# ── Latent traversal ─────────────────────────────────────────────────────────

def latent_traversal(vae, X_bw, labels, idx1, idx2, n_steps=10):
    """Interpolate between two emojis in latent space."""
    mu1 = vae.encode(X_bw[idx1:idx1+1])
    mu2 = vae.encode(X_bw[idx2:idx2+1])

    ts = np.linspace(0, 1, n_steps)
    interpolated = []
    for t in ts:
        z = (1 - t) * mu1 + t * mu2
        decoded = vae.decode(z)
        interpolated.append(decoded[0])

    images = [X_bw[idx1]] + interpolated + [X_bw[idx2]]
    step_labels = [f"{labels[idx1]}"] + [f"t={t:.1f}" for t in ts] + [f"{labels[idx2]}"]

    fig, axes = plt.subplots(1, len(images), figsize=(1.3 * len(images), 1.8))
    for i, (img, lbl) in enumerate(zip(images, step_labels)):
        axes[i].imshow(img.reshape(ROWS, COLS), cmap="gray_r", vmin=0, vmax=1)
        axes[i].set_title(lbl, fontsize=7)
        axes[i].set_xticks([]); axes[i].set_yticks([])

    fig.suptitle(f"Latent Traversal: {labels[idx1]} -> {labels[idx2]}", fontsize=11)
    plt.tight_layout()
    safe_name = f"vae_traversal_{idx1}_{idx2}.png"
    path = os.path.join(OUT_DIR, safe_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path



def latent_traversal_annotated(vae, X_bw, labels, idx1, idx2,
                               n_steps=9, out_dir=".", filename=None):
    latent_dim = vae.layer_dims[vae.bottleneck_idx]
    if latent_dim != 2:
        print(f"  annotated traversal needs latent_dim=2 (got {latent_dim}); "
              f"the decoded strip still works, but the 2-D map is skipped.")

    # encode endpoints to their MEANS (deterministic latent position)
    mu_all = vae.encode(X_bw)                  # (n, 2) — for the background scatter
    mu1 = vae.encode(X_bw[idx1:idx1 + 1])[0]   # (2,)
    mu2 = vae.encode(X_bw[idx2:idx2 + 1])[0]   # (2,)

    # the straight line between them, sampled at n_steps values of t
    ts = np.linspace(0.0, 1.0, n_steps)
    zs = np.array([(1.0 - t) * mu1 + t * mu2 for t in ts])      # (n_steps, 2)
    frames = np.array([vae.decode(z[None, :])[0] for z in zs])  # (n_steps, 400)

    fig = plt.figure(figsize=(max(9, 1.25 * n_steps), 5.0))
    gs = fig.add_gridspec(2, n_steps, height_ratios=[3.2, 1.3], hspace=0.35)

    # ── top: latent-space map (only meaningful for 2-D latent) ──
    if latent_dim == 2:
        ax = fig.add_subplot(gs[0, :])
        ax.scatter(mu_all[:, 0], mu_all[:, 1], c="lightgray", s=70, zorder=1)
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
        ax.set_title("Latent space: the line walked between the two emojis",
                     fontsize=10)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("t", fontsize=8)

    # ── bottom: decoded image under each sampled z ──
    for i in range(n_steps):
        axi = fig.add_subplot(gs[1, i])
        axi.imshow(frames[i].reshape(ROWS, COLS), cmap="gray_r", vmin=0, vmax=1)
        axi.set_title(f"t={ts[i]:.2f}", fontsize=7)
        axi.set_xticks([]); axi.set_yticks([])

    fig.suptitle(f"Latent traversal: {labels[idx1]}  →  {labels[idx2]}", fontsize=12)
    filename = filename or f"vae_traversal_annotated_{idx1}_{idx2}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
# ── 2-D latent grid ─────────────────────────────────────────────────────────

def latent_grid_decode(vae, grid_size=15, z_range=2.5):
    """Decode a uniform grid over the 2-D latent space."""
    latent_dim = vae.layer_dims[vae.bottleneck_idx]
    if latent_dim != 2:
        print(f"  Skipping latent grid (latent_dim={latent_dim}, need 2)")
        return None

    z1 = np.linspace(-z_range, z_range, grid_size)
    z2 = np.linspace(-z_range, z_range, grid_size)

    canvas = np.zeros((grid_size * ROWS, grid_size * COLS))
    for i, zi in enumerate(reversed(z2)):
        for j, zj in enumerate(z1):
            z = np.array([[zj, zi]])
            decoded = vae.decode(z)[0].reshape(ROWS, COLS)
            canvas[i*ROWS:(i+1)*ROWS, j*COLS:(j+1)*COLS] = decoded

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"Latent Space Grid ({grid_size}x{grid_size}, range=[-{z_range},{z_range}])")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "vae_latent_grid.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emojis.h")
    X_color, X_bw, bitmaps_color, bitmaps_bw, labels = load_emojis(data_path)
    print(f"Loaded {len(labels)} emojis, X_bw shape = {X_bw.shape}")

    # Load best config or use defaults
    cfg_path = os.path.join(OUT_DIR, "best_vae_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"Using config from {cfg_path}: {cfg}")
    else:
        cfg = {
            "layer_dims": [400, 128, 2, 128, 400],
            "beta": 0.5,
            "lr": 1e-3,
            "epochs": 3000,
            "activation": "relu",
            "seed": 42,
            "batch_size": 20,
        }
        print(f"No saved config found, using defaults: {cfg}")

    # Train
    print("\nTraining VAE...")
    vae = VariationalAutoencoder(
        cfg["layer_dims"], activation=cfg["activation"],
        seed=cfg["seed"], beta=cfg["beta"]
    )
    train_vae_with_logging(
        vae, X_bw, cfg["epochs"], cfg["lr"],
        batch_size=cfg["batch_size"], log_every=500, patience=500
    )

    # 1) Prior sampling
    print("\n--- Prior Sampling ---")
    p1, p2 = prior_sample_grid(vae, n=16, seed=123)
    print(f"  Continuous -> {p1}")
    print(f"  Binarised  -> {p2}")

    # 2) Latent traversals between several emoji pairs
    print("\n--- Latent Traversals ---")
    n = len(labels)
    pairs = [
        (0, n//2),
        (1, n-1),
        (0, n-1),
        (n//4, 3*n//4),
    ]
    for idx1, idx2 in pairs:
        if idx1 < n and idx2 < n:
            p = latent_traversal_annotated(vae, X_bw, labels, idx1, idx2,
                                       n_steps=9, out_dir=OUT_DIR)
            print(f"  {labels[idx1]} -> {labels[idx2]}: {p}")

    # 3) 2-D latent grid
    print("\n--- Latent Grid ---")
    p = latent_grid_decode(vae, grid_size=15, z_range=2.5)
    if p:
        print(f"  Grid -> {p}")

    print("\n=== generate_vae.py complete ===")


if __name__ == "__main__":
    main()
