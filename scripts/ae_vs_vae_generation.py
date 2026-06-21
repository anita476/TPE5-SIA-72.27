"""ae_vs_vae_generation.py
Prove that a plain autoencoder CANNOT generate new images, while a VAE can.

Teaching dataset: 20 B&W emojis, 20x20, built from emoji glyphs.

The script trains both an AE and a VAE with the SAME architecture, then:
  1. Shows both latent spaces side by side (AE = scattered, VAE = organised)
  2. Samples random z points and decodes them with both models
     -> AE produces garbage, VAE produces recognisable emojis
  3. Decodes a uniform grid over the 2-D latent space with both
     -> AE grid has meaningless gaps, VAE grid morphs smoothly

All outputs saved to output/ with ae_vs_vae_ prefix.

Usage:
    python scripts/ae_vs_vae_generation.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.emoji_loader import load_emojis
from autoencoders.SimpleAutoencoder import SimpleAutoencoder
from autoencoders.VariationalAutoencoder import VariationalAutoencoder

# ── Config ──────────────────────────────────────────────────────────────────
SEED       = 42
LAYER_DIMS = [400, 128, 2, 128, 400]
ACTIVATION = "tanh"
LR         = 1e-3
EPOCHS     = 10000
BATCH_SIZE = 20
PATIENCE   = 500
LOG_EVERY  = 1000

ROWS, COLS = 20, 20
CMAP       = "gray_r"
PREFIX     = "ae_vs_vae_"
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


# ── 1. Latent space comparison ──────────────────────────────────────────────

def plot_latent_comparison(mu_ae, mu_vae, labels, bitmaps_bw):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, mu, title in [(axes[0], mu_ae,  "Plain Autoencoder"),
                           (axes[1], mu_vae, "Variational Autoencoder")]:
        for i in range(len(mu)):
            img = OffsetImage(bitmaps_bw[i], zoom=1.3, cmap=CMAP)
            ab = AnnotationBbox(img, (mu[i, 0], mu[i, 1]),
                                frameon=True, pad=0.25,
                                bboxprops=dict(edgecolor="steelblue",
                                               linewidth=0.6))
            ax.add_artist(ab)

        # 1-sigma circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "--", color="tomato",
                alpha=0.6, linewidth=1.2, label="unit circle (1-sigma)")

        # set limits
        all_pts = mu
        for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
            lo, hi = all_pts[:, dim].min(), all_pts[:, dim].max()
            span = max(hi - lo, 2.5)
            mid = (lo + hi) / 2
            setter(mid - 0.65 * span, mid + 0.65 * span)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Latent Space: AE vs VAE  (20 B&W emojis, latent_dim=2)",
                 fontsize=14)
    plt.tight_layout()
    _save(fig, "latent_spaces.png")


# ── 2. Prior sampling comparison ────────────────────────────────────────────

def plot_prior_samples(ae, vae, mu_ae, mu_vae, bitmaps_bw):
    rng = np.random.default_rng(7)
    n = 16

    # Sample from N(0, I) — the VAE's prior
    z_prior = rng.standard_normal((n, 2))

    # Also sample from the convex hull / bounding box of the AE's actual codes,
    # to be maximally fair to the AE
    ae_lo = mu_ae.min(axis=0)
    ae_hi = mu_ae.max(axis=0)
    z_ae_range = ae_lo + rng.random((n, 2)) * (ae_hi - ae_lo)

    dec_ae_prior = ae.decode(z_prior)
    dec_ae_range = ae.decode(z_ae_range)
    dec_vae_prior = vae.decode(z_prior)

    fig, axes = plt.subplots(3, n, figsize=(1.2 * n, 4.2))

    row_labels = [
        "AE decode\nz ~ N(0,I)",
        "AE decode\nz ~ AE range",
        "VAE decode\nz ~ N(0,I)",
    ]
    decoded_rows = [dec_ae_prior, dec_ae_range, dec_vae_prior]

    for row, (decoded, label) in enumerate(zip(decoded_rows, row_labels)):
        for col in range(n):
            _imshow(axes[row, col], decoded[col])
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right",
                                 va="center", labelpad=60)

    fig.suptitle("Can the model generate new images from random z?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "prior_samples.png")

    # Also show WHERE those z points sit relative to training data
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mu, z, title in [
        (axes[0], mu_ae, z_prior, "AE latent + sampled z ~ N(0,I)"),
        (axes[1], mu_vae, z_prior, "VAE latent + sampled z ~ N(0,I)"),
    ]:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        for i in range(len(mu)):
            img = OffsetImage(bitmaps_bw[i], zoom=1.0, cmap=CMAP)
            ab = AnnotationBbox(img, (mu[i, 0], mu[i, 1]),
                                frameon=True, pad=0.15,
                                bboxprops=dict(edgecolor="lightgray",
                                               linewidth=0.4, alpha=0.5))
            ax.add_artist(ab)
        ax.scatter(z[:, 0], z[:, 1], c="tab:red", s=80, marker="*",
                   edgecolor="black", linewidth=0.5, zorder=5,
                   label="sampled z")
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "--", color="tomato",
                alpha=0.5, linewidth=1.0, label="1-sigma")
        all_pts = np.vstack([mu, z])
        for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
            lo, hi = all_pts[:, dim].min(), all_pts[:, dim].max()
            span = max(hi - lo, 2.5)
            mid = (lo + hi) / 2
            setter(mid - 0.6 * span, mid + 0.6 * span)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Where do the sampled z points land?", fontsize=12)
    plt.tight_layout()
    _save(fig, "sample_locations.png")


# ── 3. Latent grid comparison ──────────────────────────────────────────────

def plot_latent_grid_comparison(ae, vae, grid_size=12, z_range=2.0):
    z1 = np.linspace(-z_range, z_range, grid_size)
    z2 = np.linspace(-z_range, z_range, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, model, title in [(axes[0], ae, "Plain AE"),
                               (axes[1], vae, "VAE")]:
        canvas = np.zeros((grid_size * ROWS, grid_size * COLS))
        for i, zi in enumerate(reversed(z2)):
            for j, zj in enumerate(z1):
                decoded = model.decode(np.array([[zj, zi]]))[0]
                canvas[i*ROWS:(i+1)*ROWS, j*COLS:(j+1)*COLS] = \
                    decoded.reshape(ROWS, COLS)

        ax.imshow(canvas, cmap=CMAP, vmin=0, vmax=1)
        ax.set_title(f"{title}: decode every point in "
                     f"[{-z_range}, {z_range}]^2", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("2-D Latent Grid: what does the decoder see everywhere?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "latent_grids.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print("Loading 20 B&W emojis (20x20)...")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emojis.h")
    X_color, X_bw, bitmaps_color, bitmaps_bw, labels = load_emojis(data_path)
    X = X_bw
    print(f"  {len(labels)} emojis, X shape = {X.shape}")

    # Train plain AE
    print(f"\nTraining plain AE: {LAYER_DIMS}, {EPOCHS} epochs...")
    t0 = time.time()
    ae = SimpleAutoencoder(LAYER_DIMS, activation=ACTIVATION, seed=SEED)
    ae.train(X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
             log_every=LOG_EVERY, optimizer="adam", patience=PATIENCE)
    ae_time = time.time() - t0
    ae_recon = np.mean((ae.decode(ae.encode(X)) - X) ** 2)
    print(f"  AE done in {ae_time:.1f}s, recon MSE = {ae_recon:.6f}")

    # Train VAE
    print(f"\nTraining VAE: {LAYER_DIMS}, {EPOCHS} epochs...")
    t1 = time.time()
    vae = VariationalAutoencoder(LAYER_DIMS, activation=ACTIVATION,
                                  seed=SEED, recon_loss="bce")
    total, recon, kl = vae.train(X, EPOCHS, LR, batch_size=BATCH_SIZE,
                                  log_every=LOG_EVERY, optimizer="adam",
                                  patience=PATIENCE)
    vae_time = time.time() - t1
    vae_recon = np.mean((vae.decode(vae.encode(X)) - X) ** 2)
    print(f"  VAE done in {vae_time:.1f}s, recon MSE = {vae_recon:.6f}")

    # Encode
    mu_ae = ae.encode(X)
    mu_vae = vae.encode(X)

    # Generate all figures
    print("\n--- 1. Latent space comparison ---")
    plot_latent_comparison(mu_ae, mu_vae, labels, bitmaps_bw)

    print("\n--- 2. Prior sampling: can the model generate? ---")
    plot_prior_samples(ae, vae, mu_ae, mu_vae, bitmaps_bw)

    print("\n--- 3. Latent grid: what does the decoder see? ---")
    plot_latent_grid_comparison(ae, vae)

    print(f"\nDone. AE={ae_time:.1f}s, VAE={vae_time:.1f}s")


if __name__ == "__main__":
    main()
