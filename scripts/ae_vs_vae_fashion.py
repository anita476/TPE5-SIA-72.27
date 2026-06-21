"""ae_vs_vae_fashion.py
Prove that a plain AE cannot generate new images, while a VAE can.

Dataset: Fashion-MNIST (28x28 grayscale garments), 4000-sample subsample.
With thousands of images in a 2-D latent space, the AE's codes scatter
arbitrarily — huge dead zones between clusters. The VAE packs everything
into N(0,I), so sampling the prior produces coherent garments.

Outputs saved to output/ with fashion_ae_vs_vae_ prefix.

Usage:
    python scripts/ae_vs_vae_fashion.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.fashion_mnist_loader import load_fashion_mnist, LABEL_NAMES
from autoencoders.SimpleAutoencoder import SimpleAutoencoder
from autoencoders.VariationalAutoencoder import VariationalAutoencoder

# ── Config ──────────────────────────────────────────────────────────────────
SEED       = 42
N_SAMPLES  = 4000
LAYER_DIMS = [784, 256, 2, 256, 784]
ACTIVATION = "tanh"
LR         = 1e-3
EPOCHS     = 200
BATCH_SIZE = 64
PATIENCE   = 30
LOG_EVERY  = 25

ROWS, COLS = 28, 28
CMAP       = "gray_r"
PREFIX     = "fashion_ae_vs_vae_"
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


# ── 1. Latent scatter comparison (class-colored) ───────────────────────────

def plot_latent_comparison(mu_ae, mu_vae, class_ids):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.tab10

    for ax, mu, title in [(axes[0], mu_ae, "Plain Autoencoder"),
                           (axes[1], mu_vae, "Variational Autoencoder")]:
        for c in range(10):
            mask = class_ids == c
            if not np.any(mask):
                continue
            ax.scatter(mu[mask, 0], mu[mask, 1], c=[cmap(c)],
                       s=6, alpha=0.5, label=LABEL_NAMES[c])

        # 1-sigma circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "--", color="red",
                alpha=0.6, linewidth=1.5, label="1-sigma circle")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.legend(fontsize=6, markerscale=3, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Latent Space: AE vs VAE  (Fashion-MNIST, latent_dim=2)",
                 fontsize=14)
    plt.tight_layout()
    _save(fig, "latent_spaces.png")


# ── 2. Prior sampling comparison ──────────────────────────────────────────

def plot_prior_samples(ae, vae, mu_ae):
    rng = np.random.default_rng(7)
    n = 16

    # Sample from N(0, I)
    z_prior = rng.standard_normal((n, 2))

    # Sample from AE's own bounding box (fair to AE)
    ae_lo = mu_ae.min(axis=0)
    ae_hi = mu_ae.max(axis=0)
    z_ae_range = ae_lo + rng.random((n, 2)) * (ae_hi - ae_lo)

    dec_ae_prior = ae.decode(z_prior)
    dec_ae_range = ae.decode(z_ae_range)
    dec_vae_prior = vae.decode(z_prior)

    fig, axes = plt.subplots(3, n, figsize=(1.4 * n, 5.0))

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
                                 va="center", labelpad=65)

    fig.suptitle("Can the model generate new garments from random z?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "prior_samples.png")


# ── 3. Latent grid comparison ─────────────────────────────────────────────

def plot_latent_grid_comparison(ae, vae, grid_size=15, z_range=2.5):
    z1 = np.linspace(-z_range, z_range, grid_size)
    z2 = np.linspace(-z_range, z_range, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, model, title in [(axes[0], ae, "Plain AE"),
                               (axes[1], vae, "VAE")]:
        canvas = np.zeros((grid_size * ROWS, grid_size * COLS))
        for i, zi in enumerate(reversed(z2)):
            for j, zj in enumerate(z1):
                decoded = model.decode(np.array([[zj, zi]]))[0]
                canvas[i*ROWS:(i+1)*ROWS, j*COLS:(j+1)*COLS] = \
                    decoded.reshape(ROWS, COLS)

        ax.imshow(canvas, cmap=CMAP, vmin=0, vmax=1)
        ax.set_title(f"{title}: decode every z in "
                     f"[-{z_range}, {z_range}]^2", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("2-D Latent Grid: what does the decoder produce everywhere?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "latent_grids.png")


# ── 4. Reconstructions side by side ───────────────────────────────────────

def plot_reconstructions(X, ae, vae, class_ids, n=10):
    # Pick one of each class
    chosen = []
    for c in range(10):
        idxs = np.where(class_ids == c)[0]
        if len(idxs):
            chosen.append(idxs[0])
    chosen = chosen[:n]

    fig, axes = plt.subplots(3, len(chosen), figsize=(1.5 * len(chosen), 4.5))

    for col, idx in enumerate(chosen):
        _imshow(axes[0, col], X[idx])
        axes[0, col].set_title(LABEL_NAMES[class_ids[idx]], fontsize=6)
        _imshow(axes[1, col], ae.decode(ae.encode(X[idx:idx+1]))[0])
        _imshow(axes[2, col], vae.decode(vae.encode(X[idx:idx+1]))[0])

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("AE recon", fontsize=9)
    axes[2, 0].set_ylabel("VAE recon", fontsize=9)

    fig.suptitle("Reconstruction quality (both models are comparable)",
                 fontsize=11)
    plt.tight_layout()
    _save(fig, "reconstructions.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print(f"Loading Fashion-MNIST ({N_SAMPLES} samples)...")
    X, class_ids, _ = load_fashion_mnist(n_samples=N_SAMPLES, seed=0)
    print(f"  X shape = {X.shape}, 10 classes")

    # Train plain AE
    print(f"\nTraining plain AE: {LAYER_DIMS}, {EPOCHS} epochs, batch={BATCH_SIZE}...")
    t0 = time.time()
    ae = SimpleAutoencoder(LAYER_DIMS, activation=ACTIVATION, seed=SEED)
    ae.train(X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
             log_every=LOG_EVERY, optimizer="adam", patience=PATIENCE)
    ae_time = time.time() - t0
    ae_recon = np.mean((ae.decode(ae.encode(X)) - X) ** 2)
    print(f"  AE done in {ae_time:.1f}s, recon MSE = {ae_recon:.6f}")

    # Train VAE
    print(f"\nTraining VAE: {LAYER_DIMS}, {EPOCHS} epochs, batch={BATCH_SIZE}...")
    t1 = time.time()
    vae = VariationalAutoencoder(LAYER_DIMS, activation=ACTIVATION,
                                  seed=SEED, recon_loss="mse")
    vae.train(X, EPOCHS, LR, batch_size=BATCH_SIZE,
              log_every=LOG_EVERY, optimizer="adam", patience=PATIENCE)
    vae_time = time.time() - t1
    vae_recon = np.mean((vae.decode(vae.encode(X)) - X) ** 2)
    print(f"  VAE done in {vae_time:.1f}s, recon MSE = {vae_recon:.6f}")

    # Encode
    mu_ae = ae.encode(X)
    mu_vae = vae.encode(X)

    print(f"\n  AE latent range: z1=[{mu_ae[:,0].min():.1f}, {mu_ae[:,0].max():.1f}]  "
          f"z2=[{mu_ae[:,1].min():.1f}, {mu_ae[:,1].max():.1f}]")
    print(f"  VAE latent range: z1=[{mu_vae[:,0].min():.1f}, {mu_vae[:,0].max():.1f}]  "
          f"z2=[{mu_vae[:,1].min():.1f}, {mu_vae[:,1].max():.1f}]")

    # Generate figures
    print("\n--- 1. Latent spaces (class-colored) ---")
    plot_latent_comparison(mu_ae, mu_vae, class_ids)

    print("\n--- 2. Reconstructions ---")
    plot_reconstructions(X, ae, vae, class_ids)

    print("\n--- 3. Prior sampling: can the model generate? ---")
    plot_prior_samples(ae, vae, mu_ae)

    print("\n--- 4. Latent grid ---")
    plot_latent_grid_comparison(ae, vae)

    print(f"\nDone. AE={ae_time:.1f}s, VAE={vae_time:.1f}s")


if __name__ == "__main__":
    main()
