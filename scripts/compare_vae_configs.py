"""compare_vae_configs.py
Compare BCE vs MSE and ReLU+He vs tanh+Xavier.

Supports: --dataset emoji (default) or --dataset fashion

Trains 4 VAE configs (same architecture, same seed):
  1. BCE  + ReLU  + He
  2. BCE  + tanh  + Xavier
  3. MSE  + ReLU  + He
  4. MSE  + tanh  + Xavier

Outputs per config (in output/vae_compare_<dataset>/<name>/):
  - loss_curves.png          (total, recon, KL)
  - reconstructions.png      (original vs reconstructed)
  - latent_scatter.png       (2-D latent space)
  - latent_grid.png          (decoded grid over z1, z2)

Outputs (in output/):
  - vae_compare_<dataset>_summary.png  (bar chart)
  - vae_compare_<dataset>_table.txt    (text table)
"""
from __future__ import annotations

import _bootstrap

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoencoders.VariationalAutoencoder import VariationalAutoencoder

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Module-level; set in main() per dataset.
ROWS, COLS = 20, 20
GRAY_CMAP = "gray_r"

CONFIGS = [
    {
        "name": "BCE + ReLU + He",
        "recon_loss": "bce",
        "activation": "relu",
        "weight_init": "he",
    },
    {
        "name": "BCE + tanh + Xavier",
        "recon_loss": "bce",
        "activation": "tanh",
        "weight_init": "xavier",
    },
    {
        "name": "MSE + ReLU + He",
        "recon_loss": "mse",
        "activation": "relu",
        "weight_init": "he",
    },
    {
        "name": "MSE + tanh + Xavier",
        "recon_loss": "mse",
        "activation": "tanh",
        "weight_init": "xavier",
    },
]

DATASET_PARAMS = {
    "emoji": {
        "layer_dims": [400, 128, 2, 128, 400],
        "lr": 1e-3,
        "epochs": 10000,
        "batch_size": 20,
        "seed": 42,
        "patience": 500,
        "log_every": 1000,
        "title": "B&W Emojis",
    },
    "fashion": {
        "layer_dims": [784, 256, 2, 256, 784],
        "lr": 1e-3,
        "epochs": 200,
        "batch_size": 64,
        "seed": 42,
        "patience": 50,
        "log_every": 20,
        "title": "Fashion-MNIST",
    },
}


# -- helpers ------------------------------------------------------------------

def _imshow(ax, flat):
    ax.imshow(flat.reshape(ROWS, COLS), cmap=GRAY_CMAP, vmin=0, vmax=1)


def plot_loss_curves(total, recon, kl, title, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(total, lw=0.8, color="steelblue")
    axes[0].set_title("Total Loss"); axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(recon, lw=0.8, color="darkorange")
    axes[1].set_title("Reconstruction Loss"); axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(kl, lw=0.8, color="green")
    axes[2].set_title("KL Divergence"); axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_reconstructions(X, vae, labels, title, out_dir, n_show=20):
    latent = vae.encode(X)
    recon = vae.decode(latent)
    n = min(len(X), n_show)
    fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 3.0))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        _imshow(axes[0, i], X[i])
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[0, i].set_title(labels[i], fontsize=5)
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=8)
        _imshow(axes[1, i], recon[i])
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "reconstructions.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_latent_scatter(latent, labels, title, out_dir,
                        bitmaps_bw=None, class_ids=None, label_names=None):
    fig, ax = plt.subplots(figsize=(10, 9))

    if bitmaps_bw is not None and len(bitmaps_bw) <= 50:
        # Small dataset (emojis): show thumbnails
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
            lo, hi = latent[:, dim].min(), latent[:, dim].max()
            span = max(hi - lo, 1.0)
            setter(lo - 0.2 * span, hi + 0.2 * span)
        for i in range(len(latent)):
            img = OffsetImage(bitmaps_bw[i], zoom=1.5, cmap=GRAY_CMAP)
            ab = AnnotationBbox(img, (latent[i, 0], latent[i, 1]),
                                frameon=True, pad=0.3,
                                bboxprops=dict(edgecolor="steelblue", lw=0.8))
            ax.add_artist(ab)
    elif class_ids is not None and label_names is not None:
        # Large dataset: colored scatter by class
        for c, name in enumerate(label_names):
            mask = class_ids == c
            ax.scatter(latent[mask, 0], latent[mask, 1],
                       s=8, alpha=0.4, label=name)
        ax.legend(fontsize=7, markerscale=2, loc="best")
    else:
        ax.scatter(latent[:, 0], latent[:, 1], s=10, alpha=0.4)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "latent_scatter.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_latent_grid(vae, title, out_dir, grid_size=15, z_range=2.5):
    z1 = np.linspace(-z_range, z_range, grid_size)
    z2 = np.linspace(-z_range, z_range, grid_size)
    canvas = np.zeros((grid_size * ROWS, grid_size * COLS))
    for i, zi in enumerate(reversed(z2)):
        for j, zj in enumerate(z1):
            z = np.array([[zj, zi]])
            decoded = vae.decode(z)[0]
            canvas[i*ROWS:(i+1)*ROWS, j*COLS:(j+1)*COLS] = decoded.reshape(ROWS, COLS)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap=GRAY_CMAP, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(out_dir, "latent_grid.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_summary(results, dataset_title, dataset_name):
    names = [r["name"] for r in results]
    recons = [r["final_recon"] for r in results]
    kls = [r["final_kl"] for r in results]
    mses = [r["recon_mse"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))
    w = 0.5

    axes[0].bar(x, recons, w, color="darkorange")
    axes[0].set_title("Final Reconstruction Loss (training)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, kls, w, color="green")
    axes[1].set_title("Final KL Divergence")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(x, mses, w, color="steelblue")
    axes[2].set_title("Reconstruction MSE (encode->decode)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")

    arch = DATASET_PARAMS[dataset_name]["layer_dims"]
    fig.suptitle(f"VAE Config Comparison ({dataset_title}, {arch})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"vae_compare_{dataset_name}_summary.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    return path


def write_table(results, dataset_title, dataset_name):
    lines = []
    header = (f"{'Config':<25}  {'Recon Loss':>12}  {'Activation':>10}  "
              f"{'Init':>8}  {'Final Recon':>12}  {'Final KL':>10}  "
              f"{'Final Total':>12}  {'MSE':>10}  {'Epochs':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r['name']:<25}  {r['recon_loss']:>12}  {r['activation']:>10}  "
            f"{r['weight_init']:>8}  {r['final_recon']:>12.4f}  "
            f"{r['final_kl']:>10.4f}  {r['final_total']:>12.4f}  "
            f"{r['recon_mse']:>10.6f}  {r['n_epochs']:>6}"
        )
    table = "\n".join(lines)
    path = os.path.join(OUT_DIR, f"vae_compare_{dataset_name}_table.txt")
    with open(path, "w") as f:
        f.write(f"VAE Config Comparison — {dataset_title}\n")
        f.write("=" * len(header) + "\n")
        f.write(table + "\n")
    print(f"\n{table}")
    print(f"\nTable saved to {path}")
    return path


# -- main ---------------------------------------------------------------------

def main():
    global ROWS, COLS, GRAY_CMAP

    parser = argparse.ArgumentParser(
        description="Compare VAE loss/activation/init configs")
    parser.add_argument("--dataset", type=str, default="emoji",
                        choices=["emoji", "fashion"],
                        help="Dataset to use (default: emoji)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of Fashion-MNIST samples (default: 4000)")
    args = parser.parse_args()

    dataset_name = args.dataset
    shared = DATASET_PARAMS[dataset_name]
    compare_dir = os.path.join(OUT_DIR, f"vae_compare_{dataset_name}")
    os.makedirs(compare_dir, exist_ok=True)

    # ---- Load data ----------------------------------------------------------
    bitmaps_bw = None
    class_ids = None
    label_names = None

    if dataset_name == "emoji":
        ROWS, COLS = 20, 20
        GRAY_CMAP = "gray_r"
        from utils.emoji_loader import load_emojis
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emojis.h")
        _, X, _, bitmaps_bw, labels = load_emojis(data_path)
        print(f"Loaded {len(labels)} emojis, X shape = {X.shape}")
    elif dataset_name == "fashion":
        ROWS, COLS = 28, 28
        GRAY_CMAP = "gray_r"
        from utils.fashion_mnist_loader import load_fashion_mnist
        n_samples = args.n_samples or 4000
        X, class_ids, label_names = load_fashion_mnist(n_samples=n_samples, seed=0)
        labels = [label_names[c] for c in class_ids]
        print(f"Loaded {len(X)} Fashion-MNIST samples, X shape = {X.shape}")

    dataset_title = shared["title"]
    results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        safe_name = name.lower().replace(" ", "_").replace("+", "").replace("__", "_")
        config_dir = os.path.join(compare_dir, safe_name)
        os.makedirs(config_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  recon_loss={cfg['recon_loss']}  activation={cfg['activation']}  "
              f"weight_init={cfg['weight_init']}")
        print(f"{'='*60}")

        vae = VariationalAutoencoder(
            shared["layer_dims"],
            activation=cfg["activation"],
            seed=shared["seed"],
            recon_loss=cfg["recon_loss"],
            weight_init=cfg["weight_init"],
        )
        total, recon, kl = vae.train(
            X, shared["epochs"], shared["lr"],
            batch_size=shared["batch_size"],
            log_every=shared["log_every"],
            patience=shared["patience"],
        )

        # Metrics
        latent = vae.encode(X)
        recon_out = vae.decode(latent)
        recon_mse = np.mean((recon_out - X) ** 2)

        result = {
            "name": name,
            "recon_loss": cfg["recon_loss"],
            "activation": cfg["activation"],
            "weight_init": cfg["weight_init"],
            "final_recon": recon[-1] if recon else 0,
            "final_kl": kl[-1] if kl else 0,
            "final_total": total[-1] if total else 0,
            "recon_mse": recon_mse,
            "n_epochs": len(total),
        }
        results.append(result)
        print(f"  Final: recon={result['final_recon']:.4f}  KL={result['final_kl']:.4f}  "
              f"MSE={recon_mse:.6f}  epochs={result['n_epochs']}")

        # Per-config plots
        plot_loss_curves(total, recon, kl, f"Loss Curves — {name}", config_dir)
        plot_reconstructions(X, vae, labels, f"Reconstructions — {name}",
                             config_dir)
        plot_latent_scatter(latent, labels, f"Latent Space — {name}",
                            config_dir, bitmaps_bw=bitmaps_bw,
                            class_ids=class_ids, label_names=label_names)
        plot_latent_grid(vae, f"Latent Grid — {name}", config_dir)
        print(f"  Plots -> {config_dir}/")

    # Summary
    p = plot_summary(results, dataset_title, dataset_name)
    print(f"\nSummary chart -> {p}")
    write_table(results, dataset_title, dataset_name)

    print(f"\n=== compare_vae_configs.py ({dataset_name}) complete ===")


if __name__ == "__main__":
    main()
