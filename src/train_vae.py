"""train_vae.py
Train a Variational Autoencoder on the B&W emoji dataset (20x20).

Produces:
  output/vae_loss_curves.png        — total / recon / KL vs epoch
  output/vae_latent_scatter.png     — 2-D latent scatter (VAE)
  output/ae_latent_scatter.png      — 2-D latent scatter (plain AE, for comparison)
  output/vae_vs_ae_latent.png       — side-by-side comparison
  output/vae_reconstructions.png    — original vs reconstructed for all emojis
  output/vae_tuning_table.txt       — hyperparameter sweep results
"""
from __future__ import annotations

import argparse
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.emoji_loader import load_emojis
from autoencoders.VariationalAutoencoder import VariationalAutoencoder
from autoencoders.SimpleAutoencoder import SimpleAutoencoder

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


# ── helpers ──────────────────────────────────────────────────────────────────

def train_vae_with_logging(vae, X, epochs, lr, batch_size, optimizer="adam",
                           log_every=100, patience=None, min_delta=1e-6):
    """Train VAE and record recon, KL, and total loss per epoch separately."""
    if optimizer == "adam":
        vae._init_adam_state()

    total_losses, recon_losses, kl_losses = [], [], []
    best_loss = float("inf")
    best_snapshot = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        idx = vae.rng.permutation(len(X))
        X_shuffled = X[idx]
        ep_total, ep_recon, ep_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for start in range(0, len(X), batch_size):
            batch = X_shuffled[start:start + batch_size]
            out = vae.forward(batch)
            N = batch.shape[0]
            recon = np.sum((out - batch) ** 2) / N
            kl = vae.kl_divergence(vae._mu, vae._logvar)
            total = recon + vae.beta * kl

            ep_recon += recon
            ep_kl += kl
            ep_total += total

            dW, db = vae._compute_grads(batch)
            if optimizer == "adam":
                vae._apply_adam(dW, db, lr)
            else:
                vae._apply_sgd(dW, db, lr)
            n_batches += 1

        mean_total = ep_total / n_batches
        mean_recon = ep_recon / n_batches
        mean_kl = ep_kl / n_batches

        total_losses.append(mean_total)
        recon_losses.append(mean_recon)
        kl_losses.append(mean_kl)

        if mean_total < best_loss - min_delta:
            best_loss = mean_total
            best_snapshot = vae._snapshot()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if log_every and (epoch + 1) % log_every == 0:
            print(f"  Epoch {epoch+1}/{epochs} | total={mean_total:.6f} "
                  f"recon={mean_recon:.6f} KL={mean_kl:.6f}")

        if patience is not None and epochs_no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1} (best={best_loss:.6f})")
            break

    if best_snapshot is not None:
        vae._restore(best_snapshot)

    return total_losses, recon_losses, kl_losses


def plot_loss_curves(total, recon, kl, title_extra="", filename="vae_loss_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(total, linewidth=0.8, color="steelblue")
    axes[0].set_title("Total Loss (recon + beta*KL)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(recon, linewidth=0.8, color="darkorange")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(kl, linewidth=0.8, color="green")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("KL"); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"VAE Training Curves {title_extra}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _set_latent_limits(ax, latent, margin_frac=0.15):
    """Set axis limits with padding proportional to the data range."""
    for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
        lo, hi = latent[:, dim].min(), latent[:, dim].max()
        span = hi - lo if hi > lo else 1.0
        margin = span * margin_frac
        setter(lo - margin, hi + margin)


def _add_latent_images(ax, latent, bitmaps, zoom, cmap=None):
    """Place bitmap thumbnails at latent coordinates."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    _set_latent_limits(ax, latent, margin_frac=0.20)
    for i in range(len(latent)):
        kwargs = {"zoom": zoom}
        if cmap is not None:
            kwargs["cmap"] = cmap
        img = OffsetImage(bitmaps[i], **kwargs)
        ab = AnnotationBbox(img, (latent[i, 0], latent[i, 1]),
                            frameon=True, pad=0.3,
                            bboxprops=dict(edgecolor="steelblue", linewidth=0.8))
        ax.add_artist(ab)


def _add_latent_dots(ax, latent, labels):
    """Colored dots with text labels."""
    ax.scatter(latent[:, 0], latent[:, 1],
               c=np.arange(len(latent)), cmap="tab20", s=100, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (latent[i, 0], latent[i, 1]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")


def plot_latent_scatter(latent, labels, title, filename,
                        bitmaps_bw=None, bitmaps_color=None, mode="bw"):
    """Plot a single latent scatter.
    mode: 'dots', 'bw', or 'color'.
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    if mode == "color" and bitmaps_color is not None:
        _add_latent_images(ax, latent, bitmaps_color, zoom=1.5)
    elif mode == "bw" and bitmaps_bw is not None:
        _add_latent_images(ax, latent, bitmaps_bw, zoom=1.5, cmap="gray_r")
    else:
        _add_latent_dots(ax, latent, labels)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latent_comparison(latent_vae, latent_ae, labels,
                           bitmaps_bw=None, bitmaps_color=None,
                           filename="vae_vs_ae_latent.png", mode="bw"):
    """Side-by-side AE vs VAE latent scatter.
    mode: 'dots', 'bw', or 'color'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, latent, title in [
        (axes[0], latent_ae, "Plain Autoencoder - Latent Space"),
        (axes[1], latent_vae, "Variational Autoencoder - Latent Space"),
    ]:
        if mode == "color" and bitmaps_color is not None:
            _add_latent_images(ax, latent, bitmaps_color, zoom=1.2)
        elif mode == "bw" and bitmaps_bw is not None:
            _add_latent_images(ax, latent, bitmaps_bw, zoom=1.2, cmap="gray_r")
        else:
            _add_latent_dots(ax, latent, labels)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_reconstructions(X_bw, model, labels, filename="vae_reconstructions.png",
                         rows=20, cols=20):
    latent = model.encode(X_bw)
    recon = model.decode(latent)
    n = len(X_bw)

    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(X_bw[i].reshape(rows, cols), cmap="gray_r", vmin=0, vmax=1)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[0, i].set_title(labels[i], fontsize=7)
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=9)

        axes[1, i].imshow(recon[i].reshape(rows, cols), cmap="gray_r", vmin=0, vmax=1)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=9)

    fig.suptitle("VAE Reconstructions", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── tuning sweep ─────────────────────────────────────────────────────────────

def load_sweep_configs(config_path: str) -> tuple[list[dict], dict]:
    """Load sweep configurations from a JSON file.

    Expected format:
        { "sweep": [ {per-run overrides}, ... ],
          "shared": {defaults applied to every run} }
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    shared = raw.get("shared", {})
    sweep = raw.get("sweep", [])
    if not sweep:
        raise ValueError(f"No 'sweep' entries found in {config_path}")

    configs = []
    for entry in sweep:
        merged = {**shared, **entry}
        configs.append(merged)

    return configs, shared


def run_sweep(X_bw, labels, config_path: str | None = None):
    """Hyperparameter sweep; loads configs from JSON if provided."""
    default_config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "vae_sweep.json"
    )
    path = config_path or default_config_path

    configs, shared = load_sweep_configs(path)
    print(f"Loaded {len(configs)} sweep configs from {path}")

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n--- Sweep {i+1}/{len(configs)}: {cfg} ---")
        vae = VariationalAutoencoder(
            cfg["layer_dims"],
            activation=cfg.get("activation", "relu"),
            seed=cfg.get("seed", 42),
            beta=cfg.get("beta", 1.0),
        )
        total, recon, kl = train_vae_with_logging(
            vae, X_bw, cfg["epochs"], cfg["lr"],
            batch_size=cfg.get("batch_size", 20),
            log_every=cfg.get("log_every", 500),
            patience=cfg.get("patience"),
        )

        # evaluate
        latent = vae.encode(X_bw)
        recon_out = vae.decode(latent)
        recon_err = np.mean((recon_out - X_bw) ** 2)
        final_kl = kl[-1] if kl else 0
        final_total = total[-1] if total else 0

        results.append({
            "config": cfg,
            "final_total": final_total,
            "final_recon": recon_err,
            "final_kl": final_kl,
            "n_epochs": len(total),
            "vae": vae,
            "losses": (total, recon, kl),
        })
        print(f"  Final: total={final_total:.6f} recon_mse={recon_err:.6f} KL={final_kl:.6f}")

    return results


def write_tuning_table(results, filename="vae_tuning_table.txt"):
    path = os.path.join(OUT_DIR, filename)
    lines = []
    lines.append(f"{'#':>2}  {'Architecture':<30}  {'Beta':>5}  {'LR':>6}  "
                 f"{'Epochs':>6}  {'Recon':>8}  {'KL':>8}  {'Total':>8}")
    lines.append("-" * 95)

    for i, r in enumerate(results):
        c = r["config"]
        lines.append(
            f"{i+1:>2}  {str(c['layer_dims']):<30}  {c['beta']:>5.1f}  {c['lr']:>6.4f}  "
            f"{r['n_epochs']:>6}  {r['final_recon']:>8.4f}  {r['final_kl']:>8.4f}  "
            f"{r['final_total']:>8.4f}"
        )

    table = "\n".join(lines)
    with open(path, "w") as f:
        f.write("VAE Hyperparameter Tuning Results\n")
        f.write("=" * 95 + "\n")
        f.write(table + "\n")
    print(f"\nTuning table saved to {path}")
    print(table)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train VAE on emoji dataset")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to sweep config JSON (default: configs/vae_sweep.json)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading emojis...")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emojis.h")
    _, X_bw, bitmaps_color, bitmaps_bw, labels = load_emojis(data_path)
    print(f"  {len(labels)} emojis, X_bw shape = {X_bw.shape}")

    # ── Step 1: Hyperparameter sweep ─────────────────────────────────────
    print("\n=== Hyperparameter Sweep ===")
    sweep_results = run_sweep(X_bw, labels, config_path=args.config)
    write_tuning_table(sweep_results)

    # Pick the best 2-D latent config by lowest reconstruction error
    # (total loss isn't comparable across different beta values)
    best_2d = min(
        [r for r in sweep_results if r["config"]["layer_dims"][2] == 2],
        key=lambda r: r["final_recon"]
    )
    best_cfg = best_2d["config"]
    best_vae = best_2d["vae"]
    best_total, best_recon, best_kl = best_2d["losses"]

    print(f"\nBest 2-D config: {best_cfg}")

    # ── Step 2: Loss curves for best model ───────────────────────────────
    p1 = plot_loss_curves(
        best_total, best_recon, best_kl,
        title_extra=f"(beta={best_cfg['beta']})", filename="vae_loss_curves.png"
    )
    print(f"Loss curves -> {p1}")

    # ── Step 3: Reconstructions ──────────────────────────────────────────
    p2 = plot_reconstructions(X_bw, best_vae, labels)
    print(f"Reconstructions -> {p2}")

    # ── Step 4: VAE latent scatter (3 variants) ─────────────────────────
    latent_vae = best_vae.encode(X_bw)
    for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
        p = plot_latent_scatter(latent_vae, labels,
                                f"VAE Latent Space (beta={best_cfg['beta']})",
                                f"vae_latent_scatter{suffix}.png",
                                bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                mode=mode)
        print(f"VAE latent ({mode}) -> {p}")

    # ── Step 5: Plain AE for comparison ──────────────────────────────────
    print("\nTraining plain AE for latent space comparison...")
    ae = SimpleAutoencoder(best_cfg["layer_dims"], activation="relu", seed=42)
    ae.train(X_bw, epochs=3000, lr=1e-3, batch_size=20, log_every=500,
             optimizer="adam", patience=500)

    latent_ae = ae.encode(X_bw)
    for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
        p = plot_latent_scatter(latent_ae, labels,
                                "Plain AE Latent Space",
                                f"ae_latent_scatter{suffix}.png",
                                bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                mode=mode)
        print(f"AE latent ({mode}) -> {p}")

    # ── Step 6: Side-by-side comparison (3 variants) ─────────────────────
    for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
        p = plot_latent_comparison(latent_vae, latent_ae, labels,
                                   bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                   filename=f"vae_vs_ae_latent{suffix}.png", mode=mode)
        print(f"Comparison ({mode}) -> {p}")

    # ── Save best model config ───────────────────────────────────────────
    cfg_out = {
        "layer_dims": best_cfg["layer_dims"],
        "beta": best_cfg["beta"],
        "lr": best_cfg["lr"],
        "epochs": best_cfg["epochs"],
        "activation": "relu",
        "seed": 42,
        "batch_size": 20,
    }
    cfg_path = os.path.join(OUT_DIR, "best_vae_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"Best config -> {cfg_path}")

    print("\n=== train_vae.py complete ===")


if __name__ == "__main__":
    main()
