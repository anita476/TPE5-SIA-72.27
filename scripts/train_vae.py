"""train_vae.py
Train a standard Variational Autoencoder on the B&W emoji dataset (20x20).

Architecture sweep with multi-seed reporting (mean +/- std).

Produces:
  output/vae_latent_dim_sweep.png   -- recon MSE vs latent dim with error bars
  output/vae_arch_sweep.png         -- recon MSE per architecture with error bars
  output/vae_arch_tuning_table.txt  -- all configs with mean +/- std
  output/vae_loss_curves.png        -- total / recon / KL vs epoch (best config)
  output/vae_latent_scatter_*.png   -- 2-D latent scatter
  output/vae_vs_ae_latent_*.png     -- side-by-side AE vs VAE
  output/vae_reconstructions.png    -- original vs reconstructed
  output/best_vae_config.json       -- chosen config for generate_vae.py
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
from autoencoders.SimpleAutoencoder import SimpleAutoencoder

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


# -- helpers -----------------------------------------------------------------

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
            total = recon + kl

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


def train_and_evaluate(X, cfg, seed):
    """Train one VAE with given config and seed, return metrics."""
    vae = VariationalAutoencoder(
        cfg["layer_dims"],
        activation=cfg.get("activation", "relu"),
        seed=seed,
    )
    total, recon, kl = train_vae_with_logging(
        vae, X, cfg["epochs"], cfg["lr"],
        batch_size=cfg.get("batch_size", 20),
        log_every=cfg.get("log_every", 500),
        patience=cfg.get("patience"),
    )
    # Evaluate: encode -> decode -> MSE
    latent = vae.encode(X)
    recon_out = vae.decode(latent)
    recon_mse = np.mean((recon_out - X) ** 2)
    final_kl = kl[-1] if kl else 0.0
    return {
        "recon_mse": recon_mse,
        "final_kl": final_kl,
        "vae": vae,
        "losses": (total, recon, kl),
        "n_epochs": len(total),
    }


# -- plotting ----------------------------------------------------------------

def plot_loss_curves(total, recon, kl, title_extra="", filename="vae_loss_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(total, linewidth=0.8, color="steelblue")
    axes[0].set_title("Total Loss (recon + KL)")
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
    for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
        lo, hi = latent[:, dim].min(), latent[:, dim].max()
        span = hi - lo if hi > lo else 1.0
        margin = span * margin_frac
        setter(lo - margin, hi + margin)


def _add_latent_images(ax, latent, bitmaps, zoom, cmap=None):
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
    ax.scatter(latent[:, 0], latent[:, 1],
               c=np.arange(len(latent)), cmap="tab20", s=100, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (latent[i, 0], latent[i, 1]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")


def plot_latent_scatter(latent, labels, title, filename,
                        bitmaps_bw=None, bitmaps_color=None, mode="bw"):
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


def _imshow_emoji(ax, flat, is_color=False):
    if is_color:
        img = np.clip(flat.reshape(20, 20, 3), 0, 1)
        ax.imshow(img)
    else:
        ax.imshow(flat.reshape(20, 20), cmap="gray_r", vmin=0, vmax=1)


def plot_reconstructions(X, model, labels, filename="vae_reconstructions.png",
                         is_color=False):
    latent = model.encode(X)
    recon = model.decode(latent)
    n = len(X)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        _imshow_emoji(axes[0, i], X[i], is_color)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[0, i].set_title(labels[i], fontsize=7)
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=9)
        _imshow_emoji(axes[1, i], recon[i], is_color)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle("VAE Reconstructions", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- sweep logic -------------------------------------------------------------

def load_sweep_configs(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def run_one_sweep(X, sweep_configs, shared, seeds, sweep_name):
    """Run one sweep axis: for each config, train over all seeds.
    Returns list of dicts with mean/std metrics.
    """
    results = []
    for i, cfg_overrides in enumerate(sweep_configs):
        cfg = {**shared, **cfg_overrides}
        label = str(cfg["layer_dims"])
        print(f"\n  Config {i+1}/{len(sweep_configs)}: {label}")

        mses, kls = [], []
        best_vae = None
        best_mse = float("inf")
        best_losses = None

        for seed in seeds:
            print(f"    seed={seed} ...", end=" ", flush=True)
            res = train_and_evaluate(X, cfg, seed)
            mses.append(res["recon_mse"])
            kls.append(res["final_kl"])
            print(f"MSE={res['recon_mse']:.6f} KL={res['final_kl']:.4f}")
            if res["recon_mse"] < best_mse:
                best_mse = res["recon_mse"]
                best_vae = res["vae"]
                best_losses = res["losses"]

        results.append({
            "config": cfg,
            "label": label,
            "mse_mean": np.mean(mses),
            "mse_std": np.std(mses),
            "kl_mean": np.mean(kls),
            "kl_std": np.std(kls),
            "n_seeds": len(seeds),
            "best_vae": best_vae,
            "best_losses": best_losses,
        })
        print(f"    => MSE: {results[-1]['mse_mean']:.6f} +/- {results[-1]['mse_std']:.6f}  "
              f"KL: {results[-1]['kl_mean']:.4f} +/- {results[-1]['kl_std']:.4f}")

    return results


def plot_sweep_errorbar(results, x_label, title, filename, x_values=None):
    """Generic error-bar plot for a sweep axis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [r["mse_mean"] for r in results]
    stds = [r["mse_std"] for r in results]

    if x_values is not None:
        ax.errorbar(x_values, means, yerr=stds, fmt="o-", capsize=5,
                    color="steelblue", markersize=8)
        ax.set_xticks(x_values)
    else:
        x = range(len(results))
        ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=5,
                    color="steelblue", markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in results], fontsize=8, rotation=15)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Reconstruction MSE", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_tuning_table(all_results, shared, seeds, filename="vae_arch_tuning_table.txt"):
    path = os.path.join(OUT_DIR, filename)
    lines = []
    lines.append(f"{'#':>2}  {'Architecture':<40}  {'Recon MSE':>18}  "
                 f"{'KL':>18}  {'Seeds':>5}")
    lines.append("-" * 100)

    for i, r in enumerate(all_results):
        lines.append(
            f"{i+1:>2}  {r['label']:<40}  "
            f"{r['mse_mean']:>8.6f} +/- {r['mse_std']:<7.6f}  "
            f"{r['kl_mean']:>8.4f} +/- {r['kl_std']:<7.4f}  "
            f"{r['n_seeds']:>5}"
        )

    table = "\n".join(lines)
    header = (f"VAE Architecture Tuning Results (standard VAE, KL weight=1)\n"
              f"Fixed: lr={shared['lr']}, epochs={shared['epochs']}, "
              f"batch_size={shared['batch_size']}, optimizer={shared['optimizer']}, "
              f"activation={shared['activation']}\n"
              f"Seeds: {seeds}\n")

    with open(path, "w") as f:
        f.write(header)
        f.write("=" * 100 + "\n")
        f.write(table + "\n")
    print(f"\nTuning table saved to {path}")
    print(table)
    return path


# -- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VAE on emoji dataset")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to sweep config JSON")
    parser.add_argument("--color", action="store_true",
                        help="Train on color (RGB 1200-dim) instead of B&W (400-dim)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    is_color = args.color
    prefix = "color_" if is_color else ""

    print("Loading emojis...")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emojis.h")
    X_color, X_bw, bitmaps_color, bitmaps_bw, labels = load_emojis(data_path)

    if is_color:
        X = X_color
        default_sweep = os.path.join(os.path.dirname(__file__), "..",
                                     "configs", "vae_sweep_color.json")
        print(f"  COLOR mode: {len(labels)} emojis, X shape = {X.shape}")
    else:
        X = X_bw
        default_sweep = os.path.join(os.path.dirname(__file__), "..",
                                     "configs", "vae_sweep.json")
        print(f"  B&W mode: {len(labels)} emojis, X shape = {X.shape}")

    # Load config
    config_path = args.config or default_sweep
    raw = load_sweep_configs(config_path)
    shared = raw["shared"]
    seeds = raw["seeds"]

    # ---- Sweep A: Latent dimension ----------------------------------------
    print("\n=== Sweep A: Latent Dimension ===")
    latent_configs = raw["latent_dim_sweep"]["configs"]
    latent_results = run_one_sweep(X, latent_configs, shared, seeds, "latent_dim")

    latent_dims = [r["config"]["layer_dims"][len(r["config"]["layer_dims"]) // 2]
                   for r in latent_results]
    p = plot_sweep_errorbar(
        latent_results,
        x_label="Latent Dimension",
        title=(f"Reconstruction MSE vs Latent Dimension\n"
               f"(body=[..., 128, L, 128, ...], lr={shared['lr']}, "
               f"epochs={shared['epochs']}, n_seeds={len(seeds)})"),
        filename=f"{prefix}vae_latent_dim_sweep.png",
        x_values=latent_dims,
    )
    print(f"Latent dim sweep plot -> {p}")

    # ---- Sweep B: Architecture (depth/width) ------------------------------
    print("\n=== Sweep B: Depth/Width ===")
    arch_configs = raw["arch_sweep"]["configs"]
    arch_results = run_one_sweep(X, arch_configs, shared, seeds, "architecture")

    p = plot_sweep_errorbar(
        arch_results,
        x_label="Architecture",
        title=(f"Reconstruction MSE vs Architecture (latent_dim=2)\n"
               f"(lr={shared['lr']}, epochs={shared['epochs']}, "
               f"n_seeds={len(seeds)})"),
        filename=f"{prefix}vae_arch_sweep.png",
    )
    print(f"Arch sweep plot -> {p}")

    # ---- Tuning table (all results) ----------------------------------------
    all_results = latent_results + arch_results
    write_tuning_table(all_results, shared, seeds,
                       filename=f"{prefix}vae_arch_tuning_table.txt")

    # ---- Pick best 2-D config for downstream plots -------------------------
    candidates_2d = [r for r in all_results
                     if r["config"]["layer_dims"][len(r["config"]["layer_dims"]) // 2] == 2]
    if candidates_2d:
        best = min(candidates_2d, key=lambda r: r["mse_mean"])
    else:
        best = min(all_results, key=lambda r: r["mse_mean"])

    best_cfg = best["config"]
    best_vae = best["best_vae"]
    best_total, best_recon, best_kl = best["best_losses"]

    print(f"\nBest 2-D config: {best_cfg['layer_dims']} "
          f"(MSE={best['mse_mean']:.6f} +/- {best['mse_std']:.6f})")

    # ---- Loss curves for best model ----------------------------------------
    p = plot_loss_curves(
        best_total, best_recon, best_kl,
        title_extra=f"(arch={best_cfg['layer_dims']})",
        filename=f"{prefix}vae_loss_curves.png"
    )
    print(f"Loss curves -> {p}")

    # ---- Reconstructions ---------------------------------------------------
    p = plot_reconstructions(X, best_vae, labels,
                             filename=f"{prefix}vae_reconstructions.png",
                             is_color=is_color)
    print(f"Reconstructions -> {p}")

    # ---- VAE latent scatter (3 variants) -----------------------------------
    latent_vae = best_vae.encode(X)
    for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
        p = plot_latent_scatter(latent_vae, labels,
                                f"VAE Latent Space (arch={best_cfg['layer_dims']})",
                                f"{prefix}vae_latent_scatter{suffix}.png",
                                bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                mode=mode)
        print(f"VAE latent ({mode}) -> {p}")

    # ---- Plain AE for comparison -------------------------------------------
    print("\nTraining plain AE for latent space comparison...")
    ae = SimpleAutoencoder(best_cfg["layer_dims"], activation="relu", seed=42)
    ae.train(X, epochs=3000, lr=1e-3, batch_size=20, log_every=500,
             optimizer="adam", patience=500)

    latent_ae = ae.encode(X)
    for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
        p = plot_latent_scatter(latent_ae, labels,
                                "Plain AE Latent Space",
                                f"{prefix}ae_latent_scatter{suffix}.png",
                                bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                mode=mode)
        print(f"AE latent ({mode}) -> {p}")

    # ---- Side-by-side comparison -------------------------------------------
    for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
        p = plot_latent_comparison(latent_vae, latent_ae, labels,
                                   bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                   filename=f"{prefix}vae_vs_ae_latent{suffix}.png",
                                   mode=mode)
        print(f"Comparison ({mode}) -> {p}")

    # ---- Save best config --------------------------------------------------
    cfg_out = {
        "layer_dims": best_cfg["layer_dims"],
        "lr": best_cfg["lr"],
        "epochs": best_cfg["epochs"],
        "activation": best_cfg.get("activation", "relu"),
        "seed": 42,
        "batch_size": best_cfg.get("batch_size", 20),
        "is_color": is_color,
        "criterion": ("latent_dim=2 for 2-D visualizations, since it best supports "
                      "the latent-space and traversal plots, noting the "
                      "reconstruction tradeoff"),
    }
    cfg_path = os.path.join(OUT_DIR, f"{prefix}best_vae_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"Best config -> {cfg_path}")

    print("\n=== train_vae.py complete ===")


if __name__ == "__main__":
    main()
