"""train_vae.py
Train a standard Variational Autoencoder on emoji (20x20) or Fashion-MNIST (28x28).

Architecture sweep with multi-seed reporting (mean +/- std).

Produces (with optional prefix):
  output/<prefix>vae_latent_dim_sweep.png
  output/<prefix>vae_arch_sweep.png
  output/<prefix>vae_arch_tuning_table.txt
  output/<prefix>vae_configs/<name>/
  output/<prefix>best_vae_config.json
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


def train_and_evaluate(X, cfg, seed):
    """Train one VAE with given config and seed, return metrics."""
    vae = VariationalAutoencoder(
        cfg["layer_dims"],
        activation=cfg.get("activation", "relu"),
        seed=seed,
        recon_loss=cfg.get("recon_loss", "mse"),
    )
    total, recon, kl = vae.train(
        X, cfg["epochs"], cfg["lr"],
        batch_size=cfg.get("batch_size", 20),
        log_every=cfg.get("log_every", 500),
        optimizer=cfg.get("optimizer", "adam"),
        patience=cfg.get("patience"),
        min_delta=cfg.get("min_delta", 1e-6),
    )
    latent = vae.encode(X)
    recon_out = vae.decode(latent)
    recon_mse = np.mean((recon_out - X) ** 2)
    vae.forward(X)
    final_kl = float(vae.kl_divergence(vae._mu, vae._logvar))
    return {
        "recon_mse": recon_mse,
        "final_kl": final_kl,
        "vae": vae,
        "losses": (total, recon, kl),
        "n_epochs": len(total),
    }


# -- plotting ----------------------------------------------------------------

def plot_loss_curves(total, recon, kl, title_extra="", out_dir=None,
                     filename="loss_curves.png"):
    out_dir = out_dir or OUT_DIR
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
    path = os.path.join(out_dir, filename)
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


def plot_latent_scatter(latent, labels, title, out_dir=None,
                        filename="latent_scatter.png",
                        bitmaps_bw=None, bitmaps_color=None, mode="bw"):
    out_dir = out_dir or OUT_DIR
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
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latent_comparison(latent_vae, latent_ae, labels,
                           bitmaps_bw=None, bitmaps_color=None,
                           out_dir=None, filename="vae_vs_ae_latent.png",
                           mode="bw"):
    out_dir = out_dir or OUT_DIR
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
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _imshow_flat(ax, flat, rows=20, cols=20, is_color=False):
    if is_color:
        img = np.clip(flat.reshape(rows, cols, 3), 0, 1)
        ax.imshow(img)
    else:
        ax.imshow(flat.reshape(rows, cols), cmap="gray_r", vmin=0, vmax=1)


def plot_reconstructions(X, model, labels, out_dir=None,
                         filename="reconstructions.png", is_color=False,
                         title="VAE Reconstructions", rows=20, cols=20):
    out_dir = out_dir or OUT_DIR
    latent = model.encode(X)
    recon = model.decode(latent)
    n = min(len(X), 20)  # cap for fashion (thousands of samples)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        _imshow_flat(axes[0, i], X[i], rows, cols, is_color)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        lbl = labels[i] if isinstance(labels[i], str) else str(labels[i])
        axes[0, i].set_title(lbl, fontsize=7)
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=9)
        _imshow_flat(axes[1, i], recon[i], rows, cols, is_color)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latent_scatter_classes(latent, class_ids, label_names, title,
                                out_dir=None, filename="latent_scatter_classes.png"):
    """Scatter colored by class id with a legend — for large datasets."""
    out_dir = out_dir or OUT_DIR
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.tab10
    for c in range(len(label_names)):
        mask = class_ids == c
        if not np.any(mask):
            continue
        ax.scatter(latent[mask, 0], latent[mask, 1], c=[cmap(c)],
                   s=8, alpha=0.5, label=label_names[c])
    ax.legend(fontsize=7, markerscale=3, loc="best")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _config_dir_name(layer_dims):
    """Create a short folder name from layer_dims, e.g. '400_128_2_128_400'."""
    return "_".join(str(d) for d in layer_dims)


def generate_per_config_plots(vae, losses, cfg, X, labels,
                               bitmaps_bw, bitmaps_color, is_color, prefix="",
                               rows=20, cols=20, class_ids=None,
                               label_names=None, dataset="emoji"):
    """Generate loss curves, reconstructions, and latent scatter for one config."""
    dims = cfg["layer_dims"]
    latent_dim = dims[len(dims) // 2]
    dir_name = _config_dir_name(dims)
    config_dir = os.path.join(OUT_DIR, f"{prefix}vae_configs", dir_name)
    os.makedirs(config_dir, exist_ok=True)

    total, recon, kl = losses
    arch_str = str(dims)

    # Loss curves
    plot_loss_curves(total, recon, kl,
                     title_extra=f"(arch={arch_str})",
                     out_dir=config_dir)

    # Reconstructions
    plot_reconstructions(X, vae, labels, out_dir=config_dir,
                         is_color=is_color,
                         title=f"VAE Reconstructions (arch={arch_str})",
                         rows=rows, cols=cols)

    # Latent scatter (only for 2-D latent)
    if latent_dim == 2:
        latent = vae.encode(X)
        if dataset == "fashion" and class_ids is not None:
            plot_latent_scatter_classes(
                latent, class_ids, label_names,
                f"VAE Latent Space (arch={arch_str})",
                out_dir=config_dir)
        else:
            for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
                plot_latent_scatter(latent, labels,
                                    f"VAE Latent Space (arch={arch_str})",
                                    out_dir=config_dir,
                                    filename=f"latent_scatter{suffix}.png",
                                    bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                    mode=mode)

    print(f"    Per-config plots -> {config_dir}/")


# -- sweep logic -------------------------------------------------------------

def load_sweep_configs(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def run_one_sweep(X, sweep_configs, shared, seeds, sweep_name,
                  labels, bitmaps_bw, bitmaps_color, is_color, prefix="",
                  rows=20, cols=20, class_ids=None, label_names=None,
                  dataset="emoji"):
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

        # Generate per-config plots
        generate_per_config_plots(best_vae, best_losses, cfg, X, labels,
                                   bitmaps_bw, bitmaps_color, is_color, prefix,
                                   rows=rows, cols=cols, class_ids=class_ids,
                                   label_names=label_names, dataset=dataset)

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
            f"{r['mse_mean']:>10.8f} +/- {r['mse_std']:<10.8f}  "
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
    parser = argparse.ArgumentParser(description="Train VAE on emoji or Fashion-MNIST")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to sweep config JSON")
    parser.add_argument("--color", action="store_true",
                        help="Train on color (RGB 1200-dim) instead of B&W (400-dim) [emoji only]")
    parser.add_argument("--dataset", type=str, default="emoji",
                        choices=["emoji", "fashion"],
                        help="Dataset to train on (default: emoji)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Dataset loading ---------------------------------------------------
    if args.dataset == "fashion":
        from utils.fashion_mnist_loader import load_fashion_mnist, ROWS as F_ROWS, COLS as F_COLS
        print("Loading Fashion-MNIST...")
        X, class_ids, label_names = load_fashion_mnist(n_samples=4000, seed=0)
        labels = [label_names[c] for c in class_ids]
        is_color = False
        bitmaps_bw = None
        bitmaps_color = None
        prefix = "fashion_"
        rows, cols = F_ROWS, F_COLS
        default_sweep = os.path.join(os.path.dirname(__file__), "..",
                                     "configs", "vae_sweep_fashion.json")
        print(f"  {len(X)} samples, X shape = {X.shape}")
    else:
        is_color = args.color
        prefix = "color_" if is_color else ""
        rows, cols = 20, 20

        print("Loading emojis...")
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emojis.h")
        X_color, X_bw, bitmaps_color, bitmaps_bw, labels = load_emojis(data_path)
        class_ids = None
        label_names = None

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

    sweep_kw = dict(labels=labels, bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                    is_color=is_color, prefix=prefix, rows=rows, cols=cols,
                    class_ids=class_ids if args.dataset == "fashion" else None,
                    label_names=label_names, dataset=args.dataset)

    # ---- Sweep A: Latent dimension ----------------------------------------
    print("\n=== Sweep A: Latent Dimension ===")
    latent_configs = raw["latent_dim_sweep"]["configs"]
    latent_results = run_one_sweep(X, latent_configs, shared, seeds, "latent_dim",
                                   **sweep_kw)

    latent_dims = [r["config"]["layer_dims"][len(r["config"]["layer_dims"]) // 2]
                   for r in latent_results]
    p = plot_sweep_errorbar(
        latent_results,
        x_label="Latent Dimension",
        title=(f"Reconstruction MSE vs Latent Dimension\n"
               f"(lr={shared['lr']}, epochs={shared['epochs']}, "
               f"n_seeds={len(seeds)})"),
        filename=f"{prefix}vae_latent_dim_sweep.png",
        x_values=latent_dims,
    )
    print(f"Latent dim sweep plot -> {p}")

    # ---- Sweep B: Architecture (depth/width) ------------------------------
    print("\n=== Sweep B: Depth/Width ===")
    arch_configs = raw["arch_sweep"]["configs"]
    arch_results = run_one_sweep(X, arch_configs, shared, seeds, "architecture",
                                 **sweep_kw)

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

    print(f"\nBest 2-D config: {best_cfg['layer_dims']} "
          f"(MSE={best['mse_mean']:.6f} +/- {best['mse_std']:.6f})")

    # ---- Plain AE for comparison (best 2-D config only) --------------------
    print("\nTraining plain AE for latent space comparison...")
    ae = SimpleAutoencoder(best_cfg["layer_dims"],
                           activation="tanh", seed=42)
    ae.train(X, epochs=shared["epochs"], lr=shared["lr"],
             batch_size=shared.get("batch_size", 20), log_every=shared.get("log_every", 500),
             optimizer=shared.get("optimizer", "adam"),
             patience=shared.get("patience"))

    latent_vae = best_vae.encode(X)
    latent_ae = ae.encode(X)

    # AE-only plots in best config's folder
    best_dir = os.path.join(OUT_DIR, f"{prefix}vae_configs",
                            _config_dir_name(best_cfg["layer_dims"]))
    os.makedirs(best_dir, exist_ok=True)

    if args.dataset == "fashion":
        # Class-colored scatter for fashion
        plot_latent_scatter_classes(latent_vae, class_ids, label_names,
                                    "VAE Latent Space", out_dir=best_dir,
                                    filename="vae_latent_scatter_classes.png")
        plot_latent_scatter_classes(latent_ae, class_ids, label_names,
                                    "Plain AE Latent Space", out_dir=best_dir,
                                    filename="ae_latent_scatter_classes.png")
        # Side-by-side AE vs VAE comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        cmap = plt.cm.tab10
        for ax, lat, title in [(axes[0], latent_ae, "Plain AE"),
                                (axes[1], latent_vae, "VAE")]:
            for c in range(len(label_names)):
                mask = class_ids == c
                if np.any(mask):
                    ax.scatter(lat[mask, 0], lat[mask, 1], c=[cmap(c)],
                               s=8, alpha=0.5, label=label_names[c])
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("z1"); ax.set_ylabel("z2")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, markerscale=2, loc="best")
        plt.tight_layout()
        cmp_path = os.path.join(best_dir, "vae_vs_ae_latent_classes.png")
        plt.savefig(cmp_path, dpi=150)
        plt.close(fig)
        print(f"AE vs VAE comparison -> {cmp_path}")
    else:
        for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
            plot_latent_scatter(latent_ae, labels,
                                "Plain AE Latent Space",
                                out_dir=best_dir,
                                filename=f"ae_latent_scatter{suffix}.png",
                                bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                mode=mode)

        for mode, suffix in [("dots", "_dots"), ("bw", "_bw"), ("color", "_color")]:
            plot_latent_comparison(latent_vae, latent_ae, labels,
                                   bitmaps_bw=bitmaps_bw, bitmaps_color=bitmaps_color,
                                   out_dir=best_dir,
                                   filename=f"vae_vs_ae_latent{suffix}.png",
                                   mode=mode)
    print(f"AE comparison plots -> {best_dir}/")

    # ---- Save best config --------------------------------------------------
    cfg_out = {
        "layer_dims": best_cfg["layer_dims"],
        "lr": best_cfg["lr"],
        "epochs": best_cfg["epochs"],
        "activation": best_cfg.get("activation", "relu"),
        "seed": 42,
        "batch_size": best_cfg.get("batch_size", 20),
        "recon_loss": best_cfg.get("recon_loss", "mse"),
        "dataset": args.dataset,
        "is_color": is_color,
    }
    cfg_path = os.path.join(OUT_DIR, f"{prefix}best_vae_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"Best config -> {cfg_path}")

    print("\n=== train_vae.py complete ===")


if __name__ == "__main__":
    main()
