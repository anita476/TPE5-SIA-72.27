"""multiseed.py
Run experiments over several random seeds and aggregate the results
(mean ± std) so the plots reflect statistical robustness instead of a single
lucky/unlucky run.

Reuses :func:`utils.single_run.execute_run` (and the autoencoder classes) as
the per-seed training unit, so the same code path is exercised. The number of
parallel worker processes is always caller-controlled (``workers``) — it is a
plain argument, never tied to the host CPU count, so it can be tuned per
machine.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_style  # noqa: F401  (applies the TP plot style on import)
from utils.single_run import execute_run, RunResult
from utils.denoising_eval import evaluate_at_level
from autoencoders.SimpleAutoencoder import SimpleAutoencoder
from autoencoders.DenoisingAutoencoder import DenoisingAutoencoder

ROWS, COLS = 7, 5


# ---------------------------------------------------------------------------
# Generic parallel dispatch
# ---------------------------------------------------------------------------

def _execute_worker(item):
    """Top-level (picklable) wrapper around execute_run for the process pool."""
    return execute_run(*item)


def _dispatch(worker, work, workers):
    """Run *worker* over every item in *work*, sequentially or in a pool."""
    out = []
    if workers <= 1:
        for item in work:
            out.append(worker(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, item) for item in work]
            for future in as_completed(futures):
                out.append(future.result())
    return out


# ---------------------------------------------------------------------------
# 1a — basic autoencoder over N seeds
# ---------------------------------------------------------------------------

def run_seeds(base_params, X, labels, seeds, out_dir, workers=1):
    """Train one model per seed and return the results sorted by seed order.

    ``base_params`` is copied for every seed with ``seed`` overridden and
    per-epoch logging silenced. ``workers`` controls how many runs execute in
    parallel (1 = sequential).
    """
    os.makedirs(out_dir, exist_ok=True)

    work = []
    for run_id, s in enumerate(seeds, start=1):
        params = dict(base_params)
        params["seed"] = s
        params["log_every"] = 0
        params["write_report"] = False
        work.append((run_id, params, X, labels, out_dir))

    results = _dispatch(_execute_worker, work, workers)
    results.sort(key=lambda r: r.run_id)
    return results


# ---------------------------------------------------------------------------
# 1a-2 — variant comparison over N seeds
# ---------------------------------------------------------------------------

def run_variant_seeds(variants, base, X, labels, seeds, out_dir, workers=1):
    """Train every (variant, seed) combination. Returns ``{name: [RunResult]}``."""
    os.makedirs(out_dir, exist_ok=True)

    work, names = [], []
    rid = 0
    for i, var in enumerate(variants, start=1):
        name = var.get("name", f"variant_{i}")
        for s in seeds:
            rid += 1
            params = dict(base)
            params.update({k: v for k, v in var.items() if k != "name"})
            params["seed"] = s
            params["log_every"] = 0
            params["write_report"] = False
            work.append((rid, params, X, labels, out_dir))
            names.append(name)

    raw = _dispatch(_execute_worker, work, workers)
    by_id = {r.run_id: r for r in raw}

    grouped: dict[str, list[RunResult]] = {}
    for (run_id, *_), name in zip(work, names):
        grouped.setdefault(name, []).append(by_id[run_id])
    return grouped


# ---------------------------------------------------------------------------
# 1b — denoising study over N seeds
# ---------------------------------------------------------------------------

def _denoising_seed_worker(args):
    """Train a DAE for one seed and evaluate it at every noise level."""
    seed, base, X, noise_levels, noise_type = args
    ae = DenoisingAutoencoder(
        base["layer_dims"], base["activation"], seed,
        noise_type=noise_type, noise_level=base["noise_level"],
        loss=base.get("loss", "mse"),
    )
    ae.train_and_collect(
        X, base["epochs"], base["lr"], base["batch_size"], 0, base["optimizer"],
        patience=base.get("patience"), min_delta=base.get("min_delta", 1e-6),
    )
    rng = np.random.default_rng(seed)
    noisy, recon, passed = [], [], []
    for lvl in noise_levels:
        res = evaluate_at_level(
            ae, X, lvl, noise_type, rng,
            base.get("threshold", 0.5), base.get("max_errors", 1),
        )
        noisy.append(res["avg_noisy_err"])
        recon.append(res["avg_err"])
        passed.append(res["passed"])
    return {"seed": seed, "noise_type": noise_type, "levels": list(noise_levels),
            "noisy": noisy, "recon": recon, "passed": passed}


def run_denoising_seeds(base, X, noise_levels, noise_type, seeds, workers=1):
    """Return a per-seed list of denoising metrics for a single noise type."""
    work = [(s, base, X, noise_levels, noise_type) for s in seeds]
    return _dispatch(_denoising_seed_worker, work, workers)


def run_noise_type_seeds(base, X, noise_levels, noise_types, seeds, workers=1):
    """Return ``{noise_type: [per-seed metrics]}`` across types and seeds."""
    work = [(s, base, X, noise_levels, nt) for nt in noise_types for s in seeds]
    raw = _dispatch(_denoising_seed_worker, work, workers)
    grouped: dict[str, list[dict]] = {}
    for d in raw:
        grouped.setdefault(d["noise_type"], []).append(d)
    return grouped


# ---------------------------------------------------------------------------
# 1a-4 — trained models over N seeds (for generation visuals)
# ---------------------------------------------------------------------------

def _simple_model_worker(args):
    seed, base, X = args
    ae = SimpleAutoencoder(base["layer_dims"], base["activation"], seed,
                           loss=base.get("loss", "mse"))
    ae.train_and_collect(
        X, base["epochs"], base["lr"], base["batch_size"], 0, base["optimizer"],
        patience=base.get("patience"), min_delta=base.get("min_delta", 1e-6),
    )
    return seed, ae


def run_model_seeds(base, X, seeds, workers=1):
    """Train a SimpleAutoencoder per seed and return ``[(seed, model)]``."""
    work = [(s, base, X) for s in seeds]
    out = _dispatch(_simple_model_worker, work, workers)
    out.sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def stack_losses(results):
    """Stack per-epoch losses into ``(n_seeds, n_epochs)``.

    Runs may early-stop at different epochs, so curves are truncated to the
    shortest length to keep the band well-defined at every x.
    """
    min_len = min(len(r.epoch_losses) for r in results)
    return np.array([r.epoch_losses[:min_len] for r in results])


def stack_per_char(results):
    """Stack per-character pixel errors into ``(n_seeds, n_chars)``."""
    return np.array([r.per_char_errors for r in results])


# ---------------------------------------------------------------------------
# Plots — quantitative (mean ± std)
# ---------------------------------------------------------------------------

def plot_loss_band(results, out_path, title, logscale=True, color="steelblue"):
    arr = stack_losses(results)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    x = np.arange(len(mean))
    loss_name = str(results[0].params.get("loss", "mse")).upper()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, mean, linewidth=1.0, color=color, label="media")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25,
                    label="±1 desv.")
    if logscale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_name + (" (log scale)" if logscale else ""))
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_char_band(results, max_errors, out_path, title):
    E = stack_per_char(results)
    mean, std = E.mean(axis=0), E.std(axis=0)
    labels = results[0].labels
    colours = ["seagreen" if m <= max_errors else "indianred" for m in mean]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(mean)), mean, yerr=std, capsize=2,
           color=colours, edgecolor="black", linewidth=0.5,
           error_kw={"elinewidth": 0.7})
    ax.axhline(max_errors, color="red", linestyle="--",
               label=f"Umbral ({max_errors})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Pixel errors (media ± desv.)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_compare_loss_bands(results_by_variant, out_path,
                            title="Training loss comparison (media ± desv.)"):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    first = next(iter(results_by_variant.values()))
    loss_name = str(first[0].params.get("loss", "mse")).upper()
    for i, (name, results) in enumerate(results_by_variant.items()):
        arr = stack_losses(results)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        x = np.arange(len(mean))
        c = cmap(i % 10)
        ax.plot(x, mean, linewidth=1.0, color=c, label=name)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{loss_name} (log scale)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# Obs!!! Change for ordering according to necessity !
VARIANT_KEY_ORDER = [
    "activation",
    "optimizer",
    "learning_rate",
    "layers",
    "hidden_size",
    "dropout",
    "batch_size",
    "epochs",
]

def plot_compare_metric_bars(results_by_variant, total, out_path, max_errors):
    def sort_key(name):
        name_lower = name.lower()
        for i, key in enumerate(VARIANT_KEY_ORDER):
            if key.replace("_", "") in name_lower.replace("_", "").replace(" ", ""):
                return (i, name)
        return len(VARIANT_KEY_ORDER), name

    names = sorted(results_by_variant.keys(), key=sort_key)

    passed_mean = [np.mean([r.passed for r in results_by_variant[n]]) for n in names]
    passed_std  = [np.std( [r.passed for r in results_by_variant[n]]) for n in names]
    err_mean    = [np.mean([r.avg_pixel_errors for r in results_by_variant[n]]) for n in names]
    err_std     = [np.std( [r.avg_pixel_errors for r in results_by_variant[n]]) for n in names]

    passed_pct_mean = [m / total * 100 for m in passed_mean]
    passed_pct_std  = [s / total * 100 for s in passed_std]

    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Left chart: % characters learned ──────────────────────────────────
    bars1 = ax1.bar(
        x, passed_pct_mean, yerr=passed_pct_std, capsize=3,
        color="seagreen", edgecolor="black", error_kw={"elinewidth": 0.8}
    )

    for bar, mean_val, std_val in zip(bars1, passed_pct_mean, passed_pct_std):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(passed_pct_std) if passed_pct_std else 0) + 1,
            f"{mean_val:.1f}%\n±{std_val:.1f}%",
            ha="center", va="bottom", fontsize=7
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Passed (% of characters)")
    ax1.set_ylim(0, 115)
    ax1.set_title(
        "Characters learned\n"
        f"(total: {total} chars · passed if pixel error ≤ {max_errors})",
        fontsize=10
    )

    # ── Right chart: avg pixel error ──────────────────────────────────────
    bars2 = ax2.bar(
        x, err_mean, yerr=err_std, capsize=3,
        color="indianred", edgecolor="black", error_kw={"elinewidth": 0.8}
    )

    for bar, mean_val, std_val in zip(bars2, err_mean, err_std):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(err_std) if err_std else 0) + 0.001,
            f"{mean_val:.3f}\n±{std_val:.3f}",
            ha="center", va="bottom", fontsize=7
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Avg pixel error (mean ± std)")
    ax2.set_title(f"Average reconstruction error  [max errors = {total}]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_denoising_band(per_seed, out_path,
                        title="Denoising capability (media ± desv.)"):
    levels = per_seed[0]["levels"]
    noisy = np.array([d["noisy"] for d in per_seed])
    recon = np.array([d["recon"] for d in per_seed])

    fig, ax = plt.subplots(figsize=(7, 5))
    nm, ns = noisy.mean(0), noisy.std(0)
    rm, rs = recon.mean(0), recon.std(0)
    ax.plot(levels, nm, "o--", color="gray", label="Noisy input")
    ax.fill_between(levels, nm - ns, nm + ns, color="gray", alpha=0.2)
    ax.plot(levels, rm, "o-", color="steelblue", label="DAE output")
    ax.fill_between(levels, rm - rs, rm + rs, color="steelblue", alpha=0.2)
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Mean pixel error (vs clean)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_noise_compare_bands(grouped, out_path,
                             title="Denoising by noise type (media ± desv.)"):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, (ntype, per_seed) in enumerate(grouped.items()):
        levels = per_seed[0]["levels"]
        recon = np.array([d["recon"] for d in per_seed])
        m, s = recon.mean(0), recon.std(0)
        c = cmap(i % 10)
        ax.plot(levels, m, "o-", color=c, label=ntype)
        ax.fill_between(levels, m - s, m + s, color=c, alpha=0.18)
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Mean recon pixel error (vs clean)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plots — qualitative (per-seed grids)
# ---------------------------------------------------------------------------

def plot_latent_seeds(results, out_path, n_show=6, title="Espacio latente por seed"):
    """Lay out the latent scatter for several seeds side by side."""
    subset = results[:n_show]
    n = len(subset)
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2),
                             squeeze=False)
    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        r = subset[idx]
        z = r.latent
        ax.scatter(z[:, 0], z[:, 1], c=np.arange(len(z)), cmap="tab20",
                   s=40, zorder=3)
        for i, label in enumerate(r.labels):
            ax.annotate(label, (z[i, 0], z[i, 1]), fontsize=5, ha="center",
                        va="bottom", xytext=(0, 3), textcoords="offset points")
        ax.set_title(f"seed={r.params.get('seed')}  ({r.passed}/{len(z)})",
                     fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_interpolation_seeds(models, X, labels, idx_a, idx_b, out_path,
                             steps=9, threshold=0.5):
    """Rows = seeds, columns = interpolation steps between two characters."""
    n = len(models)
    alphas = np.linspace(0.0, 1.0, steps)
    fig, axes = plt.subplots(n, steps, figsize=(steps * 1.1, n * 1.35),
                             squeeze=False)
    for row, (seed, ae) in enumerate(models):
        z = ae.encode(X)
        za, zb = z[idx_a], z[idx_b]
        for k, a in enumerate(alphas):
            point = ((1 - a) * za + a * zb).reshape(1, -1).astype(np.float32)
            glyph = (ae.decode(point)[0] >= threshold).astype(np.float32)
            ax = axes[row, k]
            ax.imshow(glyph.reshape(ROWS, COLS), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{a:.2f}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"seed {seed}", fontsize=8)

    fig.suptitle(
        f"Interpolación '{labels[idx_a]}' → '{labels[idx_b]}' por seed "
        f"(letras nuevas en los pasos intermedios)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_generated_point_seeds(models, X, out_path, threshold=0.5):
    """One generated glyph (latent centroid) per seed, shown side by side."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 2.0), squeeze=False)
    for i, (seed, ae) in enumerate(models):
        z = ae.encode(X)
        centroid = z.mean(axis=0).reshape(1, -1).astype(np.float32)
        glyph = (ae.decode(centroid)[0] >= threshold).astype(np.float32)
        ax = axes[0, i]
        ax.imshow(glyph.reshape(ROWS, COLS), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"seed {seed}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Carácter generado desde el centroide del latente, por seed",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Scalar summary
# ---------------------------------------------------------------------------

def summarise(results):
    """Return scalar mean/std summaries across seeds."""
    passed = np.array([r.passed for r in results], dtype=float)
    avg_err = np.array([r.avg_pixel_errors for r in results], dtype=float)
    total = len(results[0].per_char_errors)
    return {
        "n_seeds": len(results),
        "total_chars": total,
        "passed_mean": float(passed.mean()),
        "passed_std": float(passed.std()),
        "passed_min": float(passed.min()),
        "passed_max": float(passed.max()),
        "avg_err_mean": float(avg_err.mean()),
        "avg_err_std": float(avg_err.std()),
    }
