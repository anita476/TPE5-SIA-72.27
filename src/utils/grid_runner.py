"""grid_runner.py
Parallel grid-search runner.
----------------
* Dispatch each hyperparameter combination to a worker process.
* Collect :class:`~utils.training_run.RunResult` objects as they complete.
* Write the consolidated ``grid_results.csv`` (hyperparams + per-epoch MSE).
* Produce one ``autoencoder_results_<run_id>.png`` plot per run.

"""
from __future__ import annotations

import csv
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.single_run import execute_run, RunResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> RunResult:
    """Top-level picklable wrapper for ProcessPoolExecutor."""
    run_id, params, X, labels, out_dir = args
    return execute_run(run_id, params, X, labels, out_dir)


def _plot_run(result: RunResult, out_dir: str, max_errors: int) -> tuple[str, str, str]:
    """Save three separate figures for *result* and return their paths.

    Files written
    -------------
    loss_<run_id>.png       — MSE per epoch
    latent_<run_id>.png     — 2-D latent space scatter
    errors_<run_id>.png     — per-character pixel-error bar chart
    """
    rid = result.run_id
    title_suffix = (
        f"run {rid} | lr={result.params.get('lr')} "
        f"bs={result.params.get('batch_size')} "
        f"act={result.params.get('activation')} "
        f"seed={result.params.get('seed')}"
    )

    # ── Plot 1: training loss curve ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(result.epoch_losses, linewidth=0.8, color="steelblue")
    ax.set_title(f"Training Loss\n{title_suffix}", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, f"loss_{rid}.png")
    plt.savefig(loss_path, dpi=150)
    plt.close(fig)

    # ── Plot 2: 2-D latent space ──────────────────────────────────────────
    latent = result.latent
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        latent[:, 0], latent[:, 1],
        c=np.arange(len(result.X)), cmap="tab20", s=80, zorder=3,
    )
    for i, label in enumerate(result.labels):
        ax.annotate(
            label, (latent[i, 0], latent[i, 1]),
            fontsize=7, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
        )
    ax.set_title(f"2-D Latent Space\n{title_suffix}", fontsize=8)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    latent_path = os.path.join(out_dir, f"latent_{rid}.png")
    plt.savefig(latent_path, dpi=150)
    plt.close(fig)

    # ── Plot 3: per-character pixel errors ────────────────────────────────
    colours = ["green" if e <= max_errors else "red" for e in result.per_char_errors]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        range(len(result.per_char_errors)), result.per_char_errors,
        color=colours, edgecolor="black", linewidth=0.5,
    )
    ax.axhline(max_errors, color="red", linestyle="--", label=f"Threshold ({max_errors})")
    ax.set_xticks(range(len(result.labels)))
    ax.set_xticklabels(result.labels, rotation=90, fontsize=7)
    ax.set_ylabel("Pixel errors")
    ax.set_title(f"Reconstruction errors\n{title_suffix}", fontsize=8)
    ax.legend()
    plt.tight_layout()
    errors_path = os.path.join(out_dir, f"errors_{rid}.png")
    plt.savefig(errors_path, dpi=150)
    plt.close(fig)

    return loss_path, latent_path, errors_path


def _write_csv(results: list[RunResult], out_dir: str) -> str:
    """Write ``grid_results.csv`` and return its path.

    Schema
    ------
    run_id, <all param keys>, passed, failed, avg_pixel_errors,
    max_pixel_errors, epoch_1_mse, epoch_2_mse, …, epoch_N_mse
    """
    if not results:
        return ""

    csv_path = os.path.join(out_dir, "grid_results.csv")

    # Collect the union of all param keys (preserves insertion order in Py 3.7+)
    param_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.params:
            if k not in seen:
                param_keys.append(k)
                seen.add(k)

    # Determine the maximum number of epochs across all runs.
    max_epochs = max(len(r.epoch_losses) for r in results)

    epoch_headers = [f"epoch_{e+1}_mse" for e in range(max_epochs)]

    fieldnames = (
        ["run_id"]
        + param_keys
        + ["passed", "failed", "avg_pixel_errors", "max_pixel_errors"]
        + epoch_headers
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted(results, key=lambda x: x.run_id):
            row: dict[str, Any] = {"run_id": r.run_id}

            for k in param_keys:
                val = r.params.get(k, "")
                # layer_dims is a list — serialise as a string so CSV stays flat.
                row[k] = str(val) if isinstance(val, list) else val

            row["passed"]           = r.passed
            row["failed"]           = r.failed
            row["avg_pixel_errors"] = round(r.avg_pixel_errors, 4)
            row["max_pixel_errors"] = r.max_pixel_errors

            for e in range(max_epochs):
                col = f"epoch_{e+1}_mse"
                row[col] = round(r.epoch_losses[e], 6) if e < len(r.epoch_losses) else ""

            writer.writerow(row)

    return csv_path


def run_grid(
    combinations: list[dict[str, Any]],
    X:            np.ndarray,
    labels:       list[str],
    out_dir:      str,
    workers:      int = 1,
) -> list[RunResult]:
    """Run all hyperparameter *combinations* and collect results.

    Parameters
    ----------
    combinations:
        List of flat hyperparameter dicts as produced by
        :func:`~utils.config_loader.expand_grid`.
    X:
        Input data array ``(n_samples, n_features)``.
    labels:
        Character labels matching rows of *X*.
    out_dir:
        Root output directory.  Created if absent.  Each run writes its
        ``reconstruction_<id>.txt`` directly here.
    workers:
        Maximum number of parallel worker processes.  Pass ``1`` to run
        sequentially (useful for debugging or when the training loop itself
        is already parallelised inside numpy).

    Returns
    -------
    list[RunResult]
        Results in completion order (not necessarily run_id order).
    """
    os.makedirs(out_dir, exist_ok=True)

    n = len(combinations)
    print(f"\n{'='*60}")
    print(f"Grid search: {n} combination(s), up to {workers} worker(s)")
    print(f"Output dir : {os.path.abspath(out_dir)}")
    print(f"{'='*60}\n")

    work = [
        (run_id, params, X, labels, out_dir)
        for run_id, params in enumerate(combinations, start=1)
    ]

    results: list[RunResult] = []

    if workers == 1:
        for item in work:
            try:
                results.append(_worker(item))
            except Exception:
                run_id = item[0]
                print(f"[run {run_id}] FAILED:\n{traceback.format_exc()}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, item): item[0] for item in work}
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    results.append(future.result())
                except Exception:
                    print(f"[run {run_id}] FAILED:\n{traceback.format_exc()}")

    max_errors = combinations[0].get("max_errors", 1) if combinations else 1

    print(f"\n{'='*60}")
    print("Generating plots …")
    for r in sorted(results, key=lambda x: x.run_id):
        loss_path, latent_path, errors_path = _plot_run(r, out_dir, max_errors)
        r.plot_path = loss_path  # store the first path for backwards compat
        print(f"  [run {r.run_id}] loss   → {loss_path}")
        print(f"  [run {r.run_id}] latent → {latent_path}")
        print(f"  [run {r.run_id}] errors → {errors_path}")


    print("Writing CSV …")
    csv_path = _write_csv(results, out_dir)
    print(f"  → {csv_path}")

    print(f"\n{'='*60}")
    print(f"{'ID':>4}  {'LR':>8}  {'BS':>4}  {'Act':<8}  {'Seed':>6}  "
          f"{'Pass':>5}  {'Fail':>5}  {'AvgErr':>7}  {'FinalMSE':>10}")
    print("-"*60)
    for r in sorted(results, key=lambda x: x.run_id):
        p = r.params
        final_mse = r.epoch_losses[-1] if r.epoch_losses else float("nan")
        print(
            f"{r.run_id:>4}  {p.get('lr', '?'):>8}  {p.get('batch_size', '?'):>4}  "
            f"{str(p.get('activation', '?')):<8}  {p.get('seed', '?'):>6}  "
            f"{r.passed:>5}  {r.failed:>5}  {r.avg_pixel_errors:>7.2f}  {final_mse:>10.6f}"
        )
    print(f"{'='*60}\n")

    return results